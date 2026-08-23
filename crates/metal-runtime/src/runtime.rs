//! objc2-metal runtime: device, Metal 4 encode path, pipeline cache, buffer pool,
//! persistent argument-table pattern.
//!
//! Encode is **Metal 4 only**: one `MTL4CommandBuffer` per step with
//! argument-table binds, a bump-allocated const arena (~1 MiB), residency
//! registry, and SharedEvent sync. Steady-state work never host-waits except
//! at log / loss / eval boundaries via [`GpuRuntime::synchronize`].
//!
//! Audit 4 lessons preserved: cold-buffer recycle + `removeAllocation` after CB
//! complete; one compute encoder packed across `with_binder` calls (P1);
//! working-set probe; no host-zero mid-CB.

use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSData, NSRange, NSString, NSURL};
use objc2::ClassType;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputeCommandEncoder,
    MTL4ComputePipelineDescriptor, MTL4CommandQueue, MTL4CounterHeap, MTL4CounterHeapDescriptor,
    MTL4CounterHeapType, MTL4IndirectCommandBufferSupportState, MTL4LibraryFunctionDescriptor,
    MTL4TimestampHeapEntry, MTL4VisibilityOptions, MTLAllocation, MTLBuffer,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent, MTLSize,
    MTLStages,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};

/// Const arena for Metal 4 scalar binds (distinct offsets; reset after sync).
// 31B dense decode packs hundreds of binder consts across mid-commits within a
// token before a waiting sync; 1 MiB exhausted mid-token. 16 MiB covers full
// product shapes with headroom (still tiny vs Hot weight residency).
const METAL4_CONST_ARENA_BYTES: usize = 16 * 1024 * 1024;

/// Default pool freelist cap (~2 GiB of cached slabs).
const DEFAULT_POOL_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Residency / recycle policy for pooled buffers (Audit 4 P0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// Mid-step temps — recycle + removeAllocation after CB complete.
    Cold,
    /// Weights / optim / long-lived — stay resident for the run.
    Hot,
    /// Bump slab — Drop does not recycle (cursor reset after sync).
    Bump,
}

/// Probed device memory budget (logged in train banner).
#[derive(Clone, Copy, Debug)]
pub struct DeviceMemoryInfo {
    pub recommended_working_set: u64,
    pub memory_size: u64,
    pub wired_budget: u64,
    pub pool_cache_cap: usize,
}

/// Precision mode for the training hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Parity / `--f32`: all storage+compute f32.
    F32,
    /// Phase 4 default: bf16 storage/compute, f32 accum (GEMM/softmax/RMS/loss/optim).
    Bf16,
}

struct PipelineCache {
    map: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        overlays: &[Retained<ProtocolObject<dyn MTLLibrary>>],
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        let icb = crate::decode_icb::icb_pipelines_enabled();
        let key = if icb {
            format!("icb:{name}")
        } else {
            name.to_string()
        };
        if let Some(p) = self.map.get(&key) {
            return Ok(p.clone());
        }
        let fname = NSString::from_str(name);
        let containing: &ProtocolObject<dyn MTLLibrary> =
            if library.newFunctionWithName(&fname).is_some() {
                library
            } else if let Some(lib) = overlays
                .iter()
                .find(|lib| lib.newFunctionWithName(&fname).is_some())
            {
                lib
            } else {
                return Err(format!("kernel '{name}' not found in metallib"));
            };

        let pipeline = if icb {
            let compiler_desc = MTL4CompilerDescriptor::new();
            let compiler = device
                .newCompilerWithDescriptor_error(&compiler_desc)
                .map_err(|e| format!("MTL4Compiler: {e}"))?;
            let func_desc = MTL4LibraryFunctionDescriptor::new();
            func_desc.setName(Some(&fname));
            func_desc.setLibrary(Some(containing));
            let pipe_desc = MTL4ComputePipelineDescriptor::new();
            pipe_desc.setComputeFunctionDescriptor(Some(func_desc.as_super()));
            pipe_desc
                .setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);
            let p = compiler
                .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipe_desc, None)
                .map_err(|e| format!("ICB pipeline '{name}': {e}"))?;
            if !p.supportIndirectCommandBuffers() {
                return Err(format!(
                    "ICB pipeline '{name}' supportIndirectCommandBuffers=false"
                ));
            }
            p
        } else {
            let func = containing
                .newFunctionWithName(&fname)
                .ok_or_else(|| format!("kernel '{name}' not found in metallib"))?;
            device
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| format!("pipeline '{name}': {e}"))?
        };
        self.map.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

struct BufferPool {
    freelist: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    cached_bytes: usize,
    max_cache_bytes: usize,
}

impl BufferPool {
    fn new(max_cache_bytes: usize) -> Self {
        Self {
            freelist: HashMap::new(),
            cached_bytes: 0,
            max_cache_bytes,
        }
    }

    fn bucket(nbytes: usize) -> usize {
        nbytes.next_power_of_two().max(256)
    }

    fn alloc(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        nbytes: usize,
    ) -> Result<(Retained<ProtocolObject<dyn MTLBuffer>>, bool), String> {
        let key = Self::bucket(nbytes);
        if let Some(v) = self.freelist.get_mut(&key) {
            if let Some(b) = v.pop() {
                self.cached_bytes = self.cached_bytes.saturating_sub(key);
                // true = came from freelist (already resided previously; re-add)
                return Ok((b, true));
            }
        }
        let b = device
            .newBufferWithLength_options(key, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("newBuffer({key}) failed"))?;
        Ok((b, false))
    }

    fn recycle(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let key = Self::bucket(buffer.length() as usize);
        if self.cached_bytes + key > self.max_cache_bytes {
            // Drop buffer (let ARC release) — over cache cap.
            return;
        }
        self.cached_bytes += key;
        self.freelist.entry(key).or_default().push(buffer);
    }

    fn set_max_cache_bytes(&mut self, max_cache_bytes: usize) {
        self.max_cache_bytes = max_cache_bytes;
        // Trim if over (drop largest buckets first).
        if self.cached_bytes <= max_cache_bytes {
            return;
        }
        let mut keys: Vec<usize> = self.freelist.keys().copied().collect();
        keys.sort_unstable_by(|a, b| b.cmp(a));
        for key in keys {
            while self.cached_bytes > max_cache_bytes {
                let Some(v) = self.freelist.get_mut(&key) else {
                    break;
                };
                if v.pop().is_none() {
                    break;
                }
                self.cached_bytes = self.cached_bytes.saturating_sub(key);
            }
        }
    }
}

pub struct PersistentArgumentTable {
    pub table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    pub max_buffers: u64,
}

/// One MTL4 allocator slot (ping-pong for mid-token commit without wait).
struct AllocatorSlot {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    /// SharedEvent value that must land before reset+reuse; 0 = free.
    in_flight: u64,
}

/// Metal 4 encode package (queue / dual allocators / argument table / CounterHeap).
pub struct Metal4EncodePackage {
    pub queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    /// Dual allocators — GPU runs one CB while host encodes the next.
    allocators: Mutex<[AllocatorSlot; 2]>,
    active_alloc: Mutex<usize>,
    /// Legacy alias: allocator 0 (tests / callers that expected a single handle).
    pub allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    pub command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub argument_table: PersistentArgumentTable,
    pub counter_heap: Option<Retained<ProtocolObject<dyn MTL4CounterHeap>>>,
    pub residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    pub shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Scratch for scalar `[[buffer(N)]]` constants (M4 has no setBytes).
    /// ~1 MiB bump arena; cursor advances per const pack, reset after sync.
    pub const_staging: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub const_cursor: Mutex<usize>,
    event_value: Mutex<u64>,
    /// Allocations registered into `residency` (debug / telemetry).
    pub residency_count: Mutex<usize>,
}

/// Soft mid-token commit threshold (dispatches since last commit).
/// Default off (`0`). Enable with `METAL_RUNTIME_MID_COMMIT=N` (e.g. 128–256).
/// Free-allocator pick avoids the wait-storm when the peer slot is still busy.
fn mid_commit_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("METAL_RUNTIME_MID_COMMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Open Metal 4 command buffer for the current step (one allocator CB at a time).
/// Audit 4 P1: keep a single compute encoder open across `with_binder` calls.
struct ActiveMetal4Batch {
    dispatches: usize,
    /// Dispatches since last commit (mid-token overlap).
    since_commit: usize,
    cb_open: bool,
    stamped_t0: bool,
    encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    alloc_idx: usize,
}

struct BumpState {
    buffer: crate::tensor::GpuBuffer,
    cursor: usize,
    capacity: usize,
}

/// Shared GPU runtime (Metal 4 encode required).
pub struct GpuRuntime {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// Extra metallibs (e.g. gemma-metal overlay) searched after [`Self::library`].
    overlay_libraries: Mutex<Vec<Retained<ProtocolObject<dyn MTLLibrary>>>>,
    /// Metal 4 encode package (required; init fails if unavailable).
    pub metal4: Metal4EncodePackage,
    pipelines: Mutex<PipelineCache>,
    pool: Mutex<BufferPool>,
    /// Per-step bump arena (single slab). Reset only after GPU work that used
    /// bump views has completed (or at the start of a step following a sync).
    bump: Mutex<Option<BumpState>>,
    has_tensorops: bool,
    /// When true, kernels accumulate into one CB until [`Self::synchronize`] / [`Self::commit`].
    async_encode: Mutex<bool>,
    active_m4: Mutex<Option<ActiveMetal4Batch>>,
    /// Last CounterHeap (t0, t1) resolved at synchronize.
    last_m4_stamps: Mutex<Option<(u64, u64)>>,
    /// Dispatch counter since last commit (for fusion/telemetry).
    pub dispatch_count: Mutex<usize>,
    precision: Mutex<PrecisionMode>,
    /// Phase H bridge: TensorOps f32 GEMM with `relaxed_precision` (tf32-class).
    /// Off by default so f32 goldens stay exact; enable via `--tf32` / `set_relaxed_precision`.
    relaxed_precision: Mutex<bool>,
    /// Prefer TensorOps multi-block flash probe over simdgroup FA-2 (`--flash-tensorops`).
    flash_tensorops: Mutex<bool>,
    /// Uncommitted residency adds/removes — flushed before encode / synchronize.
    residency_dirty: Mutex<bool>,
    /// Cold buffers whose last Arc dropped mid-step; recycled after CB wait.
    pending_cold_recycle: Mutex<Vec<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>>,
    /// Self weak handle so Drop on pooled buffers can schedule recycle.
    self_weak: Mutex<Weak<GpuRuntime>>,
    /// Probed working-set / wired budget (P0b).
    memory_info: Mutex<DeviceMemoryInfo>,
}

impl GpuRuntime {
    pub fn new() -> Result<Arc<Self>, String> {
        Self::from_metallib_path(Path::new(crate::metallib_path()))
    }

    /// Inference decode runtime: no CounterHeap timestamps (host encode tax).
    pub fn new_inference() -> Result<Arc<Self>, String> {
        Self::from_metallib_path_opts(Path::new(crate::metallib_path()), /*timestamps*/ false)
    }

    pub fn from_metallib_path(path: &Path) -> Result<Arc<Self>, String> {
        Self::from_metallib_path_opts(path, /*timestamps*/ true)
    }

    pub fn from_metallib_path_opts(path: &Path, timestamps: bool) -> Result<Arc<Self>, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_string())?;

        let path_str = path
            .to_str()
            .ok_or_else(|| format!("non-utf8 metallib path: {path:?}"))?;
        if !path.exists() {
            return Err(format!(
                "metallib missing at {path_str} (build.rs AOT failed?)"
            ));
        }
        let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
        let library = device
            .newLibraryWithURL_error(&url)
            .map_err(|e| format!("load metallib: {e}"))?;

        // Metal 4 encode package is required (Metal4-only doctrine).
        let metal4 = try_init_metal4(&device, timestamps).map_err(|err| {
            format!(
                "Metal 4 encode package unavailable ({err}); metal-runtime requires Metal 4"
            )
        })?;

        let has_tensorops = library
            .newFunctionWithName(&NSString::from_str("matmul2d_tensorops_f32"))
            .is_some();

        let recommended = device.recommendedMaxWorkingSetSize();
        let memory_size = probe_system_memory_size();
        let wired_budget = ((recommended as f64) * 0.9) as u64;
        let mem_info = DeviceMemoryInfo {
            recommended_working_set: recommended,
            memory_size,
            wired_budget,
            pool_cache_cap: DEFAULT_POOL_CACHE_BYTES,
        };

        let rt = Arc::new(Self {
            device,
            library,
            overlay_libraries: Mutex::new(Vec::new()),
            metal4,
            pipelines: Mutex::new(PipelineCache::new()),
            pool: Mutex::new(BufferPool::new(DEFAULT_POOL_CACHE_BYTES)),
            bump: Mutex::new(None),
            has_tensorops,
            async_encode: Mutex::new(false),
            active_m4: Mutex::new(None),
            last_m4_stamps: Mutex::new(None),
            dispatch_count: Mutex::new(0),
            precision: Mutex::new(PrecisionMode::F32),
            relaxed_precision: Mutex::new(false),
            flash_tensorops: Mutex::new(false),
            residency_dirty: Mutex::new(false),
            pending_cold_recycle: Mutex::new(Vec::new()),
            self_weak: Mutex::new(Weak::new()),
            memory_info: Mutex::new(mem_info),
        });
        if let Ok(mut w) = rt.self_weak.lock() {
            *w = Arc::downgrade(&rt);
        }
        Ok(rt)
    }

    pub fn has_tensorops(&self) -> bool {
        self.has_tensorops
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn set_precision(&self, mode: PrecisionMode) {
        *self.precision.lock().unwrap() = mode;
    }

    pub fn precision(&self) -> PrecisionMode {
        *self.precision.lock().unwrap()
    }

    /// Opt into TensorOps f32 `relaxed_precision` (tf32-class) GEMMs. Ignored when
    /// [`PrecisionMode::Bf16`] (bf16 path takes precedence) or TensorOps is absent.
    pub fn set_relaxed_precision(&self, on: bool) {
        *self.relaxed_precision.lock().unwrap() = on;
    }

    pub fn relaxed_precision(&self) -> bool {
        *self.relaxed_precision.lock().unwrap()
    }

    pub fn set_flash_tensorops(&self, on: bool) {
        *self.flash_tensorops.lock().unwrap() = on;
    }

    pub fn flash_tensorops(&self) -> bool {
        *self.flash_tensorops.lock().unwrap()
    }

    pub fn memory_info(&self) -> DeviceMemoryInfo {
        *self.memory_info.lock().unwrap()
    }

    /// Cap freelist cache bytes (CLI `--pool-cache-mb`).
    pub fn set_pool_cache_cap_bytes(&self, bytes: usize) {
        if let Ok(mut info) = self.memory_info.lock() {
            info.pool_cache_cap = bytes;
        }
        if let Ok(mut pool) = self.pool.lock() {
            pool.set_max_cache_bytes(bytes);
        }
    }

    /// Override wired budget fraction of `recommendedMaxWorkingSetSize` (logged only;
    /// raising the system `iogpu.wired_limit_mb` still requires sysctl).
    pub fn set_wired_fraction(&self, fraction: f64) {
        let frac = fraction.clamp(0.5, 0.95);
        if let Ok(mut info) = self.memory_info.lock() {
            info.wired_budget = ((info.recommended_working_set as f64) * frac) as u64;
        }
    }

    /// Last CounterHeap (t0, t1) from the most recent Metal 4 synchronize, if any.
    pub fn take_metal4_stamps(&self) -> Option<(u64, u64)> {
        self.last_m4_stamps.lock().ok().and_then(|mut g| g.take())
    }

    /// Register a buffer in the Metal 4 residency set (deferred commit).
    pub fn register_residency(&self, buf: &ProtocolObject<dyn MTLBuffer>) {
        self.register_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(buf));
    }

    /// Register any [`MTLAllocation`] (buffers, ICB, …) in the residency set.
    pub fn register_allocation(&self, alloc: &ProtocolObject<dyn MTLAllocation>) {
        let m4 = &self.metal4;
        m4.residency.addAllocation(alloc);
        if let Ok(mut c) = m4.residency_count.lock() {
            *c += 1;
        }
        if let Ok(mut d) = self.residency_dirty.lock() {
            *d = true;
        }
    }

    /// Mark allocation for removal on next residency commit (after CB complete).
    pub fn unregister_residency(&self, buf: &ProtocolObject<dyn MTLBuffer>) {
        let m4 = &self.metal4;
        m4.residency
            .removeAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(buf));
        if let Ok(mut c) = m4.residency_count.lock() {
            *c = c.saturating_sub(1);
        }
        if let Ok(mut d) = self.residency_dirty.lock() {
            *d = true;
        }
    }

    /// Called from [`crate::tensor::PooledBuffer`] Drop for cold temps.
    pub(crate) fn schedule_cold_recycle(
        &self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        nbytes: usize,
    ) {
        if let Ok(mut q) = self.pending_cold_recycle.lock() {
            q.push((buffer, nbytes));
        }
    }

    /// After GPU catch-up: remove cold allocs from residency and return to freelist.
    fn drain_cold_recycles(&self) {
        let pending = if let Ok(mut q) = self.pending_cold_recycle.lock() {
            std::mem::take(&mut *q)
        } else {
            return;
        };
        if pending.is_empty() {
            return;
        }
        for (buf, _nbytes) in pending {
            self.unregister_residency(&buf);
            if let Ok(mut pool) = self.pool.lock() {
                pool.recycle(buf);
            }
        }
        self.flush_residency();
    }

    /// Commit pending residency adds/removes and request residency (batched).
    pub fn flush_residency(&self) {
        let dirty = self
            .residency_dirty
            .lock()
            .map(|g| *g)
            .unwrap_or(false);
        if !dirty {
            return;
        }
        crate::infer_trace::on_residency_flush();
        let m4 = &self.metal4;
        m4.residency.commit();
        m4.residency.requestResidency();
        if let Ok(mut d) = self.residency_dirty.lock() {
            *d = false;
        }
        // try_lock: avoid re-entrancy when called under `active_m4`.
        if let Ok(guard) = self.active_m4.try_lock() {
            if guard.as_ref().map(|b| b.cb_open).unwrap_or(false) {
                m4.command_buffer.useResidencySet(&m4.residency);
            }
        }
    }

    /// Enable multi-kernel command buffers (training hot path).
    pub fn set_async_encode(&self, on: bool) -> Result<(), String> {
        if !on {
            self.synchronize()?;
        }
        *self.async_encode.lock().map_err(|e| e.to_string())? = on;
        Ok(())
    }

    pub fn async_encode_enabled(&self) -> bool {
        *self.async_encode.lock().unwrap()
    }

    pub fn take_dispatch_count(&self) -> usize {
        let mut g = self.dispatch_count.lock().unwrap();
        let n = *g;
        *g = 0;
        n
    }

    /// Register an additional metallib (Gemma kernels, etc.). Pipeline names
    /// must be unique across primary + overlays.
    pub fn add_metallib(&self, path: &Path) -> Result<(), String> {
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("non-utf8 metallib path: {path:?}"))?;
        if !path.exists() {
            return Err(format!("metallib missing at {path_str}"));
        }
        let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
        let library = self
            .device
            .newLibraryWithURL_error(&url)
            .map_err(|e| format!("load overlay metallib: {e}"))?;
        let mut libs = self.overlay_libraries.lock().map_err(|e| e.to_string())?;
        libs.push(library);
        Ok(())
    }

    pub fn pipeline(
        &self,
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        // DecodeIcb binder-nop prep: PSO is unused (with_binder is a no-op). Skip
        // dual-mutex HashMap lookup — largest host bind-tax cut on replay steps.
        if crate::decode_icb::binder_encode_nop() {
            return self.pipeline_nop_standin();
        }
        // Cache hit without holding overlay lock or allocating a key String.
        let icb = crate::decode_icb::icb_pipelines_enabled();
        {
            let cache = self.pipelines.lock().map_err(|e| e.to_string())?;
            if !icb {
                if let Some(p) = cache.map.get(name) {
                    return Ok(p.clone());
                }
            } else {
                let key = format!("icb:{name}");
                if let Some(p) = cache.map.get(&key) {
                    return Ok(p.clone());
                }
            }
        }
        let overlays = self.overlay_libraries.lock().map_err(|e| e.to_string())?;
        let mut cache = self.pipelines.lock().map_err(|e| e.to_string())?;
        cache.get_or_create(&self.device, &self.library, &overlays, name)
    }

    /// Cheap stand-in PSO for binder-nop replay prep (discarded; never encoded).
    fn pipeline_nop_standin(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        thread_local! {
            static STANDIN: std::cell::RefCell<
                Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
            > = const { std::cell::RefCell::new(None) };
        }
        STANDIN.with(|slot| {
            if let Some(p) = slot.borrow().as_ref() {
                return Ok(p.clone());
            }
            let cache = self.pipelines.lock().map_err(|e| e.to_string())?;
            let p = cache
                .map
                .values()
                .next()
                .cloned()
                .ok_or_else(|| {
                    "pipeline_nop_standin: empty cache (binder-nop before any live encode?)"
                        .to_string()
                })?;
            *slot.borrow_mut() = Some(p.clone());
            Ok(p)
        })
    }

    /// Snapshot of overlay metallibs (for ICB pipeline construction).
    pub fn overlay_libraries_snapshot(
        &self,
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLLibrary>>>, String> {
        let overlays = self.overlay_libraries.lock().map_err(|e| e.to_string())?;
        Ok(overlays.clone())
    }

    pub fn alloc_buffer(&self, nbytes: usize) -> Result<crate::tensor::GpuBuffer, String> {
        self.alloc_buffer_kind(nbytes, BufferKind::Cold)
    }

    pub fn alloc_buffer_hot(&self, nbytes: usize) -> Result<crate::tensor::GpuBuffer, String> {
        self.alloc_buffer_kind(nbytes, BufferKind::Hot)
    }

    pub fn alloc_buffer_kind(
        &self,
        nbytes: usize,
        kind: BufferKind,
    ) -> Result<crate::tensor::GpuBuffer, String> {
        if kind == BufferKind::Cold {
            crate::infer_trace::on_cold_alloc();
        }
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let (buffer, _from_pool) = pool.alloc(&self.device, nbytes)?;
        // Always (re)register — freelist buffers were removed on recycle.
        self.register_residency(&buffer);
        let weak = self
            .self_weak
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        Ok(crate::tensor::GpuBuffer {
            inner: Arc::new(crate::tensor::PooledBuffer {
                buffer,
                nbytes,
                kind,
                runtime: weak,
            }),
        })
    }

    pub fn recycle_buffer(&self, buf: crate::tensor::GpuBuffer) {
        // Last Arc Drop schedules cold recycle after CB complete.
        drop(buf);
    }

    /// Ensure a bump slab of at least `capacity` bytes (power-of-two bucketed).
    pub fn ensure_bump(self: &Arc<Self>, capacity: usize) -> Result<(), String> {
        let cap = capacity.next_power_of_two().max(256);
        let mut bump = self.bump.lock().map_err(|e| e.to_string())?;
        if let Some(b) = bump.as_ref() {
            if b.capacity >= cap {
                return Ok(());
            }
        }
        // Recycle the previous slab into the pool before replacing.
        if let Some(old) = bump.take() {
            self.unregister_residency(old.buffer.metal());
            if let Ok(mut pool) = self.pool.lock() {
                pool.recycle(old.buffer.inner.buffer.clone());
            }
        }
        let buffer = self.alloc_buffer_kind(cap, BufferKind::Bump)?;
        *bump = Some(BumpState {
            buffer,
            cursor: 0,
            capacity: cap,
        });
        Ok(())
    }

    /// Sub-allocate a zeroed f32 tensor from the bump slab (view with byte_offset).
    pub fn bump_alloc_f32(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        let numel: usize = shape.iter().product();
        let nbytes = numel * 4;
        let mut bump = self.bump.lock().map_err(|e| e.to_string())?;
        let state = bump
            .as_mut()
            .ok_or_else(|| "bump arena not initialized; call ensure_bump first".to_string())?;
        // Align to 16 bytes for TensorOps.
        let align = 16;
        let cursor = (state.cursor + align - 1) & !(align - 1);
        if cursor + nbytes > state.capacity {
            return Err(format!(
                "bump arena exhausted: need {} more bytes (cursor={cursor}, cap={})",
                nbytes, state.capacity
            ));
        }
        let off = cursor;
        state.cursor = cursor + nbytes;
        // Zero the logical window on the host (unified memory).
        {
            let ptr = state.buffer.metal().contents().as_ptr() as *mut u8;
            unsafe {
                std::ptr::write_bytes(ptr.add(off), 0, nbytes);
            }
        }
        Ok(crate::tensor::Tensor {
            buffer: state.buffer.clone(),
            shape: shape.to_vec(),
            dtype: crate::tensor::DType::F32,
            byte_offset: off,
            runtime: Arc::clone(self),
        })
    }

    /// Reset bump cursor to 0. Caller must ensure no in-flight GPU work still
    /// references bump views (typically after [`Self::synchronize`]).
    pub fn bump_reset(&self) {
        if let Ok(mut bump) = self.bump.lock() {
            if let Some(b) = bump.as_mut() {
                b.cursor = 0;
            }
        }
    }

    pub fn bump_enabled(&self) -> bool {
        self.bump.lock().ok().map(|b| b.is_some()).unwrap_or(false)
    }

    /// Prefer bump slab when initialized; otherwise pool-alloc a fresh tensor.
    pub fn alloc_temp_f32(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        if self.bump_enabled() {
            match self.bump_alloc_f32(shape) {
                Ok(t) => return Ok(t),
                Err(_) => {} // fall through on exhaustion
            }
        }
        self.alloc_tensor_f32(shape)
    }

    pub fn alloc_tensor_f32(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_f32_kind(shape, BufferKind::Cold)
    }

    /// Persistent weights / grads / optim / EMA — stay in residency (no cold recycle).
    pub fn alloc_tensor_f32_hot(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_f32_kind(shape, BufferKind::Hot)
    }

    fn alloc_tensor_f32_kind(
        self: &Arc<Self>,
        shape: &[usize],
        kind: BufferKind,
    ) -> Result<crate::tensor::Tensor, String> {
        let numel: usize = shape.iter().product();
        let buf = self.alloc_buffer_kind(numel * 4, kind)?;
        buf.zero();
        Ok(crate::tensor::Tensor {
            buffer: buf,
            shape: shape.to_vec(),
            dtype: crate::tensor::DType::F32,
            byte_offset: 0,
            runtime: Arc::clone(self),
        })
    }

    pub fn alloc_tensor_bf16(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_bf16_kind(shape, BufferKind::Cold)
    }

    pub fn alloc_tensor_bf16_hot(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_bf16_kind(shape, BufferKind::Hot)
    }

    fn alloc_tensor_bf16_kind(
        self: &Arc<Self>,
        shape: &[usize],
        kind: BufferKind,
    ) -> Result<crate::tensor::Tensor, String> {
        let numel: usize = shape.iter().product();
        let buf = self.alloc_buffer_kind(numel * 2, kind)?;
        buf.zero();
        Ok(crate::tensor::Tensor {
            buffer: buf,
            shape: shape.to_vec(),
            dtype: crate::tensor::DType::BF16,
            byte_offset: 0,
            runtime: Arc::clone(self),
        })
    }

    /// Encode a compute pass via [`crate::dispatch::Binder`] (Metal 4).
    ///
    /// Audit 4 P1: one compute encoder is kept open across calls within a CB
    /// (packed dispatches + per-dispatch barriers). Call sites still use one
    /// `with_binder` per op for telemetry (`dispatch_count`).
    pub fn with_binder<F>(&self, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        // DecodeIcb replay prep: skip Metal encode; caller still runs push_u32 /
        // KV host bookkeeping outside the binder closure.
        if crate::decode_icb::binder_encode_nop() {
            return Ok(());
        }
        self.flush_residency();
        let async_on = self.async_encode_enabled();
        if async_on {
            self.encode_into_batch_m4(f)?;
            if let Ok(mut c) = self.dispatch_count.lock() {
                *c += 1;
            }
            return Ok(());
        }
        self.with_binder_sync(f)
    }

    fn ensure_m4_cb_open(&self, batch: &mut Option<ActiveMetal4Batch>) -> Result<(), String> {
        // Do not call flush_residency here — caller may already hold `active_m4`.
        let m4 = &self.metal4;
        let need_begin = match batch.as_ref() {
            Some(b) => !b.cb_open,
            None => true,
        };
        if need_begin {
            let alloc_idx = {
                let mut slots = m4.allocators.lock().map_err(|e| e.to_string())?;
                // Prefer a free allocator so mid-commit never blocks host encode
                // while the peer CB is still executing.
                let free = slots
                    .iter()
                    .position(|s| s.in_flight == 0)
                    .unwrap_or(usize::MAX);
                let i = if free != usize::MAX {
                    free
                } else {
                    // Both in flight — wait for the earlier signal (smaller event).
                    let (i0, v0) = (0usize, slots[0].in_flight);
                    let (i1, v1) = (1usize, slots[1].in_flight);
                    let (i, v) = if v0 <= v1 { (i0, v0) } else { (i1, v1) };
                    if !m4
                        .shared_event
                        .waitUntilSignaledValue_timeoutMS(v, 30_000)
                    {
                        return Err("Metal 4 allocator SharedEvent wait timed out".to_string());
                    }
                    // Reset every slot that has completed (event ≥ in_flight).
                    for s in slots.iter_mut() {
                        if s.in_flight != 0 && s.in_flight <= v {
                            s.allocator.reset();
                            s.in_flight = 0;
                        }
                    }
                    i
                };
                if let Ok(mut idx) = m4.active_alloc.lock() {
                    *idx = i;
                }
                i
            };
            let alloc = {
                let slots = m4.allocators.lock().map_err(|e| e.to_string())?;
                slots[alloc_idx].allocator.clone()
            };
            m4.command_buffer.beginCommandBufferWithAllocator(&alloc);
            m4.command_buffer.useResidencySet(&m4.residency);
            let mut stamped = false;
            if let Some(heap) = m4.counter_heap.as_ref() {
                unsafe {
                    m4.command_buffer.writeTimestampIntoHeap_atIndex(heap, 0);
                }
                stamped = true;
            }
            let since = batch.as_ref().map(|b| b.since_commit).unwrap_or(0);
            let total = batch.as_ref().map(|b| b.dispatches).unwrap_or(0);
            *batch = Some(ActiveMetal4Batch {
                dispatches: total,
                since_commit: since,
                cb_open: true,
                stamped_t0: stamped,
                encoder: None,
                alloc_idx,
            });
        }
        Ok(())
    }

    fn encode_into_batch_m4<F>(&self, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        let m4 = &self.metal4;
        // Clone Retained encoder so we do not hold `active_m4` across `f`
        // (nested with_binder / flush would otherwise deadlock the Mutex).
        let enc = {
            let mut guard = self.active_m4.lock().map_err(|e| e.to_string())?;
            self.ensure_m4_cb_open(&mut guard)?;
            let batch = guard.as_mut().ok_or_else(|| "M4 batch missing".to_string())?;
            if batch.encoder.is_none() {
                let e = m4
                    .command_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| "MTL4 computeCommandEncoder failed".to_string())?;
                e.barrierAfterQueueStages_beforeStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
                batch.encoder = Some(e);
            }
            batch
                .encoder
                .as_ref()
                .ok_or_else(|| "M4 encoder missing".to_string())?
                .clone()
        };
        {
            let mut cursor = m4.const_cursor.lock().map_err(|e| e.to_string())?;
            let mut binder = crate::dispatch::Binder::new(
                enc.as_ref(),
                &m4.argument_table.table,
                &m4.const_staging,
                &mut cursor,
            );
            f(&mut binder)?;
        }
        let hit_mid = {
            let mut guard = self.active_m4.lock().map_err(|e| e.to_string())?;
            let batch = guard.as_mut().ok_or_else(|| "M4 batch missing".to_string())?;
            batch.dispatches += 1;
            batch.since_commit += 1;
            let thresh = mid_commit_threshold();
            // thresh==0 → mid-commit off (single CB / token). Hard cap avoids
            // unbounded CB growth if a client encodes without synchronize.
            (thresh > 0 && batch.since_commit >= thresh) || batch.dispatches >= 100_000
        };
        if hit_mid {
            self.commit(/*wait*/ false)?;
        }
        Ok(())
    }

    fn with_binder_sync<F>(&self, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        // Encode into the async-style batch then wait (keeps timestamps + residency).
        let was_async = self.async_encode_enabled();
        if !was_async {
            *self.async_encode.lock().map_err(|e| e.to_string())? = true;
        }
        let result = (|| {
            self.encode_into_batch_m4(f)?;
            self.synchronize()
        })();
        if !was_async {
            *self.async_encode.lock().map_err(|e| e.to_string())? = false;
        }
        if let Ok(mut c) = self.dispatch_count.lock() {
            *c += 1;
        }
        result
    }

    /// End encoding and commit without waiting (multi-step / low-sync path).
    pub fn commit(&self, wait: bool) -> Result<(), String> {
        self.commit_m4(wait)
    }

    fn commit_m4(&self, wait: bool) -> Result<(), String> {
        let mut guard = self.active_m4.lock().map_err(|e| e.to_string())?;
        let Some(batch) = guard.as_mut() else {
            if wait {
                self.drain_cold_recycles();
            }
            return Ok(());
        };
        if !batch.cb_open {
            if wait {
                // Still wait for any in-flight mid-commits.
                self.wait_all_allocators()?;
                *guard = None;
                self.drain_cold_recycles();
            }
            return Ok(());
        }
        let m4 = &self.metal4;
        crate::infer_trace::on_commit();

        // Close packed compute encoder before ending the CB.
        if let Some(enc) = batch.encoder.take() {
            enc.endEncoding();
        }

        let write_t1 = wait && batch.stamped_t0;
        if write_t1 {
            if let Some(heap) = m4.counter_heap.as_ref() {
                unsafe {
                    m4.command_buffer.writeTimestampIntoHeap_atIndex(heap, 1);
                }
            }
        }
        m4.command_buffer.endCommandBuffer();
        unsafe {
            let mut cb = NonNull::new(
                Retained::as_ptr(&m4.command_buffer) as *mut ProtocolObject<dyn MTL4CommandBuffer>,
            )
            .ok_or_else(|| "null MTL4 command buffer".to_string())?;
            m4.queue
                .commit_count(NonNull::new_unchecked(&mut cb as *mut _), 1);
        }
        batch.cb_open = false;
        batch.since_commit = 0;

        let next = {
            let mut v = m4.event_value.lock().map_err(|e| e.to_string())?;
            *v += 1;
            *v
        };
        m4.queue.signalEvent_value(
            ProtocolObject::<dyn MTLEvent>::from_ref(&*m4.shared_event),
            next,
        );
        // Mark this allocator in-flight; switch active slot for next begin.
        {
            let mut slots = m4.allocators.lock().map_err(|e| e.to_string())?;
            slots[batch.alloc_idx].in_flight = next;
            let mut idx = m4.active_alloc.lock().map_err(|e| e.to_string())?;
            *idx = 1 - batch.alloc_idx;
        }

        if wait {
            let t0 = std::time::Instant::now();
            if !m4
                .shared_event
                .waitUntilSignaledValue_timeoutMS(next, 30_000)
            {
                return Err("Metal 4 SharedEvent wait timed out".to_string());
            }
            crate::infer_trace::record_sync_wait(t0);
            self.reset_all_allocators()?;
            if write_t1 {
                if let Some(heap) = m4.counter_heap.as_ref() {
                    if let Ok(stamps) = resolve_two_timestamps(heap) {
                        if let Ok(mut g) = self.last_m4_stamps.lock() {
                            *g = Some(stamps);
                        }
                    }
                }
            }
            // Reset const arena only after GPU catch-up.
            if let Ok(mut c) = m4.const_cursor.lock() {
                *c = 0;
            }
            *guard = None;
            drop(guard);
            // Safe to removeAllocation + freelist now that CB completed.
            self.drain_cold_recycles();
        } else {
            // Keep const_cursor; do not reuse offsets until a waiting commit.
            // Next encode opens on the other allocator.
            batch.cb_open = false;
        }
        Ok(())
    }

    fn wait_all_allocators(&self) -> Result<(), String> {
        let m4 = &self.metal4;
        let slots = m4.allocators.lock().map_err(|e| e.to_string())?;
        let mut max_v = 0u64;
        for s in slots.iter() {
            max_v = max_v.max(s.in_flight);
        }
        drop(slots);
        if max_v == 0 {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        if !m4
            .shared_event
            .waitUntilSignaledValue_timeoutMS(max_v, 30_000)
        {
            return Err("Metal 4 SharedEvent wait timed out".to_string());
        }
        crate::infer_trace::record_sync_wait(t0);
        self.reset_all_allocators()
    }

    fn reset_all_allocators(&self) -> Result<(), String> {
        let m4 = &self.metal4;
        let mut slots = m4.allocators.lock().map_err(|e| e.to_string())?;
        for s in slots.iter_mut() {
            if s.in_flight != 0 {
                s.allocator.reset();
                s.in_flight = 0;
            }
        }
        Ok(())
    }

    /// Commit + wait. Required before host readbacks. SharedEvent covers the
    /// training compute CB on the Metal 4 path (not a stamp-only CB).
    pub fn synchronize(&self) -> Result<(), String> {
        self.commit(true)
    }

    /// True when a timestamp `MTL4CounterHeap` was created with the M4 package.
    pub fn metal4_counter_heap_available(&self) -> bool {
        self.metal4.counter_heap.is_some()
    }
}

fn probe_system_memory_size() -> u64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            if !o.status.success() {
                return None;
            }
            String::from_utf8(o.stdout)
                .ok()
                .and_then(|s| s.trim().parse().ok())
        })
        .unwrap_or(0)
}

fn resolve_two_timestamps(
    heap: &ProtocolObject<dyn MTL4CounterHeap>,
) -> Result<(u64, u64), String> {
    let data: Retained<NSData> = unsafe {
        heap.resolveCounterRange(NSRange {
            location: 0,
            length: 2,
        })
    }
    .ok_or_else(|| "resolveCounterRange returned nil".to_string())?;
    let need = 2 * std::mem::size_of::<MTL4TimestampHeapEntry>();
    let bytes = data.length() as usize;
    if bytes < need {
        return Err(format!("timestamp resolve too small: {bytes} bytes (need {need})"));
    }
    let mut buf = vec![0u8; need];
    unsafe {
        data.getBytes_length(NonNull::new(buf.as_mut_ptr().cast()).unwrap(), need);
        let ptr = buf.as_ptr() as *const MTL4TimestampHeapEntry;
        let a = (*ptr).timestamp;
        let b = (*ptr.add(1)).timestamp;
        Ok((a, b))
    }
}

fn try_init_metal4(
    device: &ProtocolObject<dyn MTLDevice>,
    timestamps: bool,
) -> Result<Metal4EncodePackage, String> {
    let queue = device
        .newMTL4CommandQueue()
        .ok_or_else(|| "newMTL4CommandQueue returned nil".to_string())?;
    let allocator_a = device
        .newCommandAllocator()
        .ok_or_else(|| "newCommandAllocator (A) returned nil".to_string())?;
    let allocator_b = device
        .newCommandAllocator()
        .ok_or_else(|| "newCommandAllocator (B) returned nil".to_string())?;
    let cmd = device
        .newCommandBuffer()
        .ok_or_else(|| "newCommandBuffer (MTL4) returned nil".to_string())?;

    let desc = MTL4ArgumentTableDescriptor::new();
    desc.setMaxBufferBindCount(31);
    desc.setMaxTextureBindCount(16);
    desc.setMaxSamplerStateBindCount(8);
    let table = device
        .newArgumentTableWithDescriptor_error(&desc)
        .map_err(|e| format!("newArgumentTable: {e}"))?;

    let counter_heap = if timestamps {
        let heap_desc = MTL4CounterHeapDescriptor::new();
        heap_desc.setType(MTL4CounterHeapType::Timestamp);
        unsafe {
            heap_desc.setCount(64);
        }
        match device.newCounterHeapWithDescriptor_error(&heap_desc) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("[metal-runtime] MTL4CounterHeap unavailable ({e}); timestamps off");
                None
            }
        }
    } else {
        None
    };

    let res_desc = MTLResidencySetDescriptor::new();
    let shared_event = device
        .newSharedEvent()
        .ok_or_else(|| "newSharedEvent returned nil".to_string())?;

    // Multi-slot const arena for batched argument-table encode (~1 MiB).
    let const_staging = device
        .newBufferWithLength_options(
            METAL4_CONST_ARENA_BYTES,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or_else(|| "const_staging buffer alloc failed".to_string())?;

    let residency = device
        .newResidencySetWithDescriptor_error(&res_desc)
        .map_err(|e| format!("newResidencySet: {e}"))?;
    // Const arena is always resident for M4 encode.
    residency.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*const_staging));
    residency.commit();
    residency.requestResidency();

    Ok(Metal4EncodePackage {
        queue,
        allocator: allocator_a.clone(),
        allocators: Mutex::new([
            AllocatorSlot {
                allocator: allocator_a,
                in_flight: 0,
            },
            AllocatorSlot {
                allocator: allocator_b,
                in_flight: 0,
            },
        ]),
        active_alloc: Mutex::new(0),
        command_buffer: cmd,
        argument_table: PersistentArgumentTable {
            table,
            max_buffers: 31,
        },
        counter_heap,
        residency,
        shared_event,
        const_staging,
        const_cursor: Mutex::new(0),
        event_value: Mutex::new(0),
        residency_count: Mutex::new(1),
    })
}

pub fn mtl_size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal4_encode_smoke_copy() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 64usize;
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i + 1) as f32;
            }
            let q = dst.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(q as *mut u8, 0, n * 4);
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        let width = pipe.threadExecutionWidth() as usize;
        let tpt = width.min(n).max(1);
        let groups = (n + tpt - 1) / tpt;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })
        .expect("metal4 smoke");
        rt.synchronize().unwrap();
        let out = unsafe {
            std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n)
        };
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i + 1) as f32, "smoke mismatch at {i}");
        }
        if rt.metal4_counter_heap_available() {
            let stamps = rt.take_metal4_stamps().expect("expected timestamps");
            assert!(stamps.1 >= stamps.0, "timestamps not monotonic: {stamps:?}");
        }
    }

    #[test]
    fn metal4_batched_multi_dispatch_const_arena() {
        let rt = GpuRuntime::new().expect("runtime");
        rt.set_async_encode(true).unwrap();

        let n1 = 16usize;
        let n2 = 24usize;
        let src1 = rt.alloc_buffer(n1 * 4).unwrap();
        let mid = rt.alloc_buffer(n1.max(n2) * 4).unwrap();
        let dst = rt.alloc_buffer(n2 * 4).unwrap();
        unsafe {
            let p = src1.metal().contents().as_ptr() as *mut f32;
            for i in 0..n1 {
                *p.add(i) = (i + 1) as f32;
            }
            let q = mid.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(q as *mut u8, 0, n1.max(n2) * 4);
            let r = dst.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(r as *mut u8, 0, n2 * 4);
        }
        let src2 = rt.alloc_buffer(n2 * 4).unwrap();
        unsafe {
            let p = src2.metal().contents().as_ptr() as *mut f32;
            for i in 0..n2 {
                *p.add(i) = 100.0 + i as f32;
            }
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src1, 0);
            bnd.bind_gpu_buf(&mid, 1);
            bnd.bind_u32(n1 as u32, 2);
            bnd.dispatch(mtl_size(1, 1, 1), mtl_size(n1, 1, 1));
            Ok(())
        })
        .unwrap();
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src2, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(n2 as u32, 2);
            bnd.dispatch(mtl_size(1, 1, 1), mtl_size(n2, 1, 1));
            Ok(())
        })
        .unwrap();
        rt.synchronize().unwrap();

        let mid_out = unsafe {
            std::slice::from_raw_parts(mid.metal().contents().as_ptr() as *const f32, n1).to_vec()
        };
        let dst_out = unsafe {
            std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n2).to_vec()
        };
        for i in 0..n1 {
            assert_eq!(mid_out[i], (i + 1) as f32, "first copy mismatch at {i}");
        }
        for i in 0..n2 {
            assert_eq!(dst_out[i], 100.0 + i as f32, "second copy mismatch at {i}");
        }
        assert_eq!(*rt.metal4.const_cursor.lock().unwrap(), 0);
        assert!(rt.metal4.const_staging.length() as usize >= METAL4_CONST_ARENA_BYTES);
    }

    #[test]
    fn metal4_arg_table_offset_and_multi_const() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 32usize;
        let pad = 16usize;
        let src = rt.alloc_buffer((pad + n) * 4).expect("src");
        let dst = rt.alloc_buffer(n * 4).expect("dst");
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..(pad + n) {
                *p.add(i) = if i < pad {
                    -1.0
                } else {
                    (i - pad + 1) as f32
                };
            }
            let q = dst.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(q as *mut u8, 0, n * 4);
        }
        let pipe = rt.pipeline("copy_f32").expect("pipe");
        let width = pipe.threadExecutionWidth() as usize;
        let tpt = width.min(n).max(1);
        let groups = (n + tpt - 1) / tpt;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_buf(src.metal(), pad * 4, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })
        .expect("m4 offset dispatch");
        rt.synchronize().unwrap();
        let out = unsafe {
            std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n).to_vec()
        };
        for (i, &v) in out.iter().enumerate() {
            let expect = (i + 1) as f32;
            assert!(
                (v - expect).abs() == 0.0,
                "offset bind mismatch at {i}: got {v} want {expect}"
            );
        }
        assert!(rt.metal4.const_staging.length() as usize >= METAL4_CONST_ARENA_BYTES);
        // Multi-const staging via Binder const arena.
        rt.set_async_encode(true).unwrap();
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&dst, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(1, 2);
            bnd.bind_u32(2, 3);
            bnd.bind_f32(3.5, 4);
            // No dispatch needed for arena cursor check — but setArgumentTable
            // only happens on dispatch; just advance cursor via binds then barrier.
            bnd.barrier();
            Ok(())
        })
        .unwrap();
        let cursor = *rt.metal4.const_cursor.lock().unwrap();
        assert!(cursor > 0, "const arena cursor should advance");
        rt.synchronize().unwrap();
        assert_eq!(*rt.metal4.const_cursor.lock().unwrap(), 0);
    }
}
