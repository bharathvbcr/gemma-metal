//! Mini decode ICB (compute Indirect Command Buffer).
//!
//! Extends [`crate::icb_smoke`] from one `copy_f32` to N ConcurrentDispatch
//! commands. Default path: `inheritBuffers=true` + MTL4 prebuilt argument
//! tables at execute (`setArgumentTable` × unique tables; sticky adopt skips
//! redundant switches). Opt-in **freeze-binds** (`GEMMA_METAL_ICB_FREEZE_BINDS=1`):
//! `inheritBuffers=false` + classic `setKernelBuffer`+tg_mem freeze into the ICB
//! so execute pays **0** `setArgumentTable` (requires ICB-capable pipelines;
//! parent PSO must still be set on the encoder before `execute_icb`).
//!
//! Opt-in **range-batch** (`GEMMA_METAL_ICB_RANGE_BATCH=1`, v0.5.8): under
//! freeze-binds, coalesce safe spans between captured `barrier_after` markers
//! into one `executeCommandsInBuffer:withRange:` — cuts `execute_icb`×N host
//! tax. Default **OFF** until a product tok/s win.
//!
//! **Coarse ranges (v0.5.9):** with range-batch, elide `barrier_after` markers
//! when the next cmd's large-Buf set is disjoint from the open span (≤64B
//! scalars + ambient high-frequency pools like `IcbScalarPool` ignored as
//! read-only during tape execute). Larger `executeCommandsInBuffer` spans;
//! still OFF by default with freeze/range.
//!
//! Wired into [`crate::cb_replay::PingPongCbReplay::try_replay_icb`] so
//! `try_replay` → `execute_icb` works for an attached mini or E4B Hot graph
//! (session eligibility is host-side). **31B** decode-graph migration remains
//! open. Opt-in; default OFF.

use std::sync::atomic::{AtomicI8, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4Compiler, MTL4CompilerDescriptor,
    MTL4ComputePipelineDescriptor, MTL4IndirectCommandBufferSupportState,
    MTL4LibraryFunctionDescriptor, MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDevice,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::ab_flags::env_truthy;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::GpuBuffer;

/// -1 = env, 0 = off, 1 = on.
static DECODE_ICB: AtomicI8 = AtomicI8::new(-1);

/// Force DecodeIcb replay opt-in (tests / harness). Overrides env.
pub fn set_decode_icb(on: bool) {
    DECODE_ICB.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in gate for DecodeIcb replay via ping-pong. Default **OFF**.
///
/// Env: `METAL_RUNTIME_DECODE_ICB=1` or `GEMMA_METAL_DECODE_ICB=1`.
pub fn decode_icb_enabled() -> bool {
    let v = DECODE_ICB.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_truthy(&["METAL_RUNTIME_DECODE_ICB", "GEMMA_METAL_DECODE_ICB"]).unwrap_or(false);
    DECODE_ICB.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// One argument-table bind for a captured / frozen command.
#[derive(Clone)]
pub enum DecodeIcbBind {
    /// Stable `GpuBuffer` (weights, activations, `IcbScalarPool` slots).
    Buf {
        index: usize,
        buf: GpuBuffer,
        byte_offset: usize,
        /// `buf.gpuAddress() + byte_offset` cached at capture (A2 bind-tax cut).
        gpu_addr: u64,
    },
    /// Const-arena scalar snapshotted at capture; materialized into a Hot buffer
    /// when building the ICB (frozen value — use `IcbScalarPool` for per-token).
    Immediate { index: usize, bytes: Vec<u8> },
}

#[inline]
fn buf_gpu_addr(buf: &GpuBuffer, byte_offset: usize) -> u64 {
    buf.metal()
        .gpuAddress()
        .wrapping_add(byte_offset as u64)
}

/// One frozen compute dispatch (pipeline + grid + bind recipe).
#[derive(Clone)]
pub struct DecodeIcbCommand {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub threadgroups: MTLSize,
    pub threads_per_tg: MTLSize,
    pub binds: Vec<DecodeIcbBind>,
    pub tg_mem: Option<(usize, usize)>,
    /// Owned Hot buffers for [`DecodeIcbBind::Immediate`] (keep residency).
    pub owned_immediates: Vec<GpuBuffer>,
    /// Insert a Dispatch→Dispatch Device barrier after this cmd on replay.
    ///
    /// Captured from live always-on auto-barriers and explicit [`crate::dispatch::Binder::barrier`]
    /// calls. Replay must honor these instead of forcing always-on for every cmd
    /// (product tok/s regression under shipping hazard skip-auto — D16).
    pub barrier_after: bool,
}

/// MTL4 argument-table max buffer bind count (matches runtime table descriptor).
const ARG_TABLE_SLOTS: usize = 31;

/// Sticky arg-table state for A2: skip `setAddress` when the slot already holds
/// the same GPU address across consecutive tape commands.
///
/// Used when prebuilt per-command tables are disabled; otherwise Buf binds are
/// frozen into dedicated tables at capture and execute only switches tables.
struct StickyArgTable {
    addr: [u64; ARG_TABLE_SLOTS],
    /// Bit i set ⇒ `addr[i]` is live in the table.
    valid: u32,
    /// `setAddress` calls actually issued this execute.
    set_calls: u64,
    /// Binds considered (Buf + Immediate) this execute.
    bind_total: u64,
    /// Buf binds satisfied by adopting a prebuilt table (no `setAddress`).
    prebuilt_elided: u64,
    /// `setArgumentTable` switches this execute (prebuilt path).
    set_table_calls: u64,
}

impl StickyArgTable {
    fn new() -> Self {
        Self {
            addr: [0; ARG_TABLE_SLOTS],
            valid: 0,
            set_calls: 0,
            bind_total: 0,
            prebuilt_elided: 0,
            set_table_calls: 0,
        }
    }

    #[inline]
    fn bind_addr(
        &mut self,
        bnd: &mut crate::dispatch::Binder<'_>,
        gpu_addr: u64,
        index: usize,
    ) {
        self.bind_total = self.bind_total.saturating_add(1);
        if index < ARG_TABLE_SLOTS {
            let bit = 1u32 << index;
            if (self.valid & bit) != 0 && self.addr[index] == gpu_addr {
                return;
            }
            self.addr[index] = gpu_addr;
            self.valid |= bit;
        }
        bnd.bind_addr(gpu_addr, index);
        self.set_calls = self.set_calls.saturating_add(1);
    }

    /// Immediate binds always materialize a fresh const-arena address.
    #[inline]
    fn bind_bytes(
        &mut self,
        bnd: &mut crate::dispatch::Binder<'_>,
        bytes: &[u8],
        index: usize,
    ) {
        self.bind_total = self.bind_total.saturating_add(1);
        let addr = bnd.bind_bytes(bytes, index);
        self.set_calls = self.set_calls.saturating_add(1);
        if index < ARG_TABLE_SLOTS {
            let bit = 1u32 << index;
            self.addr[index] = addr;
            self.valid |= bit;
        }
    }

    /// Prefill path: Buf already frozen in `table`; only Immediate needs
    /// `setAddress` (const-arena address changes each execute).
    #[inline]
    fn bind_prebuilt(
        &mut self,
        bnd: &mut crate::dispatch::Binder<'_>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        binds: &[DecodeIcbBind],
    ) {
        if bnd.adopt_argument_table(table) {
            self.set_table_calls = self.set_table_calls.saturating_add(1);
        }
        for b in binds {
            self.bind_total = self.bind_total.saturating_add(1);
            match b {
                DecodeIcbBind::Buf { .. } => {
                    self.prebuilt_elided = self.prebuilt_elided.saturating_add(1);
                }
                DecodeIcbBind::Immediate { index, bytes } => {
                    let addr = bnd.materialize_bytes(bytes);
                    unsafe {
                        table.setAddress_atIndex(addr, *index);
                    }
                    self.set_calls = self.set_calls.saturating_add(1);
                }
            }
        }
    }
}

/// Opt-out: `GEMMA_METAL_ICB_PREBUILT_TABLES=0` / `METAL_RUNTIME_ICB_PREBUILT_TABLES=0`.
/// Default **ON** — freeze Buf binds into per-command MTL4 argument tables.
fn prebuilt_tables_enabled() -> bool {
    env_truthy(&[
        "METAL_RUNTIME_ICB_PREBUILT_TABLES",
        "GEMMA_METAL_ICB_PREBUILT_TABLES",
    ])
    .unwrap_or(true)
}

/// -1 = env, 0 = off, 1 = on.
static ICB_FREEZE_BINDS: AtomicI8 = AtomicI8::new(-1);

/// Force freeze-binds opt-in (tests / harness). Overrides env.
pub fn set_icb_freeze_binds(on: bool) {
    ICB_FREEZE_BINDS.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in: freeze Buf binds into the ICB via classic `setKernelBuffer`
/// (`inheritBuffers=false`). Execute skips `setArgumentTable` entirely.
/// Default **OFF**. Implies true `execute_icb` + ICB-capable pipelines.
///
/// Env: `METAL_RUNTIME_ICB_FREEZE_BINDS=1` or `GEMMA_METAL_ICB_FREEZE_BINDS=1`.
pub fn icb_freeze_binds_enabled() -> bool {
    let v = ICB_FREEZE_BINDS.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_truthy(&[
        "METAL_RUNTIME_ICB_FREEZE_BINDS",
        "GEMMA_METAL_ICB_FREEZE_BINDS",
    ])
    .unwrap_or(false);
    ICB_FREEZE_BINDS.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// -1 = env, 0 = off, 1 = on.
static ICB_RANGE_BATCH: AtomicI8 = AtomicI8::new(-1);

/// Force range-batch opt-in (tests / harness). Overrides env.
pub fn set_icb_range_batch(on: bool) {
    ICB_RANGE_BATCH.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in: under freeze-binds, batch safe ICB command ranges between captured
/// `barrier_after` markers into one `executeCommandsInBuffer:withRange:`.
/// Default **OFF**. No effect without freeze-binds (per-cmd arg-table path
/// cannot coalesce). When on, also runs coarse-range barrier elision (v0.5.9).
///
/// Env: `METAL_RUNTIME_ICB_RANGE_BATCH=1` or `GEMMA_METAL_ICB_RANGE_BATCH=1`.
pub fn icb_range_batch_enabled() -> bool {
    let v = ICB_RANGE_BATCH.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_truthy(&[
        "METAL_RUNTIME_ICB_RANGE_BATCH",
        "GEMMA_METAL_ICB_RANGE_BATCH",
    ])
    .unwrap_or(false);
    ICB_RANGE_BATCH.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// -1 = env / follow range-batch, 0 = off, 1 = on.
static ICB_COARSE_RANGES: AtomicI8 = AtomicI8::new(-1);

/// Force coarse-range barrier elision (tests / harness). Overrides env.
pub fn set_icb_coarse_ranges(on: bool) {
    ICB_COARSE_RANGES.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in: elide non-interfering `barrier_after` markers before range-batch
/// (dst-slot heuristic). Default **ON when range-batch is on**; set
/// `GEMMA_METAL_ICB_COARSE_RANGES=0` to keep every captured barrier.
///
/// Env: `METAL_RUNTIME_ICB_COARSE_RANGES` / `GEMMA_METAL_ICB_COARSE_RANGES`.
pub fn icb_coarse_ranges_enabled() -> bool {
    let v = ICB_COARSE_RANGES.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    if let Some(on) = env_truthy(&[
        "METAL_RUNTIME_ICB_COARSE_RANGES",
        "GEMMA_METAL_ICB_COARSE_RANGES",
    ]) {
        ICB_COARSE_RANGES.store(if on { 1 } else { 0 }, Ordering::Relaxed);
        return on;
    }
    // Follow range-batch: coarsening only pays when spans are coalesced.
    icb_range_batch_enabled()
}

/// Fingerprint of Buf `(index, gpu_addr)` pairs for prebuilt-table dedup.
fn buf_bind_fingerprint(binds: &[DecodeIcbBind]) -> u64 {
    // FNV-1a 64 — stable, cheap, good enough for capture-time table sharing.
    let mut slots: Vec<(u16, u64)> = Vec::with_capacity(binds.len());
    for b in binds {
        if let DecodeIcbBind::Buf {
            index, gpu_addr, ..
        } = b
        {
            if *index < ARG_TABLE_SLOTS {
                slots.push((*index as u16, *gpu_addr));
            }
        }
    }
    slots.sort_unstable_by_key(|s| s.0);
    let mut h = 0xcbf29ce484222325u64;
    for &(idx, addr) in &slots {
        h ^= idx as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= addr;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Multi-command decode ICB (inheritBuffers + arg-table execute, or freeze-binds).
pub struct DecodeIcb {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    commands: Vec<DecodeIcbCommand>,
    /// Per-command MTL4 argument tables with Buf addresses frozen at capture
    /// (A2 residual: execute switches tables instead of re-`setAddress`).
    /// Empty when [`Self::freeze_binds`] is true.
    prebuilt_tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    /// Unique prebuilt tables after fingerprint dedup (≤ `prebuilt_tables.len()`).
    unique_prebuilt_tables: usize,
    /// Classic `setKernelBuffer` freeze (`inheritBuffers=false`); execute skips
    /// argument-table traffic.
    freeze_binds: bool,
    encoded: bool,
    optimized: bool,
    execute_count: u64,
    /// True when built from a Binder capture of the mini layer/head graph
    /// (not the `mini_copy_chain` smoke). Enables host-encode skip on replay.
    layer_graph: bool,
    /// Triage probe (`GEMMA_METAL_ICB_TRIAGE=1`): buffer + f32 count sampled
    /// after every replayed command to localize the first diverging dispatch.
    triage_probe: Option<(crate::tensor::GpuBuffer, usize)>,
    /// Last execute: total binds considered / `setAddress` issued (A2 sticky).
    last_bind_total: u64,
    last_set_address_calls: u64,
    /// Last execute: `setArgumentTable` switches (prebuilt path).
    last_set_argument_table_calls: u64,
    /// Last execute: Buf binds elided via prebuilt tables.
    last_prebuilt_elided: u64,
    /// Last execute: `executeCommandsInBuffer` calls (freeze / ICB execute).
    last_execute_icb_calls: u64,
    /// Last execute: total ICB cmds covered by those calls.
    last_execute_icb_cmds: u64,
    /// Capture-time: Buf binds that would be sticky-skippable across cmds.
    sticky_skippable_binds: u64,
    total_buf_binds: u64,
    /// Once-only coarse-range pass applied (v0.5.9).
    barriers_coarsened: bool,
    /// How many `barrier_after` markers elided by coarse-range analysis.
    barriers_elided: u64,
}

impl DecodeIcb {
    pub fn from_commands(
        rt: &GpuRuntime,
        commands: Vec<DecodeIcbCommand>,
    ) -> Result<Self, String> {
        Self::from_commands_ex(rt, commands, icb_freeze_binds_enabled())
    }

    /// Like [`Self::from_commands`] with explicit freeze-binds control (tests /
    /// harness). When `freeze_binds` is true, ICB uses `inheritBuffers=false` +
    /// classic `setKernelBuffer` and execute skips `setArgumentTable`.
    pub fn from_commands_ex(
        rt: &GpuRuntime,
        commands: Vec<DecodeIcbCommand>,
        freeze_binds: bool,
    ) -> Result<Self, String> {
        if commands.is_empty() {
            return Err("DecodeIcb: empty command list".into());
        }
        let mut commands = commands;
        if freeze_binds {
            // Classic setKernelBuffer needs real MTLBuffers; materialize any
            // residual Immediate bytes into owned Hot Bufs (mini+E4B usually 0).
            Self::materialize_immediates(rt, &mut commands)?;
        }

        let desc = MTLIndirectCommandBufferDescriptor::new();
        desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        desc.setInheritPipelineState(false);
        if freeze_binds {
            // Freeze kernel buffers into each ICB cmd — execute pays no arg-table.
            desc.setInheritBuffers(false);
            desc.setMaxKernelBufferBindCount(ARG_TABLE_SLOTS);
        } else {
            // Working bridge on MacOSX26: inherit arg-table at execute time.
            desc.setInheritBuffers(true);
            desc.setMaxKernelBufferBindCount(0);
        }

        let n = commands.len();
        let icb = unsafe {
            rt.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &desc,
                    n,
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or_else(|| "newIndirectCommandBuffer failed (DecodeIcb)".to_string())?;

        rt.register_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*icb));

        // Keep Immediate binds as snapshotted bytes (non-freeze). Re-pack via
        // `bind_bytes` at execute. Prefer `IcbScalarPool` Buf binds for
        // per-token scalars.
        let layer_graph = commands.len() >= Self::MIN_LAYER_GRAPH_COMMANDS;
        let (total_buf_binds, sticky_skippable_binds) = Self::analyze_sticky_buf_binds(&commands);
        let (prebuilt_tables, unique_prebuilt_tables) = if freeze_binds {
            (Vec::new(), 0)
        } else if prebuilt_tables_enabled() {
            Self::build_prebuilt_tables(rt, &commands)?
        } else {
            (Vec::new(), 0)
        };
        let mut this = Self {
            icb,
            commands,
            prebuilt_tables,
            unique_prebuilt_tables,
            freeze_binds,
            encoded: false,
            optimized: false,
            execute_count: 0,
            layer_graph,
            triage_probe: None,
            last_bind_total: 0,
            last_set_address_calls: 0,
            last_set_argument_table_calls: 0,
            last_prebuilt_elided: 0,
            last_execute_icb_calls: 0,
            last_execute_icb_cmds: 0,
            sticky_skippable_binds,
            total_buf_binds,
            barriers_coarsened: false,
            barriers_elided: 0,
        };
        this.encode_cpu()?;
        Ok(this)
    }

    /// Stable buffer identity (base GPU address) for interference analysis.
    #[inline]
    fn buf_id(buf: &GpuBuffer) -> u64 {
        buf.metal().gpuAddress()
    }

    /// Tiny Hot scalars (softcap f32×1, lone u32 dims) — read-only on tape.
    const READONLY_MAX_NBYTES: usize = 64;

    /// Large-Buf ids for a cmd, minus ambient read-only. Immediate → `None`.
    fn cmd_buf_set(cmd: &DecodeIcbCommand, ambient: &[u64]) -> Option<Vec<u64>> {
        let mut ids = Vec::with_capacity(cmd.binds.len());
        for b in &cmd.binds {
            match b {
                DecodeIcbBind::Immediate { .. } => return None,
                DecodeIcbBind::Buf { buf, .. } => {
                    if buf.nbytes() <= Self::READONLY_MAX_NBYTES {
                        continue;
                    }
                    let id = Self::buf_id(buf);
                    if ambient.iter().any(|&a| a == id) {
                        continue;
                    }
                    ids.push(id);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        Some(ids)
    }

    /// Buffers bound by ≥ half of cmds (IcbScalarPool u32/f32 arenas, etc.) —
    /// host-updated between tokens, read-only during a single tape execute.
    fn ambient_readonly_bufs(commands: &[DecodeIcbCommand]) -> Vec<u64> {
        let n = commands.len().max(1);
        let mut freq: Vec<(u64, usize)> = Vec::new();
        for cmd in commands {
            let mut seen = Vec::new();
            for b in &cmd.binds {
                if let DecodeIcbBind::Buf { buf, .. } = b {
                    if buf.nbytes() <= Self::READONLY_MAX_NBYTES {
                        continue;
                    }
                    let id = Self::buf_id(buf);
                    if seen.iter().any(|&s| s == id) {
                        continue;
                    }
                    seen.push(id);
                    if let Some(e) = freq.iter_mut().find(|(k, _)| *k == id) {
                        e.1 += 1;
                    } else {
                        freq.push((id, 1));
                    }
                }
            }
        }
        // ≥80% of cmds and at least 8 hits — avoids marking short-tape RAW
        // buffers (e.g. b in a→b / b→e) as ambient while still catching
        // IcbScalarPool arenas on mini/E4B graphs.
        freq.into_iter()
            .filter(|(_, c)| *c >= 8 && *c * 5 >= n * 4)
            .map(|(id, _)| id)
            .collect()
    }

    /// Elide `barrier_after` when the next cmd's Buf set is disjoint from every
    /// Buf touched in the open span. Returns elided count.
    pub fn elide_non_interfering_barriers(commands: &mut [DecodeIcbCommand]) -> u64 {
        if commands.len() < 2 {
            return 0;
        }
        let ambient = Self::ambient_readonly_bufs(commands);
        let mut elided = 0u64;
        let mut span: Vec<u64> = Vec::new();
        for i in 0..commands.len() {
            match Self::cmd_buf_set(&commands[i], &ambient) {
                None => {
                    span.clear();
                    continue;
                }
                Some(ids) => {
                    for id in ids {
                        if !span.iter().any(|&x| x == id) {
                            span.push(id);
                        }
                    }
                }
            }
            if !commands[i].barrier_after {
                continue;
            }
            if i + 1 >= commands.len() {
                span.clear();
                continue;
            }
            let keep = match Self::cmd_buf_set(&commands[i + 1], &ambient) {
                None => true,
                Some(next) => span.iter().any(|w| next.iter().any(|n| n == w)),
            };
            if keep {
                span.clear();
            } else {
                commands[i].barrier_after = false;
                elided = elided.saturating_add(1);
            }
        }
        elided
    }

    pub fn barriers_elided(&self) -> u64 {
        self.barriers_elided
    }

    /// Convert Immediate binds into owned Hot `Buf`s (freeze-binds path).
    fn materialize_immediates(
        rt: &GpuRuntime,
        commands: &mut [DecodeIcbCommand],
    ) -> Result<(), String> {
        for cmd in commands.iter_mut() {
            let mut owned = Vec::new();
            for b in cmd.binds.iter_mut() {
                if let DecodeIcbBind::Immediate { index, bytes } = b {
                    let buf = rt.alloc_buffer_hot(bytes.len().max(4))?;
                    unsafe {
                        let dst = buf.metal().contents().as_ptr() as *mut u8;
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                    }
                    let gpu_addr = buf_gpu_addr(&buf, 0);
                    let idx = *index;
                    owned.push(buf.clone());
                    *b = DecodeIcbBind::Buf {
                        index: idx,
                        buf,
                        byte_offset: 0,
                        gpu_addr,
                    };
                }
            }
            cmd.owned_immediates.append(&mut owned);
        }
        Ok(())
    }

    /// Freeze Buf `gpu_addr`s into MTL4 argument tables; dedup by fingerprint
    /// so sticky adopt can skip redundant `setArgumentTable` switches.
    fn build_prebuilt_tables(
        rt: &GpuRuntime,
        commands: &[DecodeIcbCommand],
    ) -> Result<(Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>, usize), String> {
        let mut unique: Vec<(u64, Retained<ProtocolObject<dyn MTL4ArgumentTable>>)> =
            Vec::new();
        let mut tables = Vec::with_capacity(commands.len());
        for cmd in commands {
            let fp = buf_bind_fingerprint(&cmd.binds);
            if let Some((_, t)) = unique.iter().find(|(f, _)| *f == fp) {
                tables.push(t.clone());
                continue;
            }
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(ARG_TABLE_SLOTS);
            desc.setMaxTextureBindCount(16);
            desc.setMaxSamplerStateBindCount(8);
            let table = rt
                .device
                .newArgumentTableWithDescriptor_error(&desc)
                .map_err(|e| format!("DecodeIcb prebuilt newArgumentTable: {e}"))?;
            for b in &cmd.binds {
                if let DecodeIcbBind::Buf {
                    index, gpu_addr, ..
                } = b
                {
                    unsafe {
                        table.setAddress_atIndex(*gpu_addr, *index);
                    }
                }
            }
            unique.push((fp, table.clone()));
            tables.push(table);
        }
        Ok((tables, unique.len()))
    }

    /// Count Buf binds and how many would skip `setAddress` under sticky replay.
    fn analyze_sticky_buf_binds(commands: &[DecodeIcbCommand]) -> (u64, u64) {
        let mut addr = [0u64; ARG_TABLE_SLOTS];
        let mut valid = 0u32;
        let mut total = 0u64;
        let mut skippable = 0u64;
        for cmd in commands {
            for b in &cmd.binds {
                match b {
                    DecodeIcbBind::Buf {
                        index, gpu_addr, ..
                    } => {
                        total = total.saturating_add(1);
                        if *index < ARG_TABLE_SLOTS {
                            let bit = 1u32 << *index;
                            if (valid & bit) != 0 && addr[*index] == *gpu_addr {
                                skippable = skippable.saturating_add(1);
                            } else {
                                addr[*index] = *gpu_addr;
                                valid |= bit;
                            }
                        }
                    }
                    DecodeIcbBind::Immediate { index, .. } => {
                        // Immediate always sets; invalidate sticky at index.
                        if *index < ARG_TABLE_SLOTS {
                            valid &= !(1u32 << *index);
                        }
                    }
                }
            }
        }
        (total, skippable)
    }

    /// Attach a triage probe: after each replayed command (when
    /// `GEMMA_METAL_ICB_TRIAGE=1`), synchronize and print the probe buffer's
    /// absmean. One M5 run then pinpoints the first command whose output
    /// diverges from the live-encode trace. Costs a sync per command — never
    /// leave on for measurement.
    pub fn set_triage_probe(&mut self, buf: crate::tensor::GpuBuffer, n_f32: usize) {
        self.triage_probe = Some((buf, n_f32));
    }

    /// Minimum captured dispatches to treat as a mini layer/head graph (not copy-chain).
    pub const MIN_LAYER_GRAPH_COMMANDS: usize = 8;

    /// Convenience: two-hop `copy_f32` mini graph (a→b→c) for try_replay tests.
    pub fn mini_copy_chain(rt: &GpuRuntime, n: usize) -> Result<(Self, GpuBuffer), String> {
        Self::mini_copy_chain_ex(rt, n, icb_freeze_binds_enabled())
    }

    /// Like [`Self::mini_copy_chain`] with explicit freeze-binds control.
    pub fn mini_copy_chain_ex(
        rt: &GpuRuntime,
        n: usize,
        freeze_binds: bool,
    ) -> Result<(Self, GpuBuffer), String> {
        if n == 0 {
            return Err("DecodeIcb::mini_copy_chain: n > 0".into());
        }
        // Hot/shared staging: classic setKernelBuffer freeze is flaky with
        // Private-only allocs on some SDK paths; Hot matches layer-graph residency.
        let a = rt.alloc_buffer_hot(n * 4)?;
        let b = rt.alloc_buffer_hot(n * 4)?;
        let c = rt.alloc_buffer_hot(n * 4)?;
        let n_buf = rt.alloc_buffer_hot(4)?;
        n_buf.write_u32(&[n as u32]);
        unsafe {
            let p = a.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i as f32) + 1.0;
            }
            let zb = b.metal().contents().as_ptr() as *mut u8;
            let zc = c.metal().contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(zb, 0, n * 4);
            std::ptr::write_bytes(zc, 0, n * 4);
        }
        let pipe = pipeline_icb(rt, "copy_f32")?;
        let tpt = (pipe.threadExecutionWidth() as usize).min(n).max(1);
        let groups = (n + tpt - 1) / tpt;
        let tg = mtl_size(groups, 1, 1);
        let tptg = mtl_size(tpt, 1, 1);
        let buf = |index, buf: &GpuBuffer| DecodeIcbBind::Buf {
            index,
            buf: buf.clone(),
            byte_offset: 0,
            gpu_addr: buf_gpu_addr(buf, 0),
        };
        let cmds = vec![
            DecodeIcbCommand {
                pipeline: pipe.clone(),
                threadgroups: tg,
                threads_per_tg: tptg,
                binds: vec![buf(0, &a), buf(1, &b), buf(2, &n_buf)],
                tg_mem: None,
                owned_immediates: Vec::new(),
                // a→b must drain before b→c (no live always-on during tape execute).
                barrier_after: true,
            },
            DecodeIcbCommand {
                pipeline: pipe,
                threadgroups: tg,
                threads_per_tg: tptg,
                binds: vec![buf(0, &b), buf(1, &c), buf(2, &n_buf)],
                tg_mem: None,
                owned_immediates: Vec::new(),
                barrier_after: false,
            },
        ];
        let dec = Self::from_commands_ex(rt, cmds, freeze_binds)?;
        Ok((dec, c))
    }

    fn encode_cpu(&mut self) -> Result<(), String> {
        // Tape-only / direct-dispatch replay: allow non-ICB pipelines (default path).
        // Freeze-binds requires ICB-capable pipelines + setKernelBuffer encode.
        let all_icb = self
            .commands
            .iter()
            .all(|c| c.pipeline.supportIndirectCommandBuffers());
        if !all_icb {
            if self.freeze_binds {
                return Err(
                    "DecodeIcb freeze_binds: not all pipelines supportIndirectCommandBuffers \
                     (capture with GEMMA_METAL_ICB_PIPELINES=1 / ICB_EXECUTE=1)"
                        .into(),
                );
            }
            self.encoded = true;
            self.optimized = true; // no ICB range to optimize
            return Ok(());
        }
        for (i, cmd) in self.commands.iter().enumerate() {
            let icmd: Retained<ProtocolObject<dyn MTLIndirectComputeCommand>> =
                unsafe { self.icb.indirectComputeCommandAtIndex(i) };
            icmd.reset();
            icmd.setComputePipelineState(&cmd.pipeline);
            if self.freeze_binds {
                for b in &cmd.binds {
                    if let DecodeIcbBind::Buf {
                        index,
                        buf,
                        byte_offset,
                        ..
                    } = b
                    {
                        unsafe {
                            icmd.setKernelBuffer_offset_atIndex(
                                buf.metal(),
                                *byte_offset,
                                *index,
                            );
                        }
                    }
                }
                // GEMV / fused kernels need TG mem on the ICB cmd itself when
                // inheritBuffers=false (encoder setThreadgroupMemory is ignored).
                if let Some((tg_idx, len)) = cmd.tg_mem {
                    unsafe {
                        icmd.setThreadgroupMemoryLength_atIndex(len, tg_idx);
                    }
                }
            }
            icmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                cmd.threadgroups,
                cmd.threads_per_tg,
            );
        }
        self.encoded = true;
        self.optimized = false;
        Ok(())
    }

    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// How many tape cmds have a captured `barrier_after` (range-batch span count
    /// is ≈ this, or +1 if the final cmd has no marker).
    pub fn barrier_after_count(&self) -> usize {
        self.commands.iter().filter(|c| c.barrier_after).count()
    }

    pub fn encoded(&self) -> bool {
        self.encoded
    }

    pub fn execute_count(&self) -> u64 {
        self.execute_count
    }

    /// Captured mini layer/head graph (skips host encode on replay).
    pub fn is_layer_graph(&self) -> bool {
        self.layer_graph
    }

    pub fn status_line(&self) -> String {
        format!(
            "decode_icb cmds={} layer_graph={} encoded={} optimized={} executes={} \
             sticky_buf={}/{} prebuilt={}/{} freeze={} barriers={} coarse_elided={} \
             last_setAddress={}/{} last_setArgTable={} elided={} last_execute_icb={}/{}",
            self.commands.len(),
            self.layer_graph,
            self.encoded,
            self.optimized,
            self.execute_count,
            self.sticky_skippable_binds,
            self.total_buf_binds,
            self.prebuilt_tables.len(),
            self.unique_prebuilt_tables,
            self.freeze_binds,
            self.barrier_after_count(),
            self.barriers_elided,
            self.last_set_address_calls,
            self.last_bind_total,
            self.last_set_argument_table_calls,
            self.last_prebuilt_elided,
            self.last_execute_icb_calls,
            self.last_execute_icb_cmds,
        )
    }

    /// True when Buf binds are frozen into the ICB (`inheritBuffers=false`).
    pub fn freeze_binds(&self) -> bool {
        self.freeze_binds
    }

    /// Unique prebuilt argument tables after fingerprint dedup.
    pub fn unique_prebuilt_table_count(&self) -> usize {
        self.unique_prebuilt_tables
    }

    /// Capture-time Buf binds that sticky replay can skip (`setAddress`).
    pub fn sticky_skippable_binds(&self) -> u64 {
        self.sticky_skippable_binds
    }

    pub fn total_buf_binds(&self) -> u64 {
        self.total_buf_binds
    }

    /// Number of per-command prebuilt argument tables (0 if disabled).
    pub fn prebuilt_table_count(&self) -> usize {
        self.prebuilt_tables.len()
    }

    /// Last execute: binds considered vs `setAddress` calls issued.
    pub fn last_set_address_stats(&self) -> (u64, u64) {
        (self.last_set_address_calls, self.last_bind_total)
    }

    /// Last execute: `setArgumentTable` switches + Buf binds elided by prebuilt.
    pub fn last_prebuilt_stats(&self) -> (u64, u64) {
        (
            self.last_set_argument_table_calls,
            self.last_prebuilt_elided,
        )
    }

    /// Last execute: `executeCommandsInBuffer` call count + cmds covered.
    pub fn last_execute_icb_stats(&self) -> (u64, u64) {
        (self.last_execute_icb_calls, self.last_execute_icb_cmds)
    }

    /// Replay each captured command with arg-table rebinds + recorded barriers.
    ///
    /// **Default (tape honesty):** direct `dispatch` with the Binder-tape
    /// pipeline+grid+binds. Inherit `execute_icb` without freeze is a residual
    /// no-op on mini (parity triage). Opt-in true ICB execute via
    /// `GEMMA_METAL_ICB_EXECUTE=1` (requires ICB-capable pipelines).
    ///
    /// **A2 residual (v0.5.4 / v0.5.7):** default freezes Buf binds into
    /// dedup'd MTL4 argument tables; execute switches `setArgumentTable` only
    /// when the table pointer changes (sticky adopt). Opt-out
    /// `GEMMA_METAL_ICB_PREBUILT_TABLES=0` falls back to sticky `setAddress`.
    ///
    /// **Freeze-binds (v0.5.7, opt-in `GEMMA_METAL_ICB_FREEZE_BINDS=1`):**
    /// classic `setKernelBuffer` into ICB; execute is `execute_icb` × cmds with
    /// **0** `setArgumentTable` (forces ICB execute).
    ///
    /// **Range-batch (v0.5.8, opt-in `GEMMA_METAL_ICB_RANGE_BATCH=1`):** under
    /// freeze-binds, coalesce consecutive cmds between `barrier_after` markers
    /// into one `executeCommandsInBuffer` range (cuts `execute_icb`×N tax).
    ///
    /// **Coarse ranges (v0.5.9):** with range-batch (or explicit
    /// `GEMMA_METAL_ICB_COARSE_RANGES=1`), elide non-interfering `barrier_after`
    /// markers before spanning (large-Buf disjoint test; small scalars ignored).
    ///
    /// **Barriers (v0.5.6):** capture records `barrier_after` from always-on
    /// auto-barriers and explicit RAW `Binder::barrier` calls. Execute skips
    /// auto-barriers and replays those markers — do not force always-on for
    /// every cmd (that regressed shipping-hazard product tok/s).
    pub fn execute(&mut self, rt: &GpuRuntime) -> Result<(), String> {
        if !self.encoded {
            return Err("DecodeIcb: encode before execute".into());
        }
        let freeze = self.freeze_binds;
        let range_batch = freeze && icb_range_batch_enabled();
        if range_batch && icb_coarse_ranges_enabled() && !self.barriers_coarsened {
            self.barriers_elided = Self::elide_non_interfering_barriers(&mut self.commands);
            self.barriers_coarsened = true;
        }
        let use_icb_exec = freeze
            || env_truthy(&["METAL_RUNTIME_ICB_EXECUTE", "GEMMA_METAL_ICB_EXECUTE"])
                .unwrap_or(false);
        let need_opt = !self.optimized;
        let icb = self.icb.clone();
        let n = self.commands.len() as u64;
        let use_prebuilt = !freeze
            && !self.prebuilt_tables.is_empty()
            && self.prebuilt_tables.len() == self.commands.len();

        if use_icb_exec && need_opt {
            let all_icb = self
                .commands
                .iter()
                .all(|c| c.pipeline.supportIndirectCommandBuffers());
            if all_icb {
                rt.with_binder(|bnd| {
                    bnd.optimize_icb(&icb, 0, n);
                    Ok(())
                })?;
            }
        }
        // Skip auto-barriers during tape encode; replay captured `barrier_after`
        // markers instead. Forcing always-on here inflated E4B barriers ~364→599
        // and regressed product tok/s despite lower host encode_us (D16).
        let prev_hazard = crate::ab_flags::hazard_barriers();
        crate::ab_flags::set_hazard_barriers(true);
        let triage = env_truthy(&["GEMMA_METAL_ICB_TRIAGE"]).unwrap_or(false);
        let mut sticky = StickyArgTable::new();
        let mut execute_icb_calls = 0u64;
        let mut execute_icb_cmds = 0u64;
        let exec_result: Result<(), String> = if triage {
            // Per-cmd binder + sync so probe samples localize the first diverge.
            // Sticky/prebuilt resets each with_binder (new encoder scope).
            // Triage never range-batches (needs per-cmd sync points).
            let total = self.commands.len();
            for i in 0..total {
                let cmd = &self.commands[i];
                rt.with_binder(|bnd| {
                    let mut s = StickyArgTable::new();
                    // Eager `then_some(tables[i])` panics when freeze leaves tables empty.
                    let prebuilt = if use_prebuilt {
                        Some(self.prebuilt_tables[i].as_ref())
                    } else {
                        None
                    };
                    let (icb_n, icb_cmds) = Self::encode_cmd(
                        bnd, cmd, &icb, i, use_icb_exec, freeze, prebuilt, &mut s,
                    );
                    execute_icb_calls = execute_icb_calls.saturating_add(icb_n);
                    execute_icb_cmds = execute_icb_cmds.saturating_add(icb_cmds);
                    sticky.bind_total = sticky.bind_total.saturating_add(s.bind_total);
                    sticky.set_calls = sticky.set_calls.saturating_add(s.set_calls);
                    sticky.prebuilt_elided =
                        sticky.prebuilt_elided.saturating_add(s.prebuilt_elided);
                    sticky.set_table_calls =
                        sticky.set_table_calls.saturating_add(s.set_table_calls);
                    Ok(())
                })?;
                if let Some((probe, n)) = self.triage_probe.as_ref() {
                    rt.synchronize()?;
                    let host = probe.read_f32();
                    let take = (*n).min(host.len());
                    let absmean = if take > 0 {
                        host[..take].iter().map(|v| v.abs()).sum::<f32>() / take as f32
                    } else {
                        0.0
                    };
                    eprintln!("[icb_triage] cmd={i}/{total} probe_absmean={absmean:.6}");
                }
            }
            Ok(())
        } else if range_batch {
            // Freeze + range-batch: one executeCommandsInBuffer per safe span
            // between barrier_after markers (inclusive of the barrier cmd).
            let cmds = &self.commands;
            rt.with_binder(|bnd| {
                let mut i = 0usize;
                while i < cmds.len() {
                    let start = i;
                    while i < cmds.len() {
                        let bar = cmds[i].barrier_after;
                        i += 1;
                        if bar {
                            break;
                        }
                    }
                    let count = i - start;
                    // Parent PSO: some MTL4 paths ignore inheritPipelineState=false
                    // without one; each ICB cmd still carries its own frozen PSO.
                    bnd.set_pipeline(&cmds[start].pipeline);
                    bnd.execute_icb_ex(&icb, start as u64, count as u64, false);
                    execute_icb_calls = execute_icb_calls.saturating_add(1);
                    execute_icb_cmds = execute_icb_cmds.saturating_add(count as u64);
                    for cmd in &cmds[start..i] {
                        for b in &cmd.binds {
                            sticky.bind_total = sticky.bind_total.saturating_add(1);
                            if matches!(b, DecodeIcbBind::Buf { .. }) {
                                sticky.prebuilt_elided =
                                    sticky.prebuilt_elided.saturating_add(1);
                            }
                        }
                    }
                    if cmds[i - 1].barrier_after {
                        bnd.barrier();
                    }
                }
                Ok(())
            })
        } else {
            // Pack all tape cmds into one binder scope (A2 sticky / prebuilt / freeze).
            let tables = &self.prebuilt_tables;
            rt.with_binder(|bnd| {
                for (i, cmd) in self.commands.iter().enumerate() {
                    let prebuilt = if use_prebuilt {
                        Some(tables[i].as_ref())
                    } else {
                        None
                    };
                    let (icb_n, icb_cmds) = Self::encode_cmd(
                        bnd, cmd, &icb, i, use_icb_exec, freeze, prebuilt, &mut sticky,
                    );
                    execute_icb_calls = execute_icb_calls.saturating_add(icb_n);
                    execute_icb_cmds = execute_icb_cmds.saturating_add(icb_cmds);
                }
                Ok(())
            })
        };
        crate::ab_flags::set_hazard_barriers(prev_hazard);
        exec_result?;
        self.last_bind_total = sticky.bind_total;
        self.last_set_address_calls = sticky.set_calls;
        self.last_set_argument_table_calls = sticky.set_table_calls;
        self.last_prebuilt_elided = sticky.prebuilt_elided;
        self.last_execute_icb_calls = execute_icb_calls;
        self.last_execute_icb_cmds = execute_icb_cmds;
        if use_icb_exec && need_opt {
            self.optimized = true;
        } else if !use_icb_exec {
            self.optimized = true; // tape-only; no ICB optimize range
        }
        self.execute_count = self.execute_count.saturating_add(1);
        // One-shot visibility for range-batch A/B (default quiet).
        if range_batch && self.execute_count == 1 {
            eprintln!(
                "decode_icb range_batch: cmds={} barriers={} coarse_elided={} \
                 execute_icb={}/{} setArgTable={}",
                self.commands.len(),
                self.barrier_after_count(),
                self.barriers_elided,
                execute_icb_calls,
                execute_icb_cmds,
                sticky.set_table_calls,
            );
        }
        Ok(())
    }

    /// Encode one tape cmd. Returns `(execute_icb_calls, cmds_covered)`.
    #[inline]
    fn encode_cmd(
        bnd: &mut crate::dispatch::Binder<'_>,
        cmd: &DecodeIcbCommand,
        icb: &Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
        i: usize,
        use_icb_exec: bool,
        freeze_binds: bool,
        prebuilt: Option<&ProtocolObject<dyn MTL4ArgumentTable>>,
        sticky: &mut StickyArgTable,
    ) -> (u64, u64) {
        let mut icb_calls = 0u64;
        let mut icb_cmds = 0u64;
        if freeze_binds {
            // Pipeline + kernel buffers + tg_mem already frozen in the ICB cmd.
            // Still set pipeline on the encoder (some MTL4 paths ignore
            // inheritPipelineState=false without a parent PSO). No arg-table latch.
            bnd.set_pipeline(&cmd.pipeline);
            bnd.execute_icb_ex(icb, i as u64, 1, false);
            icb_calls = 1;
            icb_cmds = 1;
            // Count Buf binds as elided (frozen at capture).
            for b in &cmd.binds {
                sticky.bind_total = sticky.bind_total.saturating_add(1);
                if matches!(b, DecodeIcbBind::Buf { .. }) {
                    sticky.prebuilt_elided = sticky.prebuilt_elided.saturating_add(1);
                }
            }
        } else {
            bnd.set_pipeline(&cmd.pipeline);
            if let Some(table) = prebuilt {
                sticky.bind_prebuilt(bnd, table, &cmd.binds);
            } else {
                for b in &cmd.binds {
                    match b {
                        DecodeIcbBind::Buf {
                            index, gpu_addr, ..
                        } => {
                            sticky.bind_addr(bnd, *gpu_addr, *index);
                        }
                        DecodeIcbBind::Immediate { index, bytes } => {
                            sticky.bind_bytes(bnd, bytes, *index);
                        }
                    }
                }
            }
            if let Some((tg_idx, len)) = cmd.tg_mem {
                bnd.set_threadgroup_memory(tg_idx, len);
            }
            if use_icb_exec && cmd.pipeline.supportIndirectCommandBuffers() {
                bnd.execute_icb(icb, i as u64, 1);
                icb_calls = 1;
                icb_cmds = 1;
            } else {
                let _ = (icb, i);
                bnd.dispatch(cmd.threadgroups, cmd.threads_per_tg);
            }
        }
        if cmd.barrier_after {
            bnd.barrier();
        }
        (icb_calls, icb_cmds)
    }
}

/// Build an ICB-capable MTL4 pipeline for `fn_name` from the runtime library.
pub fn pipeline_icb(
    rt: &GpuRuntime,
    fn_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let compiler_desc = MTL4CompilerDescriptor::new();
    let compiler = rt
        .device
        .newCompilerWithDescriptor_error(&compiler_desc)
        .map_err(|e| format!("MTL4Compiler: {e}"))?;

    let func_desc = MTL4LibraryFunctionDescriptor::new();
    func_desc.setName(Some(&NSString::from_str(fn_name)));
    // Prefer overlay (gemma) then primary library.
    let lib = {
        let overlays = rt.overlay_libraries_snapshot()?;
        let fname = NSString::from_str(fn_name);
        let mut found = None;
        for o in &overlays {
            if o.newFunctionWithName(&fname).is_some() {
                found = Some(o.clone());
                break;
            }
        }
        found.unwrap_or_else(|| rt.library.clone())
    };
    func_desc.setLibrary(Some(&lib));

    let pipe_desc = MTL4ComputePipelineDescriptor::new();
    pipe_desc.setComputeFunctionDescriptor(Some(func_desc.as_super()));
    pipe_desc.setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);

    let pipe = compiler
        .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipe_desc, None)
        .map_err(|e| format!("ICB MTL4 pipeline '{fn_name}': {e}"))?;
    if !pipe.supportIndirectCommandBuffers() {
        return Err(format!(
            "ICB pipeline '{fn_name}' supportIndirectCommandBuffers=false"
        ));
    }
    Ok(pipe)
}

// --- Capture tape (thread-local) --------------------------------------------

thread_local! {
    static CAPTURE: std::cell::RefCell<Option<DecodeIcbCapture>> = const { std::cell::RefCell::new(None) };
}

/// Host-side recording of dispatches for later [`DecodeIcb::from_commands`].
#[derive(Default)]
pub struct DecodeIcbCapture {
    pub commands: Vec<DecodeIcbCommand>,
    current_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    current_binds: Vec<DecodeIcbBind>,
    current_tg_mem: Option<(usize, usize)>,
}

impl DecodeIcbCapture {
    pub fn note_pipeline(&mut self, p: Retained<ProtocolObject<dyn MTLComputePipelineState>>) {
        self.current_pipeline = Some(p);
        self.current_binds.clear();
        self.current_tg_mem = None;
    }

    pub fn note_bind(&mut self, index: usize, buf: &GpuBuffer, byte_offset: usize) {
        let bind = DecodeIcbBind::Buf {
            index,
            buf: buf.clone(),
            byte_offset,
            gpu_addr: buf_gpu_addr(buf, byte_offset),
        };
        if let Some(slot) = self.current_binds.iter_mut().find(|b| match b {
            DecodeIcbBind::Buf { index: i, .. } | DecodeIcbBind::Immediate { index: i, .. } => {
                *i == index
            }
        }) {
            *slot = bind;
        } else {
            self.current_binds.push(bind);
        }
    }

    pub fn note_immediate(&mut self, index: usize, bytes: &[u8]) {
        let bind = DecodeIcbBind::Immediate {
            index,
            bytes: bytes.to_vec(),
        };
        if let Some(slot) = self.current_binds.iter_mut().find(|b| match b {
            DecodeIcbBind::Buf { index: i, .. } | DecodeIcbBind::Immediate { index: i, .. } => {
                *i == index
            }
        }) {
            *slot = bind;
        } else {
            self.current_binds.push(bind);
        }
    }

    pub fn note_tg_mem(&mut self, index: usize, length: usize) {
        self.current_tg_mem = Some((index, length));
    }

    pub fn note_dispatch(&mut self, threadgroups: MTLSize, threads_per_tg: MTLSize) {
        let Some(pipeline) = self.current_pipeline.clone() else {
            return;
        };
        self.commands.push(DecodeIcbCommand {
            pipeline,
            threadgroups,
            threads_per_tg,
            binds: std::mem::take(&mut self.current_binds),
            tg_mem: self.current_tg_mem.take(),
            owned_immediates: Vec::new(),
            barrier_after: false,
        });
    }

    /// Mark the most recent command as needing a post-dispatch Device barrier.
    pub fn note_barrier_after(&mut self) {
        if let Some(cmd) = self.commands.last_mut() {
            cmd.barrier_after = true;
        }
    }
}

/// Begin recording dispatches on this thread (Binder hooks).
pub fn begin_decode_icb_capture() {
    CAPTURE.with(|c| {
        *c.borrow_mut() = Some(DecodeIcbCapture::default());
    });
}

/// Finish capture; returns recorded commands (may be empty).
pub fn take_decode_icb_capture() -> Option<DecodeIcbCapture> {
    CAPTURE.with(|c| c.borrow_mut().take())
}

pub fn decode_icb_capture_active() -> bool {
    CAPTURE.with(|c| c.borrow().is_some())
}

pub(crate) fn capture_note_pipeline(p: Retained<ProtocolObject<dyn MTLComputePipelineState>>) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_pipeline(p);
        }
    });
}

pub(crate) fn capture_note_bind(index: usize, buf: &GpuBuffer, byte_offset: usize) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_bind(index, buf, byte_offset);
        }
    });
}

pub(crate) fn capture_note_immediate(index: usize, bytes: &[u8]) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_immediate(index, bytes);
        }
    });
}

pub(crate) fn capture_note_tg_mem(index: usize, length: usize) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_tg_mem(index, length);
        }
    });
}

pub(crate) fn capture_note_dispatch(threadgroups: MTLSize, threads_per_tg: MTLSize) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_dispatch(threadgroups, threads_per_tg);
        }
    });
}

/// Record that the last captured command needs a Device barrier after it.
pub(crate) fn capture_note_barrier() {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_barrier_after();
        }
    });
}

/// -1 = env, 0 = off, 1 = on — force ICB-capable pipelines in the cache.
static ICB_PIPELINES: AtomicI8 = AtomicI8::new(-1);

pub fn set_icb_pipelines(on: bool) {
    ICB_PIPELINES.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

pub fn icb_pipelines_enabled() -> bool {
    let v = ICB_PIPELINES.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_truthy(&["METAL_RUNTIME_ICB_PIPELINES", "GEMMA_METAL_ICB_PIPELINES"])
        .unwrap_or(false);
    ICB_PIPELINES.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// When set, [`crate::runtime::GpuRuntime::with_binder`] is a no-op (no Metal encode).
///
/// Used on the DecodeIcb replay path: re-run the Rust layer loop so `IcbScalarPool`
/// / KV host metadata stay in sync, then [`DecodeIcb::execute`] does GPU work.
static BINDER_ENCODE_NOP: AtomicI8 = AtomicI8::new(0);

pub fn set_binder_encode_nop(on: bool) {
    BINDER_ENCODE_NOP.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

pub fn binder_encode_nop() -> bool {
    BINDER_ENCODE_NOP.load(Ordering::Relaxed) == 1
}

/// RAII: enable binder encode nop; restore off on drop.
pub struct BinderEncodeNopGuard;

impl BinderEncodeNopGuard {
    pub fn enter() -> Self {
        set_binder_encode_nop(true);
        Self
    }
}

impl Drop for BinderEncodeNopGuard {
    fn drop(&mut self) {
        set_binder_encode_nop(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_icb_flag_default_off() {
        set_decode_icb(false);
        assert!(!decode_icb_enabled());
        set_decode_icb(true);
        assert!(decode_icb_enabled());
        set_decode_icb(false);
    }

    #[test]
    fn decode_icb_multi_copy_f32() {
        let rt = GpuRuntime::new().expect("runtime");
        let (mut dec, c) = DecodeIcb::mini_copy_chain(&rt, 32).expect("mini copy chain");
        assert_eq!(dec.command_count(), 2);
        dec.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let n = 32usize;
        let out = unsafe {
            std::slice::from_raw_parts(c.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(out[i], (i as f32) + 1.0, "mismatch at {i}");
        }
        // Second execute without re-encode.
        unsafe {
            let q = c.metal().contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(q, 0xFF, n * 4);
        }
        dec.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let out2 = unsafe {
            std::slice::from_raw_parts(c.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(out2[i], (i as f32) + 1.0);
        }
        assert_eq!(dec.execute_count(), 2);
        let (set_calls, bind_total) = dec.last_set_address_stats();
        let (set_tables, elided) = dec.last_prebuilt_stats();
        // Prefbuilt path: Buf binds elided; mini_copy_chain has no Immediate.
        if dec.freeze_binds() {
            assert_eq!(set_calls, 0, "freeze: no setAddress");
            assert_eq!(set_tables, 0, "freeze: no setArgumentTable");
            assert_eq!(elided, bind_total);
        } else if dec.prebuilt_table_count() == 2 {
            assert_eq!(set_calls, 0, "prebuilt: no setAddress for Buf-only tape");
            // Two cmds have distinct Buf fingerprints → two setArgumentTable
            // (sticky adopt cannot skip). Dedup count == 2.
            assert_eq!(set_tables, 2, "one setArgumentTable per distinct table");
            assert_eq!(elided, bind_total);
            assert_eq!(dec.unique_prebuilt_table_count(), 2);
        }
        eprintln!("decode_icb_multi_copy_f32: {}", dec.status_line());
    }

    #[test]
    fn decode_icb_freeze_binds_zero_arg_table() {
        // Opt-in freeze: classic setKernelBuffer → execute_icb with 0 setArgumentTable.
        // Uses from_commands_ex(freeze=true) — no global flag race with parallel tests.
        let rt = GpuRuntime::new().expect("runtime");
        let (mut dec, c) =
            DecodeIcb::mini_copy_chain_ex(&rt, 32, true).expect("mini copy chain freeze");
        assert!(dec.freeze_binds(), "expected freeze_binds");
        assert_eq!(dec.prebuilt_table_count(), 0);
        dec.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let n = 32usize;
        let out = unsafe {
            std::slice::from_raw_parts(c.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(out[i], (i as f32) + 1.0, "freeze mismatch at {i}");
        }
        let (set_tables, elided) = dec.last_prebuilt_stats();
        let (set_calls, bind_total) = dec.last_set_address_stats();
        assert_eq!(set_tables, 0, "freeze must issue 0 setArgumentTable");
        assert_eq!(set_calls, 0);
        assert_eq!(elided, bind_total);
        eprintln!("decode_icb_freeze_binds_zero_arg_table: {}", dec.status_line());
    }

    #[test]
    fn decode_icb_range_batch_merges_safe_spans() {
        // Three independent copies (no mid-span RAW) + barrier after last:
        // freeze×N = 3 execute_icb; freeze+range_batch = 1.
        let rt = GpuRuntime::new().expect("runtime");
        let n = 32usize;
        let src = rt.alloc_buffer_hot(n * 4).unwrap();
        let a = rt.alloc_buffer_hot(n * 4).unwrap();
        let b = rt.alloc_buffer_hot(n * 4).unwrap();
        let c = rt.alloc_buffer_hot(n * 4).unwrap();
        let n_buf = rt.alloc_buffer_hot(4).unwrap();
        n_buf.write_u32(&[n as u32]);
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i as f32) + 1.0;
            }
            for dst in [&a, &b, &c] {
                std::ptr::write_bytes(dst.metal().contents().as_ptr() as *mut u8, 0, n * 4);
            }
        }
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb");
        let tpt = (pipe.threadExecutionWidth() as usize).min(n).max(1);
        let groups = (n + tpt - 1) / tpt;
        let tg = crate::runtime::mtl_size(groups, 1, 1);
        let tptg = crate::runtime::mtl_size(tpt, 1, 1);
        let buf = |index, buf: &GpuBuffer| DecodeIcbBind::Buf {
            index,
            buf: buf.clone(),
            byte_offset: 0,
            gpu_addr: buf_gpu_addr(buf, 0),
        };
        let mk = |dst: &GpuBuffer, barrier_after: bool| DecodeIcbCommand {
            pipeline: pipe.clone(),
            threadgroups: tg,
            threads_per_tg: tptg,
            binds: vec![buf(0, &src), buf(1, dst), buf(2, &n_buf)],
            tg_mem: None,
            owned_immediates: Vec::new(),
            barrier_after,
        };
        let cmds = vec![mk(&a, false), mk(&b, false), mk(&c, true)];

        set_icb_range_batch(false);
        let mut dec_n = DecodeIcb::from_commands_ex(&rt, cmds.clone(), true).unwrap();
        dec_n.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let (calls_n, covered_n) = dec_n.last_execute_icb_stats();
        assert_eq!(calls_n, 3, "freeze without range_batch: execute_icb × cmds");
        assert_eq!(covered_n, 3);

        set_icb_range_batch(true);
        let mut dec_b = DecodeIcb::from_commands_ex(&rt, cmds, true).unwrap();
        dec_b.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let (calls_b, covered_b) = dec_b.last_execute_icb_stats();
        assert_eq!(calls_b, 1, "range_batch should coalesce to one execute_icb");
        assert_eq!(covered_b, 3);
        assert_eq!(dec_b.last_prebuilt_stats().0, 0, "still 0 setArgumentTable");
        for (label, buf) in [("a", &a), ("b", &b), ("c", &c)] {
            let out = unsafe {
                std::slice::from_raw_parts(buf.metal().contents().as_ptr() as *const f32, n)
            };
            for i in 0..n {
                assert_eq!(out[i], (i as f32) + 1.0, "{label} mismatch at {i}");
            }
        }
        set_icb_range_batch(false);
        eprintln!(
            "decode_icb_range_batch_merges_safe_spans: n={} batch={} | {}",
            dec_n.status_line(),
            dec_b.status_line(),
            "ok"
        );
    }

    #[test]
    fn decode_icb_coarse_ranges_elides_disjoint_keeps_raw() {
        // Disjoint a→b vs c→d: spurious mid barrier elided.
        // Dependent b→e after a→b: RAW barrier kept.
        let rt = GpuRuntime::new().expect("runtime");
        let n = 32usize;
        let a = rt.alloc_buffer_hot(n * 4).unwrap();
        let b = rt.alloc_buffer_hot(n * 4).unwrap();
        let c = rt.alloc_buffer_hot(n * 4).unwrap();
        let d = rt.alloc_buffer_hot(n * 4).unwrap();
        let e = rt.alloc_buffer_hot(n * 4).unwrap();
        let n_ab = rt.alloc_buffer_hot(4).unwrap();
        let n_cd = rt.alloc_buffer_hot(4).unwrap();
        let n_be = rt.alloc_buffer_hot(4).unwrap();
        n_ab.write_u32(&[n as u32]);
        n_cd.write_u32(&[n as u32]);
        n_be.write_u32(&[n as u32]);
        unsafe {
            let pa = a.metal().contents().as_ptr() as *mut f32;
            let pc = c.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *pa.add(i) = (i as f32) + 1.0;
                *pc.add(i) = (i as f32) + 10.0;
            }
            for dst in [&b, &d, &e] {
                std::ptr::write_bytes(dst.metal().contents().as_ptr() as *mut u8, 0, n * 4);
            }
        }
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb");
        let tpt = (pipe.threadExecutionWidth() as usize).min(n).max(1);
        let groups = (n + tpt - 1) / tpt;
        let tg = crate::runtime::mtl_size(groups, 1, 1);
        let tptg = crate::runtime::mtl_size(tpt, 1, 1);
        let buf = |index, buf: &GpuBuffer| DecodeIcbBind::Buf {
            index,
            buf: buf.clone(),
            byte_offset: 0,
            gpu_addr: buf_gpu_addr(buf, 0),
        };
        let mk = |src: &GpuBuffer, dst: &GpuBuffer, nbuf: &GpuBuffer, bar: bool| DecodeIcbCommand {
            pipeline: pipe.clone(),
            threadgroups: tg,
            threads_per_tg: tptg,
            binds: vec![buf(0, src), buf(1, dst), buf(2, nbuf)],
            tg_mem: None,
            owned_immediates: Vec::new(),
            barrier_after: bar,
        };
        // a→b (bar) | c→d (bar) | b→e (bar). First bar is disjoint from c→d → elide.
        // Second bar: span writes include b (from a→b) and d; b→e reads b → keep.
        let cmds = vec![
            mk(&a, &b, &n_ab, true),
            mk(&c, &d, &n_cd, true),
            mk(&b, &e, &n_be, true),
        ];
        let mut probe = cmds.clone();
        let elided = DecodeIcb::elide_non_interfering_barriers(&mut probe);
        assert_eq!(elided, 1, "disjoint a→b vs c→d should elide one barrier");
        assert!(!probe[0].barrier_after, "first barrier elided");
        assert!(probe[1].barrier_after, "RAW b→e barrier kept");
        assert!(probe[2].barrier_after);

        set_icb_range_batch(true);
        set_icb_coarse_ranges(true);
        let mut dec = DecodeIcb::from_commands_ex(&rt, cmds, true).unwrap();
        dec.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let (calls, covered) = dec.last_execute_icb_stats();
        // Spans: [a→b, c→d] + barrier + [b→e] + barrier → 2 execute_icb
        assert_eq!(calls, 2, "coarse+range should yield 2 execute_icb, got {calls}");
        assert_eq!(covered, 3);
        assert_eq!(dec.barriers_elided(), 1);
        let out_b = unsafe {
            std::slice::from_raw_parts(b.metal().contents().as_ptr() as *const f32, n)
        };
        let out_d = unsafe {
            std::slice::from_raw_parts(d.metal().contents().as_ptr() as *const f32, n)
        };
        let out_e = unsafe {
            std::slice::from_raw_parts(e.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(out_b[i], (i as f32) + 1.0, "b mismatch");
            assert_eq!(out_d[i], (i as f32) + 10.0, "d mismatch");
            assert_eq!(out_e[i], (i as f32) + 1.0, "e mismatch (RAW via b)");
        }
        set_icb_range_batch(false);
        set_icb_coarse_ranges(false);
        eprintln!(
            "decode_icb_coarse_ranges_elides_disjoint_keeps_raw: {}",
            dec.status_line()
        );
    }
}
