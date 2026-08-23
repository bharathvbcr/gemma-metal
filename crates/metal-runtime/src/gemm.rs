//! GEMM dispatch: TensorOps `matmul2d` (preferred) or `simdgroup_matrix` fallback.
//!
//! GEMM v2: Morton 1D TG walk, packed zero+matmul (one binder), MLP/bf16 split-K,
//! `execution_simdgroups<4>` on bf16/relaxed kernels (see matmul_tensorops.metal).
//!
//! Phase H: `PrecisionMode::Bf16` uses bf16 TensorOps GEMMs (f32 accumulate).
//! Callers may keep persistent bf16 activation/weight buffers; `ensure_bf16`
//! is a no-op when the operand is already bf16. Residual/RMSNorm/CE stay f32.
//! Optional `relaxed_precision` (tf32-class) on f32 GEMMs as a bridge; off by
//! default for golden parity.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use crate::tensor::{DType, Tensor};

/// Tall-K / small-MN → split-K accumulate.
/// Attn dW: M=N=128, K=BT=4096. MLP dW: one side = mlp_dim=384.
fn prefer_tn_splitk(m: usize, n: usize, k: usize) -> bool {
    k >= 2048 && m <= 384 && n <= 384 && m.min(n) <= 128
}

/// Tile sizes for TensorOps kernels (must match matmul_tensorops.metal).
#[derive(Clone, Copy)]
struct TileGeom {
    sm: usize,
    sn: usize,
    /// Simdgroups per TG (`execution_simdgroups<N>`). Exact f32 uses 1.
    simdgroups: usize,
}

const TILE_F32: TileGeom = TileGeom {
    sm: 32,
    sn: 32,
    simdgroups: 1,
};
const TILE_V2: TileGeom = TileGeom {
    sm: 64,
    sn: 32,
    simdgroups: 4,
};

/// Exact 1D TG count for a `tiles_n × tiles_m` rectangle (no power-of-two pad —
/// padding blew up tall NN shapes like BT×C and erased the binder win).
fn morton_tg_count(tiles_n: usize, tiles_m: usize) -> usize {
    tiles_n.saturating_mul(tiles_m).max(1)
}

/// Live TN/NT TensorOps descriptors (transpose_left/right). Fixed multi-tile
/// slice axes: TN slices A's M on dim0; NT slices B's N on dim1.
const USE_TN_NT_DESCRIPTORS: bool = true;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmBackend {
    /// MPP TensorOps `matmul2d` (Metal 4 / macOS 26+, M5 accelerators).
    TensorOps,
    /// Hand-tiled `simdgroup_matrix` portable path.
    Simdgroup,
}

impl GemmBackend {
    pub fn kernel_name_f32(self) -> &'static str {
        match self {
            GemmBackend::TensorOps => "matmul2d_tensorops_f32",
            GemmBackend::Simdgroup => "matmul_simdgroup_f32",
        }
    }

    pub fn kernel_name_f32_relaxed(self) -> &'static str {
        match self {
            GemmBackend::TensorOps => "matmul2d_tensorops_f32_relaxed",
            GemmBackend::Simdgroup => "matmul_simdgroup_f32",
        }
    }

    pub fn kernel_name_bf16(self) -> &'static str {
        match self {
            GemmBackend::TensorOps => "matmul2d_tensorops_bf16_f32",
            // No simdgroup bf16 kernel — callers cast to f32 first.
            GemmBackend::Simdgroup => "matmul_simdgroup_f32",
        }
    }
}

/// Pick TensorOps when the metallib contains it; else simdgroup.
pub fn select_backend(rt: &GpuRuntime) -> GemmBackend {
    if rt.has_tensorops() {
        GemmBackend::TensorOps
    } else {
        GemmBackend::Simdgroup
    }
}

/// Cast f32 tensor → bf16 (GPU). Used at GEMM boundaries under `PrecisionMode::Bf16`.
pub fn cast_f32_to_bf16(src: &Tensor) -> Result<Tensor, String> {
    assert_eq!(src.dtype, DType::F32);
    let rt = src.runtime();
    let dst = rt.alloc_tensor_bf16(&src.shape)?;
    cast_f32_to_bf16_into(src, &dst)?;
    Ok(dst)
}

/// Cast into an existing bf16 buffer (persistent weight banks).
pub fn cast_f32_to_bf16_into(src: &Tensor, dst: &Tensor) -> Result<(), String> {
    assert_eq!(src.dtype, DType::F32);
    assert_eq!(dst.dtype, DType::BF16);
    assert_eq!(src.shape, dst.shape);
    let rt = src.runtime();
    let p = rt.pipeline("cast_f32_to_bf16")?;
    let n = src.numel();
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })?;
    Ok(())
}

/// Hot-resident bf16 clone of an f32 master (weights / EMA banks).
pub fn cast_f32_to_bf16_hot(src: &Tensor) -> Result<Tensor, String> {
    assert_eq!(src.dtype, DType::F32);
    let rt = src.runtime();
    let dst = rt.alloc_tensor_bf16_hot(&src.shape)?;
    cast_f32_to_bf16_into(src, &dst)?;
    Ok(dst)
}

/// Cast bf16 tensor → f32 (GPU).
pub fn cast_bf16_to_f32(src: &Tensor) -> Result<Tensor, String> {
    assert_eq!(src.dtype, DType::BF16);
    let rt = src.runtime();
    let dst = rt.alloc_tensor_f32(&src.shape)?;
    let p = rt.pipeline("cast_bf16_to_f32")?;
    let n = src.numel();
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, &dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })?;
    Ok(dst)
}

fn ensure_bf16(t: &Tensor) -> Result<Tensor, String> {
    match t.dtype {
        DType::BF16 => Ok(t.clone()),
        DType::F32 => cast_f32_to_bf16(t),
    }
}

fn use_bf16_gemm(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.precision() == PrecisionMode::Bf16
        && backend == GemmBackend::TensorOps
        && rt.has_tensorops()
}

fn use_relaxed_f32(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.relaxed_precision()
        && rt.precision() == PrecisionMode::F32
        && backend == GemmBackend::TensorOps
        && rt.has_tensorops()
}

/// C[M,N] = A[M,K] @ B[K,N]. Overwrites C.
///
/// - f32×f32→f32 always supported (exact or relaxed via runtime flag)
/// - bf16×bf16→f32 accum (C must be f32) via TensorOps when available
pub fn gemm(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(c.shape.len(), 2);
    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];
    assert_eq!(k, k2, "inner dims must match");
    assert_eq!(c.shape, &[m, n]);

    let use_bf16 = a.dtype == DType::BF16 && b.dtype == DType::BF16;
    if use_bf16 {
        assert_eq!(c.dtype, DType::F32, "bf16 GEMM accumulates into f32 C");
        if backend != GemmBackend::TensorOps {
            return Err(
                "bf16 GEMM requires TensorOps backend (simdgroup is f32-only)".into(),
            );
        }
    } else {
        assert_eq!(a.dtype, DType::F32);
        assert_eq!(b.dtype, DType::F32);
        assert_eq!(c.dtype, DType::F32);
    }

    match backend {
        GemmBackend::TensorOps => {}
        GemmBackend::Simdgroup => {
            if m % 16 != 0 || n % 16 != 0 || k % 8 != 0 {
                return Err(format!(
                    "simdgroup GEMM requires M,N % 16 == 0 and K % 8 == 0; got {m}x{k} @ {k}x{n}"
                ));
            }
        }
    }

    let rt = a.runtime();
    let kernel = if use_bf16 {
        backend.kernel_name_bf16()
    } else if use_relaxed_f32(rt, backend) {
        backend.kernel_name_f32_relaxed()
    } else {
        backend.kernel_name_f32()
    };
    let pipeline = rt.pipeline(kernel)?;

    match backend {
        GemmBackend::TensorOps => {
            let tile = if use_bf16 || use_relaxed_f32(rt, backend) {
                TILE_V2
            } else {
                TILE_F32
            };
            // Zero-tax: pack C-zero + matmul into one binder (~−1 binder/GEMM).
            dispatch_tensorops_nn(rt, &pipeline, a, b, c, m, n, k, tile)?;
        }
        GemmBackend::Simdgroup => {
            // simdgroup overwrites C; still zero for determinism on views.
            gpu_zero_tensor(c)?;
            let m_u = m as u32;
            let n_u = n as u32;
            let k_u = k as u32;
            let (tg_w, tg_h, tpt) = threadgroup_geometry_simdgroup(&pipeline, m, n);
            rt.with_binder(|bnd| {
                bnd.set_pipeline(&pipeline);
                bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
                bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
                bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
                bnd.bind_u32(m_u, 3);
                bnd.bind_u32(n_u, 4);
                bnd.bind_u32(k_u, 5);
                bnd.dispatch(mtl_size(tg_w, tg_h, 1), mtl_size(tpt, 1, 1));
                Ok(())
            })?;
        }
    }

    Ok(())
}

/// Pack `zero_f32(C)` + TensorOps NN matmul into a single Metal 4 binder.
fn dispatch_tensorops_nn(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
) -> Result<(), String> {
    let zero_p = rt.pipeline("zero_f32")?;
    let numel = c.numel();
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);
    let z_width = zero_p.threadExecutionWidth() as usize;
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = (numel + z_tpt - 1) / z_tpt;

    rt.with_binder(|bnd| {
        bnd.set_pipeline(&zero_p);
        bnd.bind_tensor(c, 0);
        bnd.bind_u32(numel as u32, 1);
        bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
        // Explicit barrier only when auto per-dispatch barriers are off.
        if crate::ab_flags::hazard_barriers() {
            bnd.barrier();
        }

        bnd.set_pipeline(pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        // f32 exact NN/TN/NT read buffer(8); bf16/relaxed ignore extra bind.
        bnd.bind_u32(
            if crate::ab_flags::gemm_interior_offsets() {
                1
            } else {
                0
            },
            8,
        );
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

fn dispatch_tensorops_tn_nt(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
) -> Result<(), String> {
    // Same binder packing as NN.
    dispatch_tensorops_nn(rt, pipeline, a, b, c, m, n, k, tile)
}

/// TensorOps matmul with `mode::multiply_accumulate` — no C zero (1 binder).
fn dispatch_tensorops_accum(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
    bind_interior: bool,
) -> Result<(), String> {
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);

    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        if bind_interior {
            bnd.bind_u32(
                if crate::ab_flags::gemm_interior_offsets() {
                    1
                } else {
                    0
                },
                8,
            );
        }
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// Convenience: f32 GEMM (parity path). Honors `relaxed_precision` when set.
pub fn gemm_f32(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    gemm(a, b, c, backend)
}

/// Training GEMM: under `PrecisionMode::Bf16` uses bf16 TensorOps (f32 accum into
/// `c`). Already-bf16 operands skip cast (persistent bf16 activations/weights).
/// Falls back to f32 GEMM when TensorOps is absent.
pub fn gemm_train(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let rt = a.runtime();
    if use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a)?;
        let b_bf = ensure_bf16(b)?;
        assert_eq!(c.dtype, DType::F32);
        return gemm(&a_bf, &b_bf, c, backend);
    }
    gemm_f32(a, b, c, backend)
}

fn gpu_zero_tensor(t: &Tensor) -> Result<(), String> {
    let rt = t.runtime();
    let n = t.numel();
    // Prefer GPU zero when async encoding is on (avoids host memset mid-batch).
    if rt.async_encode_enabled() {
        let p = rt.pipeline("zero_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
            crate::dispatch::set_tensor(bnd, t, 0);
            crate::dispatch::set_u32(bnd, n as u32, 1);
        })?;
    } else {
        // Views may have byte_offset — only zero the logical window via kernel.
        let p = rt.pipeline("zero_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
            crate::dispatch::set_tensor(bnd, t, 0);
            crate::dispatch::set_u32(bnd, n as u32, 1);
        })?;
    }
    Ok(())
}

/// C[M,N] = A[K,M]^T @ B[K,N] (TN). A is stored [K,M], B [K,N].
pub fn gemm_tn_f32(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    assert_eq!(a_km.shape.len(), 2);
    assert_eq!(b_kn.shape.len(), 2);
    let k = a_km.shape[0];
    let m = a_km.shape[1];
    assert_eq!(b_kn.shape[0], k);
    let n = b_kn.shape[1];
    assert_eq!(c.shape, &[m, n]);

    if USE_TN_NT_DESCRIPTORS
        && backend == GemmBackend::TensorOps
        && a_km.runtime().has_tensorops()
    {
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_f32(a_km, b_kn, c, k);
        }
        let rt = a_km.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_km, b_kn, c, m, n, k, TILE_F32);
    }

    // Default: explicit transpose + NN (golden-safe).
    let at = {
        let rt = a_km.runtime();
        let out = rt.alloc_temp_f32(&[m, k])?;
        let p = rt.pipeline("transpose2d_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, m * k, |bnd| {
            crate::dispatch::set_tensor(bnd, a_km, 0);
            crate::dispatch::set_tensor(bnd, &out, 1);
            crate::dispatch::set_u32(bnd, k as u32, 2);
            crate::dispatch::set_u32(bnd, m as u32, 3);
        })?;
        out
    };
    gemm_f32(&at, b_kn, c, backend)
}

/// Training TN GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_tn_train(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let rt = a_km.runtime();
    if use_bf16_gemm(rt, backend) {
        assert_eq!(c.dtype, DType::F32);
        let a_bf = ensure_bf16(a_km)?;
        let b_bf = ensure_bf16(b_kn)?;
        let k = a_bf.shape[0];
        let m = a_bf.shape[1];
        let n = b_bf.shape[1];
        assert_eq!(c.shape, &[m, n]);
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_bf16(&a_bf, &b_bf, c, k);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_bf16_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_V2);
    }
    gemm_tn_f32(a_km, b_kn, c, backend)
}

fn gemm_tn_splitk_f32(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
) -> Result<(), String> {
    gemm_tn_splitk_f32_opts(a_km, b_kn, c, k, /*zero_first=*/ true)
}

fn gemm_tn_splitk_f32_opts(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
    zero_first: bool,
) -> Result<(), String> {
    let m = a_km.shape[1];
    let n = b_kn.shape[1];
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_f32")?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tile = TILE_F32;
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    let numel = c.numel();
    let z_width = zero_p.threadExecutionWidth() as usize;
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = (numel + z_tpt - 1) / z_tpt;
    let k_tile = 256u32;
    let partitions: Vec<u32> = (0..k as u32).step_by(k_tile as usize).collect();

    // Zero once (optional) + all K-partitions in one binder.
    let need_explicit = crate::ab_flags::hazard_barriers();
    rt.with_binder(|bnd| {
        if zero_first {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            if need_explicit {
                bnd.barrier();
            }
        }

        bnd.set_pipeline(&pipeline);
        for (pi, &k0) in partitions.iter().enumerate() {
            if pi > 0 && need_explicit {
                bnd.barrier();
            }
            bnd.bind_buf(a_km.buffer.metal(), a_km.byte_offset, 0);
            bnd.bind_buf(b_kn.buffer.metal(), b_kn.byte_offset, 1);
            bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
            bnd.bind_u32(m as u32, 3);
            bnd.bind_u32(n as u32, 4);
            bnd.bind_u32(k as u32, 5);
            bnd.bind_u32(k0, 6);
            bnd.bind_u32(k_tile, 7);
            bnd.bind_u32(tiles_n as u32, 8);
            bnd.bind_u32(tiles_m as u32, 9);
            bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;
    Ok(())
}

fn gemm_tn_splitk_bf16(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
) -> Result<(), String> {
    gemm_tn_splitk_bf16_opts(a_km, b_kn, c, k, /*zero_first=*/ true)
}

fn gemm_tn_splitk_bf16_opts(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
    zero_first: bool,
) -> Result<(), String> {
    let m = a_km.shape[1];
    let n = b_kn.shape[1];
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_bf16_f32")?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tile = TILE_V2;
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    let numel = c.numel();
    let z_width = zero_p.threadExecutionWidth() as usize;
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = (numel + z_tpt - 1) / z_tpt;
    let k_tile = 256u32;
    let partitions: Vec<u32> = (0..k as u32).step_by(k_tile as usize).collect();

    let need_explicit = crate::ab_flags::hazard_barriers();
    rt.with_binder(|bnd| {
        if zero_first {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            if need_explicit {
                bnd.barrier();
            }
        }

        bnd.set_pipeline(&pipeline);
        for (pi, &k0) in partitions.iter().enumerate() {
            if pi > 0 && need_explicit {
                bnd.barrier();
            }
            bnd.bind_buf(a_km.buffer.metal(), a_km.byte_offset, 0);
            bnd.bind_buf(b_kn.buffer.metal(), b_kn.byte_offset, 1);
            bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
            bnd.bind_u32(m as u32, 3);
            bnd.bind_u32(n as u32, 4);
            bnd.bind_u32(k as u32, 5);
            bnd.bind_u32(k0, 6);
            bnd.bind_u32(k_tile, 7);
            bnd.bind_u32(tiles_n as u32, 8);
            bnd.bind_u32(tiles_m as u32, 9);
            bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;
    Ok(())
}

/// C[M,N] = A[M,K] @ B[N,K]^T (NT). B is stored [N,K] (e.g. W[in,out]).
pub fn gemm_nt_f32(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    assert_eq!(a_mk.shape.len(), 2);
    assert_eq!(b_nk.shape.len(), 2);
    let m = a_mk.shape[0];
    let k = a_mk.shape[1];
    let n = b_nk.shape[0];
    assert_eq!(b_nk.shape[1], k);
    assert_eq!(c.shape, &[m, n]);

    if USE_TN_NT_DESCRIPTORS
        && backend == GemmBackend::TensorOps
        && a_mk.runtime().has_tensorops()
    {
        let rt = a_mk.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_mk, b_nk, c, m, n, k, TILE_F32);
    }

    let bt = {
        let rt = b_nk.runtime();
        let out = rt.alloc_temp_f32(&[k, n])?;
        let p = rt.pipeline("transpose2d_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, n * k, |bnd| {
            crate::dispatch::set_tensor(bnd, b_nk, 0);
            crate::dispatch::set_tensor(bnd, &out, 1);
            crate::dispatch::set_u32(bnd, n as u32, 2);
            crate::dispatch::set_u32(bnd, k as u32, 3);
        })?;
        out
    };
    gemm_f32(a_mk, &bt, c, backend)
}

/// Training NT GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_nt_train(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let rt = a_mk.runtime();
    if use_bf16_gemm(rt, backend) {
        assert_eq!(c.dtype, DType::F32);
        let a_bf = ensure_bf16(a_mk)?;
        let b_bf = ensure_bf16(b_nk)?;
        let m = a_bf.shape[0];
        let k = a_bf.shape[1];
        let n = b_bf.shape[0];
        assert_eq!(c.shape, &[m, n]);
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_bf16_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_V2);
    }
    gemm_nt_f32(a_mk, b_nk, c, backend)
}

/// C += A[K,M]^T @ B[K,N] (TN accumulate). No C zero — for dW into grad banks
/// and dx accumulate into a pre-zeroed buffer.
pub fn gemm_tn_accum_train(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    assert_eq!(a_km.shape.len(), 2);
    assert_eq!(b_kn.shape.len(), 2);
    let k = a_km.shape[0];
    let m = a_km.shape[1];
    assert_eq!(b_kn.shape[0], k);
    let n = b_kn.shape[1];
    assert_eq!(c.shape, &[m, n]);
    assert_eq!(c.dtype, DType::F32);

    let rt = a_km.runtime();
    let use_accum = crate::ab_flags::gemm_accum();
    if use_accum && use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a_km)?;
        let b_bf = ensure_bf16(b_kn)?;
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_bf16_opts(&a_bf, &b_bf, c, k, /*zero_first=*/ false);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_accum_bf16_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_V2, /*bind_interior=*/ false,
        );
    }

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops()
    {
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_f32_opts(a_km, b_kn, c, k, /*zero_first=*/ false);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_accum_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, a_km, b_kn, c, m, n, k, TILE_F32, /*bind_interior=*/ true,
        );
    }

    // Fallback / Soft-bisect: temp + add (pre–Audit 6 P1a/P1a2 numerics).
    let tmp = rt.alloc_temp_f32(&[m, n])?;
    gemm_tn_train(a_km, b_kn, &tmp, backend)?;
    let p = rt.pipeline("add_inplace_f32")?;
    crate::dispatch::dispatch_1d(rt, &p, c.numel(), |bnd| {
        crate::dispatch::set_tensor(bnd, c, 0);
        crate::dispatch::set_tensor(bnd, &tmp, 1);
        crate::dispatch::set_u32(bnd, c.numel() as u32, 2);
    })?;
    Ok(())
}

/// C += A[M,K] @ B[N,K]^T (NT accumulate). No C zero.
pub fn gemm_nt_accum_train(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    assert_eq!(a_mk.shape.len(), 2);
    assert_eq!(b_nk.shape.len(), 2);
    let m = a_mk.shape[0];
    let k = a_mk.shape[1];
    let n = b_nk.shape[0];
    assert_eq!(b_nk.shape[1], k);
    assert_eq!(c.shape, &[m, n]);
    assert_eq!(c.dtype, DType::F32);

    let rt = a_mk.runtime();
    let use_accum = crate::ab_flags::gemm_accum();
    if use_accum && use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a_mk)?;
        let b_bf = ensure_bf16(b_nk)?;
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_accum_bf16_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_V2, /*bind_interior=*/ false,
        );
    }

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops()
    {
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_accum_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, a_mk, b_nk, c, m, n, k, TILE_F32, /*bind_interior=*/ true,
        );
    }

    let tmp = rt.alloc_temp_f32(&[m, n])?;
    gemm_nt_train(a_mk, b_nk, &tmp, backend)?;
    let p = rt.pipeline("add_inplace_f32")?;
    crate::dispatch::dispatch_1d(rt, &p, c.numel(), |bnd| {
        crate::dispatch::set_tensor(bnd, c, 0);
        crate::dispatch::set_tensor(bnd, &tmp, 1);
        crate::dispatch::set_u32(bnd, c.numel() as u32, 2);
    })?;
    Ok(())
}

/// Prefer bf16 / relaxed GEMM per runtime precision policy.
pub fn gemm_auto(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    gemm_train(a, b, c, backend)
}

fn threads_per_tg(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    tile: TileGeom,
) -> usize {
    let width = pipeline.threadExecutionWidth() as usize;
    width * tile.simdgroups
}

fn threadgroup_geometry_simdgroup(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    m: usize,
    n: usize,
) -> (usize, usize, usize) {
    let width = pipeline.threadExecutionWidth() as usize;
    let tg_w = (n + 15) / 16;
    let tg_h = (m + 15) / 16;
    (tg_w, tg_h, width * 4)
}

/// CPU reference GEMM for tests.
pub fn gemm_f32_cpu(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuRuntime;

    fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
        got.iter()
            .zip(exp.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max)
    }

    fn run_case(m: usize, n: usize, k: usize, backend: GemmBackend) {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        eprintln!(
            "device={} encode=Metal4 tensorops={} backend={:?}",
            rt.device_name(),
            rt.has_tensorops(),
            backend
        );

        let mut a_host = vec![0.0f32; m * k];
        let mut b_host = vec![0.0f32; k * n];
        for i in 0..a_host.len() {
            a_host[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_host.len() {
            b_host[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_host, &b_host, m, n, k);

        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_host);
        b.buffer.write_f32(&b_host);

        gemm_f32(&a, &b, &c, backend).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        assert!(
            err < 1e-4,
            "GEMM {m}x{k}@{k}x{n} backend={backend:?} max_abs_err={err}"
        );
    }

    #[test]
    fn gemm_simdgroup_16() {
        run_case(16, 16, 16, GemmBackend::Simdgroup);
    }

    #[test]
    fn gemm_simdgroup_32() {
        run_case(32, 32, 32, GemmBackend::Simdgroup);
    }

    #[test]
    fn gemm_auto_small() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let backend = select_backend(&rt);
        let dim = if backend == GemmBackend::TensorOps {
            32
        } else {
            16
        };
        run_case(dim, dim, dim, backend);
    }

    #[test]
    fn gemm_tensorops_32_if_available() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping TensorOps test: kernel not in metallib");
            return;
        }
        run_case(32, 32, 64, GemmBackend::TensorOps);
        run_case(64, 32, 32, GemmBackend::TensorOps);
    }

    #[test]
    fn gemm_bf16_tensorops_if_available() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping bf16 TensorOps test");
            return;
        }
        rt.set_precision(crate::runtime::PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for i in 0..a_f.len() {
            a_f[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_f.len() {
            b_f[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
        let a = rt.alloc_tensor_bf16(&[m, k]).unwrap();
        let b = rt.alloc_tensor_bf16(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer
            .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&a_f));
        b.buffer
            .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&b_f));
        gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        // bf16 rounding — looser than f32
        assert!(err < 2e-2, "bf16 GEMM max_abs_err={err}");
    }

    /// Phase H: `gemm_train` under Bf16 casts f32 masters → bf16 TensorOps.
    #[test]
    fn gemm_train_bf16_casts_f32_operands() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping gemm_train bf16 test");
            return;
        }
        rt.set_precision(PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for i in 0..a_f.len() {
            a_f[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_f.len() {
            b_f[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_f);
        b.buffer.write_f32(&b_f);
        gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        assert!(err < 2e-2, "gemm_train bf16 max_abs_err={err}");
    }

    /// Phase H bridge: `relaxed_precision` numerics vs exact f32 / CPU.
    /// Kept behind a flag for train; documents whether 1e-5 goldens survive.
    #[test]
    fn gemm_relaxed_precision_numerics() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping relaxed_precision test");
            return;
        }
        // Ensure pipeline exists (metallib rebuilt with Phase H kernel).
        if rt.pipeline("matmul2d_tensorops_f32_relaxed").is_err() {
            eprintln!("skipping: matmul2d_tensorops_f32_relaxed not in metallib");
            return;
        }
        let m = 64usize;
        let n = 64usize;
        let k = 128usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for i in 0..a_f.len() {
            a_f[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_f.len() {
            b_f[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);

        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c_exact = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let c_relax = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_f);
        b.buffer.write_f32(&b_f);

        rt.set_precision(PrecisionMode::F32);
        rt.set_relaxed_precision(false);
        gemm_f32(&a, &b, &c_exact, GemmBackend::TensorOps).unwrap();
        rt.set_relaxed_precision(true);
        gemm_f32(&a, &b, &c_relax, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let got_exact = c_exact.buffer.read_f32();
        let got_relax = c_relax.buffer.read_f32();
        let err_exact = max_abs_err(&got_exact, &expected);
        let err_relax = max_abs_err(&got_relax, &expected);
        let err_vs_exact = max_abs_err(&got_relax, &got_exact);
        eprintln!(
            "relaxed_precision: err_vs_cpu_exact={err_exact:.3e} err_vs_cpu_relax={err_relax:.3e} \
             err_relax_vs_exact={err_vs_exact:.3e}"
        );
        assert!(err_exact < 1e-4, "exact f32 GEMM drifted: {err_exact}");
        // Smoke: relaxed must be finite and within a generous bound (tf32-class).
        assert!(
            err_relax < 5e-2,
            "relaxed GEMM too far from CPU: {err_relax}"
        );
        // Document 1e-5 golden gate: if this fails, keep --tf32 off for parity.
        if err_relax >= 1e-5 {
            eprintln!(
                "NOTE: relaxed_precision breaks 1e-5 golden atol (err={err_relax:.3e}); \
                 leave flag off for f32 parity / enable only for throughput experiments"
            );
        } else {
            eprintln!("relaxed_precision within 1e-5 of CPU on this shape");
        }
        rt.set_relaxed_precision(false);
    }

    #[test]
    fn gemm_train_bf16_awkward_k() {
        // sota shapes: bigram_dim=48, ve_dim=24 — must not NaN under bf16 TensorOps.
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping awkward-K bf16 test");
            return;
        }
        rt.set_precision(PrecisionMode::Bf16);
        for (m, n, k) in [(64usize, 128usize, 48usize), (64, 128, 24), (4096, 128, 48)] {
            let mut a_f = vec![0.0f32; m * k];
            let mut b_f = vec![0.0f32; k * n];
            for i in 0..a_f.len() {
                a_f[i] = ((i % 17) as f32) * 0.01 - 0.08;
            }
            for i in 0..b_f.len() {
                b_f[i] = ((i % 13) as f32) * 0.007 - 0.04;
            }
            let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
            let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer.write_f32(&a_f);
            b.buffer.write_f32(&b_f);
            gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            let n_bad = got.iter().filter(|x| !x.is_finite()).count();
            let err = max_abs_err(&got, &expected);
            eprintln!("bf16 awkward {m}x{k}@{k}x{n}: nonfinite={n_bad} err={err:.3e}");
            assert_eq!(n_bad, 0, "NaN/Inf in bf16 GEMM {m}x{k}@{k}x{n}");
            assert!(err < 5e-2, "bf16 awkward K err={err}");
        }
    }

    #[test]
    fn gemm_tn_nt_bf16_train_smoke() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping tn/nt bf16 smoke");
            return;
        }
        if rt.pipeline("matmul2d_tensorops_tn_bf16_f32").is_err() {
            eprintln!("skipping: tn/nt bf16 kernels missing");
            return;
        }
        rt.set_precision(PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        // TN: A[K,M], B[K,N] → C[M,N]
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..a_km.len() {
            a_km[i] = ((i % 11) as f32) * 0.05 - 0.2;
        }
        for i in 0..b_kn.len() {
            b_kn[i] = ((i % 7) as f32) * 0.04 - 0.1;
        }
        // CPU: C = A^T @ B
        let mut a_mk = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                a_mk[j * k + i] = a_km[i * m + j];
            }
        }
        let exp_tn = gemm_f32_cpu(&a_mk, &b_kn, m, n, k);
        let a_t = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b_t = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c_tn = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a_t.buffer.write_f32(&a_km);
        b_t.buffer.write_f32(&b_kn);
        gemm_tn_train(&a_t, &b_t, &c_tn, GemmBackend::TensorOps).unwrap();

        // NT: A[M,K], B[N,K] → C[M,N]
        let mut b_nk = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                b_nk[i * k + j] = b_kn[j * n + i];
            }
        }
        let mut b_kn_from_nk = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                b_kn_from_nk[j * n + i] = b_nk[i * k + j];
            }
        }
        let exp_nt = gemm_f32_cpu(&a_mk, &b_kn_from_nk, m, n, k);
        let a_n = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b_n = rt.alloc_tensor_f32(&[n, k]).unwrap();
        let c_nt = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a_n.buffer.write_f32(&a_mk);
        b_n.buffer.write_f32(&b_nk);
        gemm_nt_train(&a_n, &b_n, &c_nt, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let err_tn = max_abs_err(&c_tn.buffer.read_f32(), &exp_tn);
        let err_nt = max_abs_err(&c_nt.buffer.read_f32(), &exp_nt);
        assert!(err_tn < 2e-2, "tn bf16 err={err_tn}");
        assert!(err_nt < 2e-2, "nt bf16 err={err_nt}");
    }

    fn gemm_tn_cpu(a_km: &[f32], b_kn: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut a_mk = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                a_mk[j * k + i] = a_km[i * m + j];
            }
        }
        gemm_f32_cpu(&a_mk, b_kn, m, n, k)
    }

    fn gemm_nt_cpu(a_mk: &[f32], b_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                b_kn[j * n + i] = b_nk[i * k + j];
            }
        }
        gemm_f32_cpu(a_mk, &b_kn, m, n, k)
    }

    #[test]
    fn gemm_tn_nt_tensorops_descriptors() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping tn/nt descriptor test");
            return;
        }
        for (m, n, k) in [(32usize, 32, 64), (64, 128, 128), (128, 128, 256)] {
            let mut a_km = vec![0.0f32; k * m];
            let mut b_kn = vec![0.0f32; k * n];
            for i in 0..a_km.len() {
                a_km[i] = ((i % 11) as f32) * 0.05 - 0.2;
            }
            for i in 0..b_kn.len() {
                b_kn[i] = ((i % 7) as f32) * 0.04 - 0.1;
            }
            let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
            let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer.write_f32(&a_km);
            b.buffer.write_f32(&b_kn);
            gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let err = max_abs_err(&c.buffer.read_f32(), &exp);
            assert!(err < 1e-4, "TN desc {m}x{k}^T@{k}x{n} err={err}");

            let mut a_mk = vec![0.0f32; m * k];
            let mut b_nk = vec![0.0f32; n * k];
            for i in 0..m {
                for j in 0..k {
                    a_mk[i * k + j] = ((i * k + j) % 13) as f32 * 0.03 - 0.15;
                }
            }
            for i in 0..n {
                for j in 0..k {
                    b_nk[i * k + j] = ((i * k + j) % 17) as f32 * 0.02 - 0.1;
                }
            }
            let exp_nt = gemm_nt_cpu(&a_mk, &b_nk, m, n, k);
            let a2 = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b2 = rt.alloc_tensor_f32(&[n, k]).unwrap();
            let c2 = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a2.buffer.write_f32(&a_mk);
            b2.buffer.write_f32(&b_nk);
            gemm_nt_f32(&a2, &b2, &c2, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let err_nt = max_abs_err(&c2.buffer.read_f32(), &exp_nt);
            assert!(err_nt < 1e-4, "NT desc {m}x{k}@{n}x{k}^T err={err_nt}");
        }
    }

    #[test]
    fn gemm_tn_splitk_tall_dw_shape() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping split-K test");
            return;
        }
        // dW-shaped: M=N=128, K=4096 (BT).
        let m = 128usize;
        let n = 128usize;
        let k = 4096usize;
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..a_km.len() {
            a_km[i] = ((i % 19) as f32) * 0.01 - 0.08;
        }
        for i in 0..b_kn.len() {
            b_kn[i] = ((i % 23) as f32) * 0.008 - 0.05;
        }
        let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
        let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_km);
        b.buffer.write_f32(&b_kn);
        assert!(prefer_tn_splitk(m, n, k));
        gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let err = max_abs_err(&c.buffer.read_f32(), &exp);
        assert!(err < 1e-3, "split-K TN dW shape err={err}");
    }

    #[test]
    fn gemm_tn_splitk_mlp_dw_shape() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping MLP split-K test");
            return;
        }
        // MLP-up dW: M=128, N=384, K=4096
        let m = 128usize;
        let n = 384usize;
        let k = 4096usize;
        assert!(prefer_tn_splitk(m, n, k));
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..a_km.len() {
            a_km[i] = ((i % 19) as f32) * 0.01 - 0.08;
        }
        for i in 0..b_kn.len() {
            b_kn[i] = ((i % 23) as f32) * 0.008 - 0.05;
        }
        let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
        let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_km);
        b.buffer.write_f32(&b_kn);
        gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let err = max_abs_err(&c.buffer.read_f32(), &exp);
        assert!(err < 1e-3, "MLP-up split-K TN err={err}");
    }
}
