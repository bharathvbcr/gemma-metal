//! MTLTensor / quantized TensorOps prep (WWDC26-330).
//!
//! Metal 4 can feed native quantized tensors into TensorOps matmul (auto-dequant
//! on NAX). Prefer this for **prefill** GEMM once Q4 banks exist; keep hand GEMV
//! for decode (M=1) until proven otherwise.
//!
//! objc2-metal 0.3 exposes `MTLTensorDataType::{Int8, …}` today. Int4 / FP8 E8M0
//! scale planes land in later SDKs — callers should treat [`QuantDType`] as the
//! stable surface and map when bindings appear.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::NSInteger;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLResourceID, MTLResourceOptions, MTLSizeAndAlign, MTLTensor,
    MTLTensorDataType, MTLTensorDescriptor, MTLTensorExtents, MTLTensorUsage,
};
use std::sync::Arc;

use crate::dispatch::Binder;
use crate::runtime::GpuRuntime;
use crate::tensor::GpuBuffer;

/// Logical quantized element type for future TensorOps / GEMV paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantDType {
    /// Native `MTLTensorDataTypeInt8` (macOS 26+).
    Int8,
    /// Planned WWDC26-330 Int4 tensor path — not in objc2-metal 0.3 yet.
    Int4,
    /// Planned FP8 E8M0 scale-plane path (macOS 27+ per Apple notes).
    Fp8E8M0,
}

impl QuantDType {
    /// Map to objc2 `MTLTensorDataType` when the SDK binding exists.
    pub fn to_mtl(self) -> Result<MTLTensorDataType, String> {
        match self {
            QuantDType::Int8 => Ok(MTLTensorDataType::Int8),
            QuantDType::Int4 => Err(
                "NAX/TensorOps Q4 unbound: MTLTensorDataType Int4 not in objc2-metal 0.3 \
                 (WWDC26-330). Decode stays on hand simdgroup Q4 GEMV/GEMM; do not claim \
                 TensorOps Int4 is shipped. Await SDK / binding bump."
                    .into(),
            ),
            QuantDType::Fp8E8M0 => Err(
                "FP8 E8M0 MTLTensor scale planes require newer Metal SDK (macOS 27+ notes)"
                    .into(),
            ),
        }
    }

    pub fn bytes_per_elem_hint(self) -> f32 {
        match self {
            QuantDType::Int8 => 1.0,
            QuantDType::Int4 => 0.5,
            QuantDType::Fp8E8M0 => 1.0,
        }
    }
}

/// Snapshot of TensorOps / NAX readiness for verify(M) / prefill planning.
///
/// Int4 is **unbound** in objc2-metal 0.3 — verify(M) remains on hand simdgroup Q4
/// GEMM (`gemm_q4_mlx_simd*`). DDTree stays parked until verify(M) flattens (it will
/// not with current simdgroup Q4 alone).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NaxVerifyReadiness {
    pub int8_tensorops_dtype: bool,
    pub int4_tensorops_dtype: bool,
    pub fp8_e8m0_tensorops_dtype: bool,
    pub quant_prefill_gemm_wired: bool,
    pub note: &'static str,
}

/// Probe dtype binding + documented wire status (no device `newTensor` calls).
pub fn nax_verify_readiness() -> NaxVerifyReadiness {
    NaxVerifyReadiness {
        int8_tensorops_dtype: QuantDType::Int8.to_mtl().is_ok(),
        int4_tensorops_dtype: QuantDType::Int4.to_mtl().is_ok(),
        fp8_e8m0_tensorops_dtype: QuantDType::Fp8E8M0.to_mtl().is_ok(),
        quant_prefill_gemm_wired: try_quant_tensorops_prefill_gemm_status().is_ok(),
        note: "Int4 unbound in objc2-metal 0.3; verify(M) = hand simdgroup Q4; TensorOps Q4 not shipped",
    }
}

fn try_quant_tensorops_prefill_gemm_status() -> Result<(), String> {
    // Same sentinel as try_quant_tensorops_prefill_gemm — no device touch.
    Err("quant TensorOps prefill GEMM not wired yet".into())
}

/// Owned MTLTensor handle (device-allocated or buffer-backed).
pub struct GpuTensor {
    pub tensor: Retained<ProtocolObject<dyn MTLTensor>>,
    pub dtype: MTLTensorDataType,
    pub dims: Vec<usize>,
    /// When buffer-backed, keep the storage alive.
    pub(crate) storage: Option<GpuBuffer>,
    pub(crate) runtime: Arc<GpuRuntime>,
}

impl GpuTensor {
    pub fn gpu_resource_id(&self) -> MTLResourceID {
        self.tensor.gpuResourceID()
    }

    pub fn data_type(&self) -> MTLTensorDataType {
        self.tensor.dataType()
    }
}

/// Bind an MTLTensor into the Metal 4 argument table via `setResource:atBufferIndex:`.
pub fn bind_mtl_tensor(bnd: &mut Binder<'_>, t: &GpuTensor, index: usize) {
    unsafe {
        bnd.bind_resource_id(t.gpu_resource_id(), index);
    }
}

/// Probe whether the device can size an MTLTensor with the given dtype/shape.
///
/// **Experimental:** some objc2 / SDK combinations have SIGSEGV'd on this
/// selector for unsupported layouts — gate behind Phase-2 smoke before use.
pub fn probe_tensor_support(
    rt: &GpuRuntime,
    dtype: QuantDType,
    dims: &[usize],
) -> Result<MTLSizeAndAlign, String> {
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute)?;
    Ok(rt.device.tensorSizeAndAlignWithDescriptor(&desc))
}

/// Allocate a device-backed MTLTensor (no storage shared with [`GpuBuffer`]).
pub fn alloc_device_tensor(
    rt: &Arc<GpuRuntime>,
    dims: &[usize],
    dtype: QuantDType,
) -> Result<GpuTensor, String> {
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute)?;
    let tensor = rt
        .device
        .newTensorWithDescriptor_error(&desc)
        .map_err(|e| format!("newTensorWithDescriptor: {e}"))?;
    Ok(GpuTensor {
        tensor,
        dtype: mtl_dtype,
        dims: dims.to_vec(),
        storage: None,
        runtime: Arc::clone(rt),
    })
}

/// Wrap an existing shared buffer as an MTLTensor view (offset must satisfy align).
///
/// # Safety
/// `byte_offset` must match `tensorSizeAndAlignWithDescriptor` alignment for `dtype`.
pub unsafe fn tensor_from_buffer(
    rt: &Arc<GpuRuntime>,
    buf: &GpuBuffer,
    byte_offset: usize,
    dims: &[usize],
    dtype: QuantDType,
) -> Result<GpuTensor, String> {
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute)?;
    let align = rt.device.tensorSizeAndAlignWithDescriptor(&desc);
    if byte_offset % (align.align as usize) != 0 {
        return Err(format!(
            "tensor buffer offset {byte_offset} not aligned to {}",
            align.align
        ));
    }
    let tensor = buf
        .metal()
        .newTensorWithDescriptor_offset_error(&desc, byte_offset as _)
        .map_err(|e| format!("buffer.newTensor: {e}"))?;
    Ok(GpuTensor {
        tensor,
        dtype: mtl_dtype,
        dims: dims.to_vec(),
        storage: Some(buf.clone()),
        runtime: Arc::clone(rt),
    })
}

fn make_descriptor(
    dims: &[usize],
    dtype: MTLTensorDataType,
    usage: MTLTensorUsage,
) -> Result<Retained<MTLTensorDescriptor>, String> {
    if dims.is_empty() || dims.len() > 16 {
        return Err(format!("MTLTensor rank {} out of range 1..=16", dims.len()));
    }
    let desc = MTLTensorDescriptor::new();
    let extents = extents_from_dims(dims)?;
    desc.setDimensions(&extents);
    desc.setDataType(dtype);
    desc.setUsage(usage);
    desc.setResourceOptions(MTLResourceOptions::StorageModeShared);
    Ok(desc)
}

fn extents_from_dims(dims: &[usize]) -> Result<Retained<MTLTensorExtents>, String> {
    let values: Vec<NSInteger> = dims.iter().map(|&d| d as NSInteger).collect();
    let extents = unsafe {
        MTLTensorExtents::initWithRank_values(
            MTLTensorExtents::alloc(),
            values.len() as _,
            values.as_ptr(),
        )
    }
    .ok_or_else(|| "MTLTensorExtents::initWithRank_values failed".to_string())?;
    Ok(extents)
}

/// Documented entry point for Phase 2: try native quant TensorOps prefill GEMM.
///
/// Returns `Err` until gemma-metal wires Q4 banks + TensorOps quant kernels.
pub fn try_quant_tensorops_prefill_gemm(
    _rt: &GpuRuntime,
    _a: &GpuTensor,
    _b: &GpuTensor,
    _c: &crate::tensor::Tensor,
) -> Result<(), String> {
    Err(
        "quant TensorOps prefill GEMM not wired yet (Phase 2); use hand GEMV for decode"
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_dtype_int8_maps() {
        assert!(QuantDType::Int8.to_mtl().is_ok());
        let err = QuantDType::Int4.to_mtl().unwrap_err();
        assert!(err.contains("Int4"), "{err}");
        assert!(err.contains("unbound") || err.contains("not in objc2"), "{err}");
        assert!(QuantDType::Fp8E8M0.to_mtl().is_err());
    }

    #[test]
    fn nax_verify_readiness_int4_unbound() {
        let r = nax_verify_readiness();
        assert!(r.int8_tensorops_dtype);
        assert!(!r.int4_tensorops_dtype);
        assert!(!r.fp8_e8m0_tensorops_dtype);
        assert!(!r.quant_prefill_gemm_wired);
        assert!(r.note.contains("Int4 unbound"));
    }

    /// Descriptor construction only (no device call). Full
    /// `tensorSizeAndAlign` / `newTensor` A/B is Phase 2 — objc2 bindings can
    /// SIGSEGV on some SDK/runtime combos when probing unsupported layouts.
    #[test]
    fn quant_descriptor_builds_for_int8() {
        let desc = make_descriptor(&[64, 64], MTLTensorDataType::Int8, MTLTensorUsage::Compute)
            .expect("descriptor");
        assert_eq!(desc.dataType(), MTLTensorDataType::Int8);
        assert_eq!(desc.dimensions().rank(), 2);
    }
}
