//! Shared Metal 4 runtime substrate for Gemma inference (and optional reuse by
//! arch_02 training later).
//!
//! Extracted from `arch_02_value_resid/metal-native`: encode path, tensors,
//! TensorOps/simdgroup GEMM, util kernels. **Metal 4-only** encode — residency,
//! packed encoder, no host-zero mid-CB (Audit 4/6 lessons preserved).
//!
//! Quantized `MTLTensor` hooks (WWDC26-330) live in [`mtl_tensor`] — prep only;
//! full Gemma decode / GEMV / FA lands in `gemma-metal`.

#![allow(dead_code)]

pub mod ab_flags;
pub mod cb_replay;
pub mod decode_icb;
pub mod dispatch;
pub mod gemm;
pub mod icb_smoke;
pub mod infer_trace;
pub mod mtl_tensor;
pub mod ops;
pub mod runtime;
pub mod tensor;

pub use cb_replay::{
    cb_replay_api_gap_summary, survey_cb_replay_api_gaps, ArgTableSlot, ArgTableSlotPlan,
    CbReplayApiGap, CbReplayError, CbReplayPhase, CbSlot, IcbCommandTypeHint, IcbReplayStub,
    IcbStubPhase, PingPongCbReplay,
};
pub use decode_icb::{
    begin_decode_icb_capture, binder_encode_nop, decode_icb_capture_active, decode_icb_enabled,
    icb_coarse_ranges_enabled, icb_freeze_binds_enabled, icb_pipelines_enabled,
    icb_range_batch_enabled, pipeline_icb, set_binder_encode_nop, set_decode_icb,
    set_icb_coarse_ranges, set_icb_freeze_binds, set_icb_pipelines, set_icb_range_batch,
    take_decode_icb_capture, BinderEncodeNopGuard, DecodeIcb, DecodeIcbBind, DecodeIcbCapture,
    DecodeIcbCommand,
};
pub use icb_smoke::{
    icb_smoke_enabled, run_copy_f32_smoke, set_icb_smoke, IcbBindBridge, IcbCopySmoke,
};
pub use gemm::{gemm, gemm_f32, GemmBackend};
pub use mtl_tensor::{nax_verify_readiness, NaxVerifyReadiness, QuantDType};
pub use ops::softcap_f32;
pub use runtime::{BufferKind, DeviceMemoryInfo, GpuRuntime, PrecisionMode};
pub use tensor::{DType, GpuBuffer, Tensor};

/// Metallib produced by `build.rs` (absolute path baked at compile time).
pub fn metallib_path() -> &'static str {
    env!("METAL_RUNTIME_METALLIB")
}
