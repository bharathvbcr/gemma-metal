//! Small device ops built on the shared util metallib (`softcap_f32`, …).

use std::sync::Arc;

use crate::dispatch::{dispatch_1d, set_f32, set_tensor, set_u32};
use crate::runtime::GpuRuntime;
use crate::tensor::Tensor;

/// `post = softcap * tanh(pre / softcap)` (elementwise).
pub fn softcap_f32(
    rt: &Arc<GpuRuntime>,
    pre: &Tensor,
    softcap: f32,
) -> Result<Tensor, String> {
    let post = rt.alloc_tensor_f32(&pre.shape)?;
    let p = rt.pipeline("softcap_f32")?;
    let n = pre.numel();
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, pre, 0);
        set_tensor(bnd, &post, 1);
        set_f32(bnd, softcap, 2);
        set_u32(bnd, n as u32, 3);
    })?;
    Ok(post)
}
