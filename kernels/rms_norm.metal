// Residual / hidden RMSNorm (f32). One thread per row.
#include <metal_stdlib>
using namespace metal;

/// out[row, :] = rms_norm(x[row, :], weight[:], eps)
kernel void rms_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    device const float *xin = x + gid * dim;
    device float *xout = out + gid * dim;
    float ss = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        float v = xin[d];
        ss += v * v;
    }
    float inv = rsqrt(ss / (float)dim + eps);
    for (uint d = 0; d < dim; ++d) {
        xout[d] = xin[d] * inv * weight[d];
    }
}

/// Same math as rms_norm_f32, writing bf16 (for simd GEMV; kills cast pass).
kernel void rms_norm_bf16(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    device const float *xin = x + gid * dim;
    device bfloat *xout = out + gid * dim;
    float ss = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        float v = xin[d];
        ss += v * v;
    }
    float inv = rsqrt(ss / (float)dim + eps);
    for (uint d = 0; d < dim; ++d) {
        xout[d] = bfloat(xin[d] * inv * weight[d]);
    }
}

/// Cast f32 activations → bf16 for simd GEMV (halves x bandwidth vs f32 stream).
kernel void cast_f32_to_bf16(
    device const float *src [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = bfloat(src[gid]);
}

/// Fused Gemma4 dual-norm residual: `resid[row] += rms_norm(x[row]) * weight`.
/// Collapses rms_norm_f32 + ple_residual_add into one dispatch (31B post-attn/post-ff).
/// When `layer_scale != 1`, folds end-of-layer `x *= scale`:
/// `resid = scale * (resid + rms_norm(x)*weight)`.
kernel void rms_norm_residual_add_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *resid [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    constant float &layer_scale [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    device const float *xin = x + gid * dim;
    device float *xout = resid + gid * dim;
    float ss = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        float v = xin[d];
        ss += v * v;
    }
    float inv = rsqrt(ss / (float)dim + eps);
    const float s = layer_scale;
    if (s == 1.0f) {
        for (uint d = 0; d < dim; ++d) {
            xout[d] += xin[d] * inv * weight[d];
        }
    } else {
        for (uint d = 0; d < dim; ++d) {
            float h = xin[d] * inv * weight[d];
            xout[d] = s * (xout[d] + h);
        }
    }
}
