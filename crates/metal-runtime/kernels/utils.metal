// Shared util kernels extracted from metal-native (copy/cast/zero/softcap/transpose).
#include <metal_stdlib>
using namespace metal;

kernel void copy_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = in[gid];
}

kernel void copy_bf16(
    device const bfloat *in [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = in[gid];
}

kernel void cast_f32_to_bf16(
    device const float *in [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = bfloat(in[gid]);
}

kernel void cast_bf16_to_f32(
    device const bfloat *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = float(in[gid]);
}

kernel void zero_f32(
    device float *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = 0.0f;
}

kernel void add_inplace_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] += src[gid];
}

kernel void transpose2d_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows * cols) return;
    uint i = gid / cols;
    uint j = gid % cols;
    out[j * rows + i] = in[i * cols + j];
}

/// logits_post = softcap * tanh(logits / softcap).
kernel void softcap_f32(
    device const float *pre [[buffer(0)]],
    device float *post [[buffer(1)]],
    constant float &softcap [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float z = pre[gid] / softcap;
    post[gid] = softcap * tanh(z);
}
