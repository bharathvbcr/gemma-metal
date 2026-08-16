// Fused gate×up with gelu_pytorch_tanh (NOT SiLU / swish).
// gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// Root cause of prior NaNs: with -O2, MSL `tanh` lowers to `air.fast_tanh`,
// which NaNs for |arg| ≳ ~10. Host/PyTorch use a saturating precise tanh.
// The gelu inner term at |x|≈20 is ~301 → fast_tanh → NaN mid even when
// gate/up are finite. Fix: clamp x (x³ overflow) + precise::tanh on a
// clamped inner (tanh already saturates by |z|≳8).
#include <metal_stdlib>
using namespace metal;

/// File-local (`static`) so metallib link does not ODR-merge with
/// `gelu_pytorch_tanh` in gemv_q4_mlx.metal.
static inline float gelu_pytorch_tanh_mlp(float x) {
    // |x|>~50 → x³ overflows f32 → Inf → NaN when fused with up.
    float xc = clamp(x, -20.0f, 20.0f);
    float x3 = xc * xc * xc;
    float inner = 0.7978845608028654f * (xc + 0.044715f * x3);
    // precise::tanh (not fast_tanh); clamp is belt+suspenders for any path
    // that still softens math.
    float t = precise::tanh(clamp(inner, -10.0f, 10.0f));
    return 0.5f * xc * (1.0f + t);
}

/// out[i] = gelu_pytorch_tanh(gate[i]) * up[i]
kernel void mlp_gelu_tanh(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = gelu_pytorch_tanh_mlp(gate[gid]) * up[gid];
}

/// Same as mlp_gelu_tanh, writing bf16 (down-proj GEMV input; kills cast pass).
kernel void mlp_gelu_tanh_bf16(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = bfloat(gelu_pytorch_tanh_mlp(gate[gid]) * up[gid]);
}
