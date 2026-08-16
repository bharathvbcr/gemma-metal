// Fused gate×up with SiLU / swish (DFlash / Qwen3 draft MLP).
// silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
#include <metal_stdlib>
using namespace metal;

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/// out[i] = silu(gate[i]) * up[i]
kernel void mlp_silu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = silu(gate[gid]) * up[gid];
}
