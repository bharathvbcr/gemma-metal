// GPU embed row lookup for Hot Q4 / Q4-MLX banks (avoids host dequant each step).
#include <metal_stdlib>
using namespace metal;

/// `n_tokens` rows: out[m*hidden + d] from token_ids[m]. M=1 keeps legacy dispatch size=hidden.
/// Q4Mlx Hot scale+bias banks are interleaved bfloat2 (one 4B load / group).
kernel void embed_lookup_q4_mlx(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const uint *token_ids [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &group_size [[buffer(6)]],
    constant uint &vocab [[buffer(7)]],
    constant uint &n_tokens [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    (void)biases_unused;
    const uint total = n_tokens * hidden;
    if (gid >= total) return;
    const uint m = gid / hidden;
    const uint d = gid % hidden;
    const uint tid = token_ids[m];
    if (tid >= vocab) {
        out[gid] = 0.0f;
        return;
    }
    const uint groups_per_row = hidden / group_size;
    const uint g = d / group_size;
    const uint scale_i = tid * groups_per_row + g;
    const bfloat2 sbv = sb[scale_i];
    const float scale = float(sbv.x);
    const float bias = float(sbv.y);
    const uint idx = tid * hidden + d;
    const uchar byte = packed[idx / 2u];
    const uchar nibble = ((idx & 1u) == 0u) ? (byte & 0x0fu) : ((byte >> 4) & 0x0fu);
    out[gid] = scale * float(nibble) + bias;
}

kernel void embed_lookup_q4(
    device const uchar *packed [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *zeros [[buffer(2)]],
    device const uint *token_ids [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &group_size [[buffer(6)]],
    constant uint &vocab [[buffer(7)]],
    constant uint &n_tokens [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = n_tokens * hidden;
    if (gid >= total) return;
    const uint m = gid / hidden;
    const uint d = gid % hidden;
    const uint tid = token_ids[m];
    if (tid >= vocab) {
        out[gid] = 0.0f;
        return;
    }
    const uint groups_per_row = hidden / group_size;
    const uint g = d / group_size;
    const uint scale_i = tid * groups_per_row + g;
    const float scale = scales[scale_i];
    const float zero = zeros[scale_i];
    const uint idx = tid * hidden + d;
    const uchar byte = packed[idx / 2u];
    const uchar nibble = ((idx & 1u) == 0u) ? (byte & 0x0fu) : ((byte >> 4) & 0x0fu);
    const int q = (int)(nibble << 28) >> 28;
    out[gid] = scale * ((float)q - zero);
}

/// In-place `x[i] *= scale` (Gemma4 embed_scale √hidden after dequant lookup).
kernel void scale_f32_inplace(
    device float *x [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] *= scale;
}
