// Fused QK-Norm → RoPE (and V-norm). Attention scale after QK-Norm is 1.0.
// V-norm is weight RMS only (no extra scale factor).
//
// Layout: q [T, Hq, D], k/v [T, Hkv, D], weights [D] (shared across heads).
// RoPE: proportional NeoX (MLX traditional=False); p-RoPE rotates first
// rotary_dim/2 pairs across dim/2 (inv-freq denom = full head_dim).
#include <metal_stdlib>
using namespace metal;

inline void rms_norm_vec(
    device float *x,
    device const float *weight,
    uint dim,
    float eps)
{
    float ss = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        float v = x[d];
        ss += v * v;
    }
    float inv = rsqrt(ss / (float)dim + eps);
    for (uint d = 0; d < dim; ++d) {
        x[d] = x[d] * inv * weight[d];
    }
}

inline void apply_rope(
    device float *x,
    uint dim,
    uint rotary_dim,
    uint pos,
    float theta)
{
    // Proportional NeoX / non-traditional RoPE (MLX `ProportionalRoPE` +
    // `nn.RoPE(traditional=False)`):
    //   pair x[i] with x[i + dim/2] for i in [0, rotary_dim/2).
    // When rotary_dim == dim (sliding), this is full NeoX over the head.
    // When rotary_dim < dim (global p-RoPE), only the first rotary_dim/2
    // pairs rotate; the rest of the dim/2 pairs stay unrotated (inf freq).
    // inv_freq denom uses full head `dim`.
    const uint half_dim = dim / 2;
    const uint n_pairs = rotary_dim / 2;
    for (uint i = 0; i < n_pairs; ++i) {
        float inv_freq = 1.0f / pow(theta, (2.0f * (float)i) / (float)dim);
        float angle = (float)pos * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = x[i];
        float x1 = x[i + half_dim];
        x[i] = x0 * c - x1 * s;
        x[i + half_dim] = x0 * s + x1 * c;
    }
}

/// One thread per (token, head) for Q; then K/V heads.
kernel void rms_qkv_rope(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_weight [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &Hq [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &rotary_dim [[buffer(10)]],
    constant uint &pos_offset [[buffer(11)]],
    constant float &theta [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total_q = T * Hq;
    const uint total_kv = T * Hkv;
    if (gid >= total_q + 2 * total_kv) return;

    if (gid < total_q) {
        const uint t = gid / Hq;
        const uint h = gid % Hq;
        device float *row = q + ((t * Hq + h) * D);
        rms_norm_vec(row, q_weight, D, eps);
        apply_rope(row, D, rotary_dim, pos_offset + t, theta);
        return;
    }
    uint g2 = gid - total_q;
    if (g2 < total_kv) {
        const uint t = g2 / Hkv;
        const uint h = g2 % Hkv;
        device float *row = k + ((t * Hkv + h) * D);
        rms_norm_vec(row, k_weight, D, eps);
        apply_rope(row, D, rotary_dim, pos_offset + t, theta);
        return;
    }
    g2 -= total_kv;
    {
        const uint t = g2 / Hkv;
        const uint h = g2 % Hkv;
        device float *row = v + ((t * Hkv + h) * D);
        // V-norm: weight RMS only, no RoPE, no attn scale.
        rms_norm_vec(row, v_weight, D, eps);
    }
}

/// Encode-once prototype: RoPE pos from a stable device u32 buffer (written once
/// per decode/verify step) instead of a const-arena scalar rebound every layer.
/// Math identical to `rms_qkv_rope` with `pos_offset = *pos_offset_ptr`.
kernel void rms_qkv_rope_posbuf(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_weight [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &Hq [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &rotary_dim [[buffer(10)]],
    device const uint *pos_offset_ptr [[buffer(11)]],
    constant float &theta [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    const uint pos_offset = *pos_offset_ptr;
    const uint total_q = T * Hq;
    const uint total_kv = T * Hkv;
    if (gid >= total_q + 2 * total_kv) return;

    if (gid < total_q) {
        const uint t = gid / Hq;
        const uint h = gid % Hq;
        device float *row = q + ((t * Hq + h) * D);
        rms_norm_vec(row, q_weight, D, eps);
        apply_rope(row, D, rotary_dim, pos_offset + t, theta);
        return;
    }
    uint g2 = gid - total_q;
    if (g2 < total_kv) {
        const uint t = g2 / Hkv;
        const uint h = g2 % Hkv;
        device float *row = k + ((t * Hkv + h) * D);
        rms_norm_vec(row, k_weight, D, eps);
        apply_rope(row, D, rotary_dim, pos_offset + t, theta);
        return;
    }
    g2 -= total_kv;
    {
        const uint t = g2 / Hkv;
        const uint h = g2 % Hkv;
        device float *row = v + ((t * Hkv + h) * D);
        rms_norm_vec(row, v_weight, D, eps);
    }
}

/// Layer-fusion: `rms_qkv_rope_posbuf` + `kv_store_timestep_pair` for producers.
/// After K/V norm(+RoPE), also write the timestep into the cache slot at
/// `kv_dst_offset`. Q stays scratch-only. Math identical to the two-pass path
/// (scratch first, then copy) — element-local, no grid sync.
/// Opt-in: `GEMMA_METAL_FUSE_ROPE_KV=1` / `GEMMA_METAL_FUSE_LAYER=1`.
kernel void rms_qkv_rope_kv_store(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_weight [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &Hq [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &rotary_dim [[buffer(10)]],
    device const uint *pos_offset_ptr [[buffer(11)]],
    constant float &theta [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    device float *dst_k [[buffer(14)]],
    device float *dst_v [[buffer(15)]],
    device const uint *kv_dst_offset_ptr [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    const uint pos_offset = *pos_offset_ptr;
    const uint kv_dst_offset = *kv_dst_offset_ptr;
    const uint total_q = T * Hq;
    const uint total_kv = T * Hkv;
    if (gid >= total_q + 2 * total_kv) return;

    if (gid < total_q) {
        const uint t = gid / Hq;
        const uint h = gid % Hq;
        device float *row = q + ((t * Hq + h) * D);
        rms_norm_vec(row, q_weight, D, eps);
        apply_rope(row, D, rotary_dim, pos_offset + t, theta);
        return;
    }
    uint g2 = gid - total_q;
    if (g2 < total_kv) {
        const uint t = g2 / Hkv;
        const uint h = g2 % Hkv;
        device float *row = k + ((t * Hkv + h) * D);
        rms_norm_vec(row, k_weight, D, eps);
        apply_rope(row, D, rotary_dim, pos_offset + t, theta);
        const uint base = kv_dst_offset + (t * Hkv + h) * D;
        for (uint d = 0; d < D; ++d) {
            dst_k[base + d] = row[d];
        }
        return;
    }
    g2 -= total_kv;
    {
        const uint t = g2 / Hkv;
        const uint h = g2 % Hkv;
        device float *row = v + ((t * Hkv + h) * D);
        rms_norm_vec(row, v_weight, D, eps);
        const uint base = kv_dst_offset + (t * Hkv + h) * D;
        for (uint d = 0; d < D; ++d) {
            dst_v[base + d] = row[d];
        }
    }
}
