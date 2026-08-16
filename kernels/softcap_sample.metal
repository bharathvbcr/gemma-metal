// Logit softcap (default 30) + greedy argmax.
// softcap: y = softcap * tanh(x / softcap)
#include <metal_stdlib>
using namespace metal;

/// In-place softcap over logits[0..n).
/// `softcap` from stable device f32 (ICB / encode-once — not const-arena).
kernel void softcap_logits(
    device float *logits [[buffer(0)]],
    device const float *softcap_ptr [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const float softcap = *softcap_ptr;
    if (gid >= n) return;
    float z = logits[gid] / softcap;
    logits[gid] = softcap * tanh(z);
}

/// Hierarchical argmax: one threadgroup reduces a slice, writes (max, idx) pairs.
/// When `has_idx_in != 0`, `idx_in[i]` carries the original vocab index through
/// subsequent reduce passes (no host remap).
/// When `softcap > 0`, apply softcap to logits[i] before compare (fused first pass).
/// `softcap` from stable device f32 (ICB freeze).
kernel void argmax_f32(
    device const float *logits [[buffer(0)]],
    device uint *out_idx [[buffer(1)]],
    device float *out_val [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    device const uint *idx_in [[buffer(4)]],
    constant uint &has_idx_in [[buffer(5)]],
    device const float *softcap_ptr [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    threadgroup float tg_val[256];
    threadgroup uint tg_idx[256];
    const float softcap = *softcap_ptr;

    const uint base = tgpig * tptg;
    const uint i = base + lid;
    float v = -INFINITY;
    uint idx = 0u;
    if (i < n) {
        v = logits[i];
        if (softcap > 0.0f) {
            float z = v / softcap;
            v = softcap * tanh(z);
        }
        idx = (has_idx_in != 0u) ? idx_in[i] : i;
    }
    tg_val[lid] = v;
    tg_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float a = tg_val[lid];
            float b = tg_val[lid + stride];
            if (b > a || (b == a && tg_idx[lid + stride] < tg_idx[lid])) {
                tg_val[lid] = b;
                tg_idx[lid] = tg_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out_idx[tgpig] = tg_idx[0];
        out_val[tgpig] = tg_val[0];
    }
}

/// Fused softcap + single-TG argmax for n <= 256 (unit tests / small heads).
/// `softcap` from stable device f32 (ICB freeze).
kernel void softcap_sample(
    device float *logits [[buffer(0)]],
    device uint *out_token [[buffer(1)]],
    device const float *softcap_ptr [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float tg_val[256];
    threadgroup uint tg_idx[256];
    const float softcap = *softcap_ptr;

    if (lid < n) {
        float z = logits[lid] / softcap;
        float sc = softcap * tanh(z);
        logits[lid] = sc;
        tg_val[lid] = sc;
        tg_idx[lid] = lid;
    } else {
        tg_val[lid] = -INFINITY;
        tg_idx[lid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float a = tg_val[lid];
            float b = tg_val[lid + stride];
            if (b > a || (b == a && tg_idx[lid + stride] < tg_idx[lid])) {
                tg_val[lid] = b;
                tg_idx[lid] = tg_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        out_token[0] = tg_idx[0];
    }
}

/// Single-pass softcap + argmax for large vocab (e.g. 262144).
/// One threadgroup; each lane scans a strided slice then TG-reduces.
/// Does NOT rewrite logits in place (decode only needs the index).
/// `softcap` from stable device f32 (ICB / encode-once).
kernel void softcap_argmax_one_pass(
    device const float *logits [[buffer(0)]],
    device uint *out_token [[buffer(1)]],
    device const float *softcap_ptr [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float tg_val[1024];
    threadgroup uint tg_idx[1024];
    const float softcap = *softcap_ptr;

    float best = -INFINITY;
    uint best_i = 0u;
    for (uint i = lid; i < n; i += tptg) {
        float v = logits[i];
        if (softcap > 0.0f) {
            float z = v / softcap;
            v = softcap * tanh(z);
        }
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    tg_val[lid] = best;
    tg_idx[lid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float a = tg_val[lid];
            float b = tg_val[lid + stride];
            if (b > a || (b == a && tg_idx[lid + stride] < tg_idx[lid])) {
                tg_val[lid] = b;
                tg_idx[lid] = tg_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        out_token[0] = tg_idx[0];
    }
}
