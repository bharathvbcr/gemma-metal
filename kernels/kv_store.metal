// GPU-resident KV: store one timestep and optional ring densify (no host roundtrip).
#include <metal_stdlib>
using namespace metal;

/// dst[dst_offset + i] = src[i] for i in [0, n).
/// `dst_offset` is a stable device u32 (ICB / encode-once — not const-arena).
kernel void kv_store_timestep(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    device const uint *dst_offset_ptr [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint dst_offset = *dst_offset_ptr;
    if (gid >= n) return;
    dst[dst_offset + gid] = src[gid];
}

/// Store K and V timesteps in one dispatch (producer hot path).
/// `dst_offset` from stable device u32 (ICB freeze).
kernel void kv_store_timestep_pair(
    device const float *src_k [[buffer(0)]],
    device const float *src_v [[buffer(1)]],
    device float *dst_k [[buffer(2)]],
    device float *dst_v [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    device const uint *dst_offset_ptr [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint dst_offset = *dst_offset_ptr;
    if (gid >= n) return;
    dst_k[dst_offset + gid] = src_k[gid];
    dst_v[dst_offset + gid] = src_v[gid];
}

/// Chronological densify from a ring: dst[t] = src[(start+t) % capacity].
/// `filled` / `start` are stable device u32s (per-step varying; ICB freeze).
kernel void kv_ring_densify(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &n_slot [[buffer(2)]],
    constant uint &capacity [[buffer(3)]],
    device const uint *filled_ptr [[buffer(4)]],
    device const uint *start_ptr [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint filled = *filled_ptr;
    const uint start = *start_ptr;
    const uint total = filled * n_slot;
    if (gid >= total) return;
    const uint t = gid / n_slot;
    const uint e = gid % n_slot;
    const uint src_t = (start + t) % capacity;
    dst[gid] = src[src_t * n_slot + e];
}
