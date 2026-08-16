// Causal sliding-window FlashAttention @ head_dim=128 (DFlash draft stub) (FA-2 tiled).
// Q: [B,Tq,H,D], K/V: [B,Tkv,Hkv,D], O: [B,Tq,H,D]
// Absolute positions: q_abs = q_pos_offset + t_q, k_abs = kv_pos_offset + t_k.
// Prefill dense: Tq=Tkv=T, offsets 0. Decode / ring densify: Tq=1, Tkv=cache_len.
// scale = 1.0 after QK-Norm. Window: q_abs - window < k_abs <= q_abs.
#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM = 128;
// Prefill Tq>1 benefits from BR=8; decode Tq=1 wastes lanes but FA is tiny vs GEMV.
constant uint BR = 8;
constant uint BC = 8;
constant uint D_TILE = 32;

/// Per-token varying FA scalars from stable device u32s (ICB / encode-once).
kernel void flash_attn_swa_h128(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &Tq [[buffer(5)]],
    device const uint *Tkv_ptr [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &window [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    device const uint *q_pos_offset_ptr [[buffer(11)]],
    device const uint *kv_pos_offset_ptr [[buffer(12)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg_vec [[threads_per_threadgroup]])
{
    threadgroup float Oacc[BR * HEAD_DIM];
    threadgroup float Ks[BC * D_TILE];
    threadgroup float Vs[BC * D_TILE];
    threadgroup float scores[BR * BC];

    const uint Tkv = *Tkv_ptr;
    const uint q_pos_offset = *q_pos_offset_ptr;
    const uint kv_pos_offset = *kv_pos_offset_ptr;

    const uint lid = tpitg.x;
    const uint tptg = tptg_vec.x;
    const uint q_block = tgpig.x;
    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = max(H / Hkv, 1u);
    const uint hkv = h / group;

    const uint t_q0 = q_block * BR;
    if (t_q0 >= Tq || Tkv == 0) return;

    const uint t_q = t_q0 + lid;
    const bool row_valid = (lid < BR) && (t_q < Tq);

    if (lid < BR) {
        for (uint d = 0; d < HEAD_DIM; d++) {
            Oacc[lid * HEAD_DIM + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int q_abs = row_valid ? (int)(q_pos_offset + t_q) : -1;
    const int k_lo = row_valid ? max(0, q_abs - (int)window + 1) : 0;
    const int k_hi = row_valid ? q_abs : -1;
    // Union window over the BR tile (uniform for all TG lanes → safe early continue).
    const int q_abs_lo = (int)(q_pos_offset + t_q0);
    const int q_abs_hi = (int)(q_pos_offset + min(t_q0 + BR, Tq) - 1u);
    const int k_lo_blk = max(0, q_abs_lo - (int)window + 1);
    const int k_hi_blk = q_abs_hi;
    const uint n_k_blocks = (Tkv + BC - 1) / BC;

    for (uint kb = 0; kb < n_k_blocks; ++kb) {
        const uint t_k0 = kb * BC;
        const uint n_k = min(BC, Tkv - t_k0);
        // Skip K blocks entirely outside the sliding window (decode / long ctx).
        {
            const int block_lo = (int)(kv_pos_offset + t_k0);
            const int block_hi = block_lo + (int)n_k - 1;
            if (block_hi < k_lo_blk || block_lo > k_hi_blk) {
                continue;
            }
        }

        if (lid < BR * BC) {
            scores[lid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d0 = 0; d0 < HEAD_DIM; d0 += D_TILE) {
            for (uint i = lid; i < n_k * D_TILE; i += tptg) {
                const uint tk = i / D_TILE;
                const uint d = i % D_TILE;
                const uint k_off = ((b * Tkv + (t_k0 + tk)) * Hkv + hkv) * HEAD_DIM;
                Ks[tk * D_TILE + d] = K[k_off + d0 + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row_valid) {
                const uint q_off = ((b * Tq + t_q) * H + h) * HEAD_DIM;
                for (uint tk = 0; tk < n_k; ++tk) {
                    float s = 0.0f;
                    for (uint d = 0; d < D_TILE; ++d) {
                        s += Q[q_off + d0 + d] * Ks[tk * D_TILE + d];
                    }
                    scores[lid * BC + tk] += s;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float m_block = -INFINITY;
        if (row_valid) {
            for (uint tk = 0; tk < n_k; ++tk) {
                const int k_abs = (int)(kv_pos_offset + t_k0 + tk);
                float score = scores[lid * BC + tk] * scale;
                if (k_abs < k_lo || k_abs > k_hi) {
                    score = -INFINITY;
                }
                scores[lid * BC + tk] = score;
                m_block = max(m_block, score);
            }
            const float m_new = max(m_i, m_block);
            const float alpha = exp(m_i - m_new);
            float l_block = 0.0f;
            for (uint tk = 0; tk < n_k; ++tk) {
                float p = (scores[lid * BC + tk] > -INFINITY)
                    ? exp(scores[lid * BC + tk] - m_new)
                    : 0.0f;
                scores[lid * BC + tk] = p;
                l_block += p;
            }
            for (uint d = 0; d < HEAD_DIM; ++d) {
                Oacc[lid * HEAD_DIM + d] *= alpha;
            }
            l_i = l_i * alpha + l_block;
            m_i = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d0 = 0; d0 < HEAD_DIM; d0 += D_TILE) {
            for (uint i = lid; i < n_k * D_TILE; i += tptg) {
                const uint tk = i / D_TILE;
                const uint d = i % D_TILE;
                const uint v_off = ((b * Tkv + (t_k0 + tk)) * Hkv + hkv) * HEAD_DIM;
                Vs[tk * D_TILE + d] = V[v_off + d0 + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row_valid) {
                for (uint tk = 0; tk < n_k; ++tk) {
                    const float p = scores[lid * BC + tk];
                    for (uint d = 0; d < D_TILE; ++d) {
                        Oacc[lid * HEAD_DIM + d0 + d] += p * Vs[tk * D_TILE + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (row_valid) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        const uint o_off = ((b * Tq + t_q) * H + h) * HEAD_DIM;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            O[o_off + d] = Oacc[lid * HEAD_DIM + d] * inv_l;
        }
    }
    (void)B;
}
