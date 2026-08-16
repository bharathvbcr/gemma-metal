// Q4 GEMV decode (M=1) with inline group-wise affine dequant.
// Dynamic TG x-cache (cols*4) + one thread per output row + uint4 peel.
// W layout: row-major [rows, cols], two int4 nibbles per byte (lo = even index).
#include <metal_stdlib>
using namespace metal;

constant uint GEMV_TG = 128u;

inline float dequant_q4_nibble(
    device const uchar *packed,
    uint idx,
    float scale,
    float zero)
{
    uchar byte = packed[idx / 2];
    uchar nibble = (idx & 1u) == 0u ? (byte & 0x0fu) : ((byte >> 4) & 0x0fu);
    int q = (int)(nibble << 28) >> 28;
    return scale * ((float)q - zero);
}

inline float dequant_nibble_bits(uchar nibble, float scale, float zero)
{
    int q = (int)(nibble << 28) >> 28;
    return scale * ((float)q - zero);
}

kernel void gemv_q4(
    device const uchar *packed [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *zeros [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float *x_cache [[threadgroup(0)]])
{
    for (uint i = lid; i < cols; i += tptg) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= rows) return;
    const uint row = gid;
    const uint groups_per_row = cols / group_size;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    const uint packed_row = row_base / 2u;

    float acc = 0.0f;
    for (uint g = 0u; g < groups_per_row; ++g) {
        const float scale = scales[scale_base + g];
        const float zero = zeros[scale_base + g];
        const uint xbase = g * group_size;
        const uint pbase = packed_row + (g * group_size) / 2u;
        device const uint *pwords = (device const uint *)(packed + pbase);
        uint i = 0u;
        for (; i + 8u <= group_size; i += 8u) {
            const uint w = pwords[i / 8u];
            // Signed nibble: reinterpret low 4 bits as int4 via sign-extend.
            const float q0 = dequant_nibble_bits((uchar)(w & 0x0fu), scale, zero);
            const float q1 = dequant_nibble_bits((uchar)((w >> 4) & 0x0fu), scale, zero);
            const float q2 = dequant_nibble_bits((uchar)((w >> 8) & 0x0fu), scale, zero);
            const float q3 = dequant_nibble_bits((uchar)((w >> 12) & 0x0fu), scale, zero);
            const float q4 = dequant_nibble_bits((uchar)((w >> 16) & 0x0fu), scale, zero);
            const float q5 = dequant_nibble_bits((uchar)((w >> 20) & 0x0fu), scale, zero);
            const float q6 = dequant_nibble_bits((uchar)((w >> 24) & 0x0fu), scale, zero);
            const float q7 = dequant_nibble_bits((uchar)((w >> 28) & 0x0fu), scale, zero);
            acc += q0 * x_cache[xbase + i];
            acc += q1 * x_cache[xbase + i + 1u];
            acc += q2 * x_cache[xbase + i + 2u];
            acc += q3 * x_cache[xbase + i + 3u];
            acc += q4 * x_cache[xbase + i + 4u];
            acc += q5 * x_cache[xbase + i + 5u];
            acc += q6 * x_cache[xbase + i + 6u];
            acc += q7 * x_cache[xbase + i + 7u];
        }
        for (; i < group_size; ++i) {
            acc += dequant_q4_nibble(packed, row_base + xbase + i, scale, zero)
                * x_cache[xbase + i];
        }
    }
    y[row] = acc;
}

kernel void gemv_q4_tiled(
    device const uchar *packed [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *zeros [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    if (tg >= rows) return;
    threadgroup float partial[GEMV_TG];
    const uint row = tg;
    const uint groups_per_row = cols / group_size;
    float acc = 0.0f;
    for (uint g = tid; g < groups_per_row; g += GEMV_TG) {
        const uint gi = row * groups_per_row + g;
        const float scale = scales[gi];
        const float zero = zeros[gi];
        const uint base = row * cols + g * group_size;
        const uint xbase = g * group_size;
        for (uint i = 0; i < group_size; ++i) {
            acc += dequant_q4_nibble(packed, base + i, scale, zero) * x[xbase + i];
        }
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GEMV_TG / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        y[row] = partial[0];
    }
}
