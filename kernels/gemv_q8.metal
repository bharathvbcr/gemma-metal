// Q8 GEMV decode (M=1) with inline group-wise affine dequant.
// W layout: row-major [rows, cols] as signed int8 bytes.
// scales/zeros: [rows * (cols / group_size)], group along K (cols).
#include <metal_stdlib>
using namespace metal;

/// y[rows] = W[rows, cols] @ x[cols] with Q8 weights.
kernel void gemv_q8(
    device const char *packed [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *zeros [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint groups_per_row = cols / group_size;
    const uint row = gid;
    float acc = 0.0f;
    for (uint g = 0; g < groups_per_row; ++g) {
        const uint gi = row * groups_per_row + g;
        const float scale = scales[gi];
        const float zero = zeros[gi];
        const uint base = row * cols + g * group_size;
        for (uint i = 0; i < group_size; ++i) {
            const float w = scale * ((float)packed[base + i] - zero);
            acc += w * x[g * group_size + i];
        }
    }
    y[row] = acc;
}
