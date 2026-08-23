// Portable tiled GEMM via simdgroup_matrix — A/B baseline vs TensorOps.
// C = A @ B for row-major f32 matrices (A: MxK, B: KxN, C: MxN).
//
// Tile geometry: 2×2 simdgroups of 8×8 → 16×16 output per threadgroup.
// Phase 0 tests use shapes divisible by 16; partial-edge handling is deferred.

#include <metal_stdlib>
using namespace metal;

kernel void matmul_simdgroup_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint TM = 8;
    constexpr uint TN = 8;
    constexpr uint TK = 8;
    constexpr uint SG_M = 2;
    constexpr uint SG_N = 2;

    const uint sg_m = sid / SG_N;
    const uint sg_n = sid % SG_N;
    const uint row0 = (tgpig.y * SG_M + sg_m) * TM;
    const uint col0 = (tgpig.x * SG_N + sg_n) * TN;

    // Out-of-range tiles (partial grid) leave their region untouched.
    if (row0 >= M || col0 >= N) {
        return;
    }

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, TM, TN>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += TK) {
        simdgroup_float8x8 a_tile;
        simdgroup_float8x8 b_tile;
        // Assumes K % 8 == 0 and tiles fully in-bounds (Phase 0 contract).
        simdgroup_load(a_tile, A + row0 * K + k0, K, ulong2(0, 0), false);
        simdgroup_load(b_tile, B + k0 * N + col0, N, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }

    simdgroup_store(acc, C + row0 * N + col0, N, ulong2(0, 0), false);
    (void)lane;
}
