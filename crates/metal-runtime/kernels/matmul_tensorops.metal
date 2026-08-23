// TensorOps GEMM via Metal Performance Primitives (M5 neural accelerators).
// Primary path for Phase 0+; requires Metal 4 / macOS 26+.
//
// GEMM v2 (MPP §2.3):
//   - Morton 1D threadgroup walk (cache-friendly tile traversal)
//   - execution_simdgroups<4> on bf16 / relaxed hot paths (64×32 TG tiles)
//   - BK=128 cooperative K-accumulate for large K (interior tiles)
//   - Compile-time tile extents via offset+dextents{SN,SM} (pointer tensors
//     lack static_slice; this is the equivalent bounds-check elision)
//   - mode::multiply still needs C zeroed once (packed with matmul on host)
//
// Note: device pointers must be non-const — `const` poisons MPP type matching.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// Decode Morton/Z-order code → (x, y) tile coordinates.
inline uint2 morton_decode_2d(uint c) {
    uint x = 0, y = 0;
#pragma unroll
    for (uint i = 0; i < 16; ++i) {
        x |= ((c >> (2 * i)) & 1u) << i;
        y |= ((c >> (2 * i + 1)) & 1u) << i;
    }
    return uint2(x, y);
}

/// Decode linear TG id → (x, y) tile. Uses Morton when the grid is square and
/// power-of-two (cache-friendly); otherwise compact row-major (avoids pad tax).
inline uint2 tile_from_linear(uint linear, uint tiles_n, uint tiles_m) {
    if (tiles_n == tiles_m && tiles_n != 0u && (tiles_n & (tiles_n - 1u)) == 0u) {
        return morton_decode_2d(linear);
    }
    return uint2(linear % tiles_n, linear / tiles_n);
}

// =============================================================================
// f32 exact — execution_simdgroup, SM=SN=32 (golden-safe)
// =============================================================================

kernel void matmul2d_tensorops_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    // Interior offset tensors measured slower on M5 Pro f32 training shapes;
    // gated by host METAL_NATIVE_GEMM_INTERIOR=1.
    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// Phase H bridge: f32 GEMM with `relaxed_precision` (tf32-class).
/// GEMM v2: execution_simdgroups<4>, 64×32 tiles, Morton, BK, static tile extents.
kernel void matmul2d_tensorops_f32_relaxed(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr int BK = 128;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, true,
                            matmul2d_descriptor::mode::multiply);
    constexpr auto desc_bk =
        matmul2d_descriptor(SM, SN, BK, false, false, true,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    matmul2d<desc_bk, execution_simdgroups<4>> op_bk;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    bool use_bk = interior && ((int)K >= BK);

    if (use_bk) {
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        int k = 0;
        for (; k + BK <= (int)K; k += BK) {
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                             array<int, 2>{1, (int)N});
            op_bk.run(tA, tB, tC);
        }
        if (k < (int)K) {
            int k_rem = (int)K - k;
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{k_rem, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, k_rem},
                             array<int, 2>{1, (int)N});
            constexpr auto desc_tail =
                matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, true,
                                    matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc_tail, execution_simdgroups<4>> op_tail;
            op_tail.run(tA, tB, tC);
        }
    } else if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] = A_stored[K,M]^T @ B[K,N] (TN).
/// Physical A[K,M]: extents {M,K} strides {1,M}. transpose_left → [M,K].
kernel void matmul2d_tensorops_tn_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        // A physical [K,M] row-major; MPP TN view extents {M,K} stride {1,M}.
        // Tile origin (ty, 0) in that view → pointer A + ty (col-major-ish dim0).
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] = A[M,K] @ B_stored[N,K]^T (NT).
kernel void matmul2d_tensorops_nt_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        // B physical [N,K]; MPP NT view extents {K,N} stride {1,K}.
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A_stored[K,M]^T @ B[K,N] (TN accumulate; no C zero).
kernel void matmul2d_tensorops_tn_accum_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A[M,K] @ B_stored[N,K]^T (NT accumulate; no C zero).
kernel void matmul2d_tensorops_nt_accum_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// Split-K TN accumulate for one K-partition.
kernel void matmul2d_tensorops_tn_splitk_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &k0 [[buffer(6)]],
    constant uint &k_tile [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto mmul_mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false, mmul_mode);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    uint k_len = min(k_tile, K - k0);
    auto mA = tensor(A + k0 * M, dextents<int, 2>{(int)M, (int)k_len}, array<int, 2>{1, (int)M});
    auto mB = tensor(B + k0 * N, dextents<int, 2>{(int)N, (int)k_len}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

// =============================================================================
// bf16 → f32 accum — execution_simdgroups<4>, 64×32 tiles
// =============================================================================

kernel void matmul2d_tensorops_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr int BK = 128;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                            matmul2d_descriptor::mode::multiply);
    constexpr auto desc_bk =
        matmul2d_descriptor(SM, SN, BK, false, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    matmul2d<desc_bk, execution_simdgroups<4>> op_bk;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    bool use_bk = interior && ((int)K >= BK);

    if (use_bk) {
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        int k = 0;
        for (; k + BK <= (int)K; k += BK) {
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                             array<int, 2>{1, (int)N});
            op_bk.run(tA, tB, tC);
        }
        if (k < (int)K) {
            int k_rem = (int)K - k;
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{k_rem, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, k_rem},
                             array<int, 2>{1, (int)N});
            constexpr auto desc_tail =
                matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                                    matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc_tail, execution_simdgroups<4>> op_tail;
            op_tail.run(tA, tB, tC);
        }
    } else if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

kernel void matmul2d_tensorops_tn_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

kernel void matmul2d_tensorops_nt_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A_stored[K,M]^T @ B[K,N] (TN accumulate bf16→f32).
kernel void matmul2d_tensorops_tn_accum_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A[M,K] @ B_stored[N,K]^T (NT accumulate bf16→f32).
kernel void matmul2d_tensorops_nt_accum_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

kernel void matmul2d_tensorops_tn_splitk_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &k0 [[buffer(6)]],
    constant uint &k_tile [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto mmul_mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false, mmul_mode);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    uint k_len = min(k_tile, K - k0);
    auto mA = tensor(A + k0 * M, dextents<int, 2>{(int)M, (int)k_len}, array<int, 2>{1, (int)M});
    auto mB = tensor(B + k0 * N, dextents<int, 2>{(int)N, (int)k_len}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}
