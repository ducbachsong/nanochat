// mma_tma_matmul_bf16.cuh — mma_matmul_bf16's products (mma.sync, the same bits) with the tiles brought by TMA instead of the threads.
//
//     tensor   type in memory   shape
//     left     bfloat16         {rows, sum_length}, turned {sum_length, rows}
//     right    bfloat16         {columns, sum_length}, turned {sum_length, columns}
//     bias     bfloat16         {columns}, or none
//     output   bfloat16         {rows, columns}
//     sums     float            in registers
//     a part   bfloat16         one step's left tile {BLOCK_ROWS, BLOCK_K} and right tile {BLOCK_COLUMNS, BLOCK_K} (or turned), as TMA lays them, in shared memory
#ifndef MMA_TMA_MATMUL_BF16_CUH
#define MMA_TMA_MATMUL_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include "mma_matmul_bf16.cuh"       // MmaMatmulBf16Setting, the mma and its packing, the shape checks: the same math

// The defaults: mma_matmul_bf16's square, step and warps (MmaMatmulBf16Setting<>), and these; swept by tune_mma_tma_matmul_bf16.
#define MMA_TMA_MATMUL_BF16_SWIZZLE true               // the tiles in TMA's swizzle (32, 64 or 128 bytes, a tile row's), or as they sit
#define MMA_TMA_MATMUL_BF16_PARTS 2                    // parts of shared memory, a step's tiles each: 1 to MMA_TMA_MATMUL_BF16_MAX_PARTS
#define MMA_TMA_MATMUL_BF16_MAX_PARTS 8                // the most parts
#define MMA_TMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP   // the block order
#define MMA_TMA_MATMUL_BF16_BLOCKS 0                   // blocks a launch: 0, one a square; more: each walks squares b, b + blocks, ...
#define MMA_TMA_MATMUL_BF16_SWIZZLED_BOX_LINES (HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / (int)sizeof(nv_bfloat16))   // a turned tile's box, swizzled: 64 lines, one 128-byte swizzle row
#define MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM   // every part and box on 1,024 bytes: the swizzle's pattern from its start
#define MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES (HARDWARE_MAX_SHARED_BYTES_PER_BLOCK - MMA_TMA_MATMUL_BF16_MAX_PARTS * (int)sizeof(uint64_t))   // the dynamic shared memory a block can ask for: the rest is the barriers

/// One operand's tile as TMA lays it (LINES: BLOCK_ROWS for left, BLOCK_COLUMNS for right): its boxes, rows and swizzle.
template <typename SETTING, bool SWIZZLE, int LINES, bool TURNED>
struct MmaTmaMatmulBf16Tile {
    // Not TURNED: the matrix sits {lines, sum_length}: one box of LINES rows of BLOCK_K values, tile[line][k].
    // TURNED: it sits {sum_length, lines}: boxes of BOX_LINES lines, each BLOCK_K rows of them, tile[box][k][line].
    // A swizzled row is at most 128 bytes, so a turned tile of more lines is cut into boxes of 64.
    static constexpr int BOX_LINES = TURNED ? (SWIZZLE ? MMA_TMA_MATMUL_BF16_SWIZZLED_BOX_LINES : LINES) : LINES;
    static constexpr int BOXES = LINES / BOX_LINES;
    static constexpr int BOX_INNER = TURNED ? BOX_LINES : SETTING::BLOCK_K;          // values a box row
    static constexpr int BOX_OUTER = TURNED ? SETTING::BLOCK_K : LINES;              // rows a box
    static constexpr int ROW_BYTES = BOX_INNER * (int)sizeof(nv_bfloat16);
    static constexpr int BOX_BYTES = BOX_OUTER * ROW_BYTES;
    static constexpr int BYTES = BOXES * BOX_BYTES;                                  // the tile: LINES x BLOCK_K values
    static constexpr int SWIZZLE_BYTES = SWIZZLE ? ROW_BYTES : 0;                    // TMA's swizzle: the row's 32, 64 or 128 bytes, or none
    static constexpr int SWIZZLE_BITS = SWIZZLE_BYTES == 32 ? 1 : SWIZZLE_BYTES == 64 ? 2 : SWIZZLE_BYTES == 128 ? 3 : 0;   // 16-byte cells XORed
    static_assert(LINES % BOX_LINES == 0 && BOX_BYTES % MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES == 0, "the boxes split the tile, each on a swizzle atom");
    static_assert(!SWIZZLE || SWIZZLE_BITS > 0, "a swizzled row is 32, 64 or 128 bytes");

    /// Gives a value's byte offset in the tile, as TMA laid it.
    ///
    ///   line_val, k_val   the value: its line (a row of left, a column of output for right) and its k, in the tile
    __device__ static __forceinline__ int offset(int line_val, int k_val) {
        // The swizzle XORs a 16-byte cell's number (bits 4 up) with the address's bits 7 up, as many as SWIZZLE_BITS:
        // 2 neighbouring values stay neighbours, a cell's 8 move together.
        int box_val = TURNED ? line_val / BOX_LINES : 0;
        int row_val = TURNED ? k_val : line_val;
        int column_val = TURNED ? line_val % BOX_LINES : k_val;
        int offset_val = box_val * BOX_BYTES + row_val * ROW_BYTES + column_val * (int)sizeof(nv_bfloat16);
        if constexpr (SWIZZLE_BITS > 0)
            offset_val ^= ((offset_val >> 7) & ((1 << SWIZZLE_BITS) - 1)) << 4;
        return offset_val;
    }
};

/// Gives the shared memory one part of SETTING takes: a step's left tile and right tile.
template <typename SETTING>
constexpr size_t mma_tma_matmul_bf16_part_bytes() {
    return (size_t)(SETTING::BLOCK_ROWS + SETTING::BLOCK_COLUMNS) * SETTING::BLOCK_K * sizeof(nv_bfloat16);
}

/// Gives the dynamic shared memory a block of SETTING asks for with parts_val parts: the parts, and room to start them on an atom.
template <typename SETTING>
constexpr size_t mma_tma_matmul_bf16_shared_bytes(int parts_val) {
    return (size_t)parts_val * mma_tma_matmul_bf16_part_bytes<SETTING>() + MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES;
}

// ── the product ───────────────────────────────────────────────────────────────
void mma_tma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                         const Tensor& bias_tensor = Tensor(), bool add_onto_val = false, bool left_turned_val = false,
                         bool right_turned_val = false);
template <typename SETTING, bool SWIZZLE>
void mma_tma_matmul_bf16_at(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor, const Tensor& bias_tensor,
                            bool add_onto_val, bool left_turned_val, bool right_turned_val, int block_rows_per_group_val, int parts_val,
                            int blocks_val);
template <typename SETTING, bool SWIZZLE, bool LEFT_TURNED, bool RIGHT_TURNED>
cudaError_t mma_tma_matmul_bf16_launch(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr,
                                       const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                                       bool add_onto_val, int block_rows_per_group_val, int parts_val, int blocks_val);

// ── for it ────────────────────────────────────────────────────────────────────
CUresult mma_tma_matmul_bf16_tensor_map(CUtensorMap* map_ptr, const nv_bfloat16* matrix_ptr, size_t outer_val, size_t inner_val,
                                        int box_inner_val, int box_outer_val, int swizzle_bytes_val);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

template <typename SETTING, bool SWIZZLE, bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(SETTING::THREADS)
mma_tma_matmul_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                           nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                           bool add_onto_val, int block_rows_per_group_val, int parts_val);
template <typename TILE, bool TURNED>
__device__ __forceinline__ uint32_t mma_tma_matmul_bf16_read_pair(const uint8_t* tile_ptr, int line_val, int k_val);
__device__ __forceinline__ uint32_t mma_tma_matmul_bf16_shared_address(const void* shared_ptr);
__device__ __forceinline__ void mma_tma_matmul_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val);
__device__ __forceinline__ void mma_tma_matmul_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val);
__device__ __forceinline__ void mma_tma_matmul_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val);
__device__ __forceinline__ void mma_tma_matmul_bf16_copy_box(void* shared_ptr, const CUtensorMap* map_ptr, int inner_val, int outer_val,
                                                             uint64_t* barrier_ptr);

/// Writes output = left · rightᵀ (+ bias) (+ output) at the defaults.
///
///   output_tensor ... right_turned_val   as in mma_matmul_bf16
inline void mma_tma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val) {
    // MmaMatmulBf16Setting<>, MMA_TMA_MATMUL_BF16_SWIZZLE, its block order, parts and blocks. Needs an H100 (TMA).
    mma_tma_matmul_bf16_at<MmaMatmulBf16Setting<>, MMA_TMA_MATMUL_BF16_SWIZZLE>(
        output_tensor, left_tensor, right_tensor, bias_tensor, add_onto_val, left_turned_val, right_turned_val,
        MMA_TMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP, MMA_TMA_MATMUL_BF16_PARTS, MMA_TMA_MATMUL_BF16_BLOCKS);
}

/// Writes output = left · rightᵀ (+ bias) (+ output) at the setting given.
///
///   output_tensor ... right_turned_val   as in mma_matmul_bf16
///   block_rows_per_group_val             the block order: rows of squares a group of blocks runs down (at least 1)
///   parts_val                            parts of shared memory, a step's tiles each: 1 to MMA_TMA_MATMUL_BF16_MAX_PARTS
///   blocks_val                           blocks a launch: 0 one a square; more, each walks squares b, b + blocks, ...
template <typename SETTING, bool SWIZZLE>
inline void mma_tma_matmul_bf16_at(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                   const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val,
                                   int block_rows_per_group_val, int parts_val, int blocks_val) {
    // Checks (mma_matmul_bf16's shapes), then launches mma_tma_matmul_bf16_kernel made for SETTING, SWIZZLE and the
    // two turned flags. The same bits as mma_matmul_bf16 at any setting: the same mma on the same values in the same
    // order of k; only who brings the tiles changes.
    static_assert(SETTING::FITS, "the warps split the square into whole mma tiles, a step is whole mma steps, at most 1024 threads");
    // 1. check the tensors and get the sizes; any of them wrong: stop with an error
    size_t rows_val = 0, columns_val = 0, sum_length_val = 0;
    mma_matmul_bf16_check_shapes(output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val,
                                 &rows_val, &columns_val, &sum_length_val);
    if (block_rows_per_group_val < 1 || parts_val < 1 || parts_val > MMA_TMA_MATMUL_BF16_MAX_PARTS || blocks_val < 0 ||
        mma_tma_matmul_bf16_shared_bytes<SETTING>(parts_val) > (size_t)MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES) {
        fprintf(stderr, "error: mma_tma_matmul_bf16 needs block_rows_per_group 1 or more, parts 1 to %d within %d B of shared memory "
                        "(%zu B a part), blocks 0 or more; got %d, %d, %d\n", MMA_TMA_MATMUL_BF16_MAX_PARTS, MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES,
                mma_tma_matmul_bf16_part_bytes<SETTING>(), block_rows_per_group_val, parts_val, blocks_val);
        exit(EXIT_FAILURE);
    }
    // 2. launch, for the turned flags
    nv_bfloat16* output_ptr = output_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* left_ptr = left_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* right_ptr = right_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* bias_ptr = bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr;
    cudaError_t error_val;
    if (!left_turned_val && !right_turned_val)
        error_val = mma_tma_matmul_bf16_launch<SETTING, SWIZZLE, false, false>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                               sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    else if (!left_turned_val && right_turned_val)
        error_val = mma_tma_matmul_bf16_launch<SETTING, SWIZZLE, false, true>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                              sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    else if (left_turned_val && !right_turned_val)
        error_val = mma_tma_matmul_bf16_launch<SETTING, SWIZZLE, true, false>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                              sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    else
        error_val = mma_tma_matmul_bf16_launch<SETTING, SWIZZLE, true, true>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                             sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    cudaCheck(error_val);
}

/// Makes the two tensor maps and launches one mma_tma_matmul_bf16_kernel, and gives the error (cudaSuccess if it started).
///
///   output_ptr, left_ptr, right_ptr, bias_ptr    as in mma_matmul_bf16; bias nullptr for none
///   rows_val, columns_val, sum_length_val        the product's sizes (checked by the caller)
///   add_onto_val                                 as in mma_matmul_bf16
///   block_rows_per_group_val, parts_val, blocks_val   as in mma_tma_matmul_bf16_at (checked by the caller)
template <typename SETTING, bool SWIZZLE, bool LEFT_TURNED, bool RIGHT_TURNED>
inline cudaError_t mma_tma_matmul_bf16_launch(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr,
                                              const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                                              bool add_onto_val, int block_rows_per_group_val, int parts_val, int blocks_val) {
    // TMA's description of left and right, a box their tile (or a turned tile's boxes of 64 lines when swizzled); past
    // the edges TMA brings 0s. The parts in dynamic shared memory, allowed up to MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES the
    // first time. A tune skips the settings that do not fit rather than stopping: every error is given back.
    using LeftTile = MmaTmaMatmulBf16Tile<SETTING, SWIZZLE, SETTING::BLOCK_ROWS, LEFT_TURNED>;
    using RightTile = MmaTmaMatmulBf16Tile<SETTING, SWIZZLE, SETTING::BLOCK_COLUMNS, RIGHT_TURNED>;
    // 1. the maps: {lines, sum_length} with boxes {BLOCK_K, lines}, or turned {sum_length, lines} with boxes {box lines, BLOCK_K}
    CUtensorMap left_map_val, right_map_val;
    CUresult left_result_val = LEFT_TURNED ? mma_tma_matmul_bf16_tensor_map(&left_map_val, left_ptr, sum_length_val, rows_val,
                                                                            LeftTile::BOX_INNER, LeftTile::BOX_OUTER, LeftTile::SWIZZLE_BYTES)
                                           : mma_tma_matmul_bf16_tensor_map(&left_map_val, left_ptr, rows_val, sum_length_val,
                                                                            LeftTile::BOX_INNER, LeftTile::BOX_OUTER, LeftTile::SWIZZLE_BYTES);
    CUresult right_result_val = RIGHT_TURNED ? mma_tma_matmul_bf16_tensor_map(&right_map_val, right_ptr, sum_length_val, columns_val,
                                                                              RightTile::BOX_INNER, RightTile::BOX_OUTER, RightTile::SWIZZLE_BYTES)
                                             : mma_tma_matmul_bf16_tensor_map(&right_map_val, right_ptr, columns_val, sum_length_val,
                                                                              RightTile::BOX_INNER, RightTile::BOX_OUTER, RightTile::SWIZZLE_BYTES);
    if (left_result_val != CUDA_SUCCESS || right_result_val != CUDA_SUCCESS) return cudaErrorInvalidValue;
    // 2. the shared memory allowed, the first time
    auto kernel_ptr = mma_tma_matmul_bf16_kernel<SETTING, SWIZZLE, LEFT_TURNED, RIGHT_TURNED>;
    static bool shared_set_val = false;
    if (!shared_set_val) {
        cudaError_t error_val = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES);
        if (error_val != cudaSuccess) return error_val;
        shared_set_val = true;
    }
    // 3. launch: one block a square, or blocks_val, never more than the squares
    size_t squares_val = CEIL_DIV(rows_val, (size_t)SETTING::BLOCK_ROWS) * CEIL_DIV(columns_val, (size_t)SETTING::BLOCK_COLUMNS);
    unsigned launched_val = (unsigned)(blocks_val > 0 && (size_t)blocks_val < squares_val ? (size_t)blocks_val : squares_val);
    kernel_ptr<<<launched_val, SETTING::THREADS, mma_tma_matmul_bf16_shared_bytes<SETTING>(parts_val)>>>(
        left_map_val, right_map_val, output_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val, block_rows_per_group_val,
        parts_val);
    return cudaGetLastError();
}

/// Works out this block's squares of output: squares b, b + blocks, ...
///
///   left_map_val, right_map_val                  TMA's descriptions of left and right
///   output_ptr, bias_ptr                         as in mma_matmul_bf16; bias nullptr for none
///   rows_val, columns_val, sum_length_val        the product's sizes
///   add_onto_val                                 as in mma_matmul_bf16
///   block_rows_per_group_val                     the block order
///   parts_val                                    parts of shared memory, a step's tiles each
template <typename SETTING, bool SWIZZLE, bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(SETTING::THREADS)
mma_tma_matmul_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                           nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                           bool add_onto_val, int block_rows_per_group_val, int parts_val) {
    // For each square, step by step: thread 0 asks TMA for a step's tile of left and of right into a part of shared
    // memory, counted at the part's mbarrier; every thread waits for it, then each warp adds its part of their product
    // into its sums on the tensor cores (mma_matmul_bf16's math). When every warp is done with a part (each thread's
    // reads fenced against TMA, then a block-wide sync), thread 0 asks for the step parts on into it.
    //
    //     1 part:   ask for a step, wait, multiply, then the next step
    //     n parts:  n steps asked for ahead; a step done frees its part for the step n on
    //
    // The parts are used in turn across the squares: a block's g-th step overall is in part g % parts, its barrier's
    // phase (g / parts) % 2.
    using LeftTile = MmaTmaMatmulBf16Tile<SETTING, SWIZZLE, SETTING::BLOCK_ROWS, LEFT_TURNED>;
    using RightTile = MmaTmaMatmulBf16Tile<SETTING, SWIZZLE, SETTING::BLOCK_COLUMNS, RIGHT_TURNED>;
    constexpr int PART_BYTES = LeftTile::BYTES + RightTile::BYTES;
    extern __shared__ __align__(16) uint8_t mma_tma_matmul_bf16_shared_list[];
    __shared__ uint64_t full_barriers_list[MMA_TMA_MATMUL_BF16_MAX_PARTS];   // part p's bytes are in
    // the parts start on a swizzle atom: the swizzle is worked out from the address's bits 7 to 9
    uint32_t shared_address_val = mma_tma_matmul_bf16_shared_address(mma_tma_matmul_bf16_shared_list);
    uint8_t* parts_ptr = mma_tma_matmul_bf16_shared_list +
                         (MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES - shared_address_val % MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES) % MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_first_row_index_val = (warp_index_val / SETTING::WARPS_ACROSS) * SETTING::WARP_ROWS;
    int warp_first_column_index_val = (warp_index_val % SETTING::WARPS_ACROSS) * SETTING::WARP_COLUMNS;
    size_t squares_val = CEIL_DIV(rows_val, (size_t)SETTING::BLOCK_ROWS) * CEIL_DIV(columns_val, (size_t)SETTING::BLOCK_COLUMNS);
    int total_steps_val = (int)CEIL_DIV(sum_length_val, (size_t)SETTING::BLOCK_K);

    // 1. the barriers, by one thread, visible to TMA
    if (threadIdx.x == 0) {
        for (int part_index_val = 0; part_index_val < parts_val; part_index_val++)
            mma_tma_matmul_bf16_barrier_init(&full_barriers_list[part_index_val], 1);   // thread 0, and the bytes
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");                 // the barriers visible to TMA
    }
    __syncthreads();

    // thread 0 asks TMA for a step's two tiles into a part: first tells the part's barrier the bytes coming
    auto ask_step_val = [&](int step_index_val, int part_index_val, size_t first_row_index_val, size_t first_column_index_val) {
        uint8_t* part_ptr = parts_ptr + (size_t)part_index_val * PART_BYTES;
        uint64_t* barrier_ptr = &full_barriers_list[part_index_val];
        int first_k_index_val = step_index_val * SETTING::BLOCK_K;
        mma_tma_matmul_bf16_barrier_expect(barrier_ptr, (uint32_t)PART_BYTES);
#pragma unroll
        for (int box_index_val = 0; box_index_val < LeftTile::BOXES; box_index_val++) {
            int line_val = (int)first_row_index_val + box_index_val * LeftTile::BOX_LINES;
            mma_tma_matmul_bf16_copy_box(part_ptr + box_index_val * LeftTile::BOX_BYTES, &left_map_val, LEFT_TURNED ? line_val : first_k_index_val,
                                         LEFT_TURNED ? first_k_index_val : line_val, barrier_ptr);
        }
#pragma unroll
        for (int box_index_val = 0; box_index_val < RightTile::BOXES; box_index_val++) {
            int line_val = (int)first_column_index_val + box_index_val * RightTile::BOX_LINES;
            mma_tma_matmul_bf16_copy_box(part_ptr + LeftTile::BYTES + box_index_val * RightTile::BOX_BYTES, &right_map_val,
                                         RIGHT_TURNED ? line_val : first_k_index_val, RIGHT_TURNED ? first_k_index_val : line_val, barrier_ptr);
        }
    };

    size_t steps_done_val = 0;   // this block's steps so far, over every square: the next step's part and phase
    for (size_t square_index_val = blockIdx.x; square_index_val < squares_val; square_index_val += gridDim.x) {
        // 2. the square (mma_matmul_bf16's order), and its first parts steps asked for
        size_t first_row_index_val = 0, first_column_index_val = 0;
        mma_matmul_bf16_square<SETTING>(square_index_val, rows_val, columns_val, block_rows_per_group_val, &first_row_index_val,
                                        &first_column_index_val);
        float sums_list[SETTING::M_TILES][SETTING::N_TILES][HARDWARE_MMA_C_VALUES_PER_THREAD_BF16] = {};
        if (threadIdx.x == 0)
            for (int step_index_val = 0; step_index_val < parts_val && step_index_val < total_steps_val; step_index_val++)
                ask_step_val(step_index_val, (int)((steps_done_val + step_index_val) % parts_val), first_row_index_val, first_column_index_val);

        // 3. walk the sum: wait for the step's part, multiply, free the part for the step parts on
        for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
            size_t overall_step_val = steps_done_val + step_index_val;
            int part_index_val = (int)(overall_step_val % parts_val);
            mma_tma_matmul_bf16_barrier_wait(&full_barriers_list[part_index_val], (uint32_t)((overall_step_val / parts_val) % 2));
            const uint8_t* left_tile_ptr = parts_ptr + (size_t)part_index_val * PART_BYTES;
            const uint8_t* right_tile_ptr = left_tile_ptr + LeftTile::BYTES;
            // the tensor cores: for each 16 values of k, this warp's A and B registers (mma_matmul_bf16's), then every A x B
#pragma unroll
            for (int k_step_index_val = 0; k_step_index_val < SETTING::K_STEPS; k_step_index_val++) {
                int first_k_index_val = k_step_index_val * HARDWARE_MMA_K_BF16;
                int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
                int pair_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
                const int half_k_val = HARDWARE_MMA_K_BF16 / 2;
                uint32_t a_list[SETTING::M_TILES][HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16];
                uint32_t b_list[SETTING::N_TILES][HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16];
#pragma unroll
                for (int m_tile_index_val = 0; m_tile_index_val < SETTING::M_TILES; m_tile_index_val++) {
                    // A's (group, pair ..), (group + 8, pair ..), (group, pair + 8 ..), (group + 8, pair + 8 ..)
                    int row_val = warp_first_row_index_val + m_tile_index_val * HARDWARE_MMA_M + group_index_val;
                    int k_val = first_k_index_val + pair_index_val;
                    a_list[m_tile_index_val][0] = mma_tma_matmul_bf16_read_pair<LeftTile, LEFT_TURNED>(left_tile_ptr, row_val, k_val);
                    a_list[m_tile_index_val][1] = mma_tma_matmul_bf16_read_pair<LeftTile, LEFT_TURNED>(left_tile_ptr, row_val + HARDWARE_MMA_ROWS_PER_WARP, k_val);
                    a_list[m_tile_index_val][2] = mma_tma_matmul_bf16_read_pair<LeftTile, LEFT_TURNED>(left_tile_ptr, row_val, k_val + half_k_val);
                    a_list[m_tile_index_val][3] = mma_tma_matmul_bf16_read_pair<LeftTile, LEFT_TURNED>(left_tile_ptr, row_val + HARDWARE_MMA_ROWS_PER_WARP,
                                                                                                        k_val + half_k_val);
                }
#pragma unroll
                for (int n_tile_index_val = 0; n_tile_index_val < SETTING::N_TILES; n_tile_index_val++) {
                    // B's (pair .., column group) and (pair + 8 .., column group), B[k][n] = right[n][k]
                    int column_val = warp_first_column_index_val + n_tile_index_val * HARDWARE_MMA_N + group_index_val;
                    int k_val = first_k_index_val + pair_index_val;
                    b_list[n_tile_index_val][0] = mma_tma_matmul_bf16_read_pair<RightTile, RIGHT_TURNED>(right_tile_ptr, column_val, k_val);
                    b_list[n_tile_index_val][1] = mma_tma_matmul_bf16_read_pair<RightTile, RIGHT_TURNED>(right_tile_ptr, column_val, k_val + half_k_val);
                }
#pragma unroll
                for (int m_tile_index_val = 0; m_tile_index_val < SETTING::M_TILES; m_tile_index_val++)
#pragma unroll
                    for (int n_tile_index_val = 0; n_tile_index_val < SETTING::N_TILES; n_tile_index_val++)
                        mma_matmul_bf16_multiply_add(sums_list[m_tile_index_val][n_tile_index_val], a_list[m_tile_index_val],
                                                     b_list[n_tile_index_val]);
            }
            // This thread's reads of the part done before TMA (the async proxy) writes it: the compiler may leave the
            // step's last reads in flight past __syncthreads, their multiply-adds after it, and TMA's write of the
            // step parts on would land first (wrong bits). The fence waits for them (MEMBAR.ALL.CTA, FENCE.VIEW.ASYNC.S).
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();                    // every warp done with the part
            if (threadIdx.x == 0 && step_index_val + parts_val < total_steps_val)
                ask_step_val(step_index_val + parts_val, part_index_val, first_row_index_val, first_column_index_val);
        }
        steps_done_val += total_steps_val;

        // 4. output = sums (+ bias) (+ output), two neighbouring columns at a time; nothing past the edges (mma_matmul_bf16's)
        int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
        int pair_column_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
#pragma unroll
        for (int m_tile_index_val = 0; m_tile_index_val < SETTING::M_TILES; m_tile_index_val++) {
#pragma unroll
            for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_MMA_ROWS_PER_THREAD; thread_row_index_val++) {
                size_t row_index_val = first_row_index_val + warp_first_row_index_val + m_tile_index_val * HARDWARE_MMA_M +
                                       thread_row_index_val * HARDWARE_MMA_ROWS_PER_WARP + group_index_val;
                if (row_index_val >= rows_val) continue;
#pragma unroll
                for (int n_tile_index_val = 0; n_tile_index_val < SETTING::N_TILES; n_tile_index_val++) {
                    size_t column_index_val = first_column_index_val + warp_first_column_index_val + n_tile_index_val * HARDWARE_MMA_N +
                                              pair_column_index_val;
                    if (column_index_val >= columns_val) continue;   // columns is even, so column + 1 is in too
                    int c_index_val = HARDWARE_MMA_COLUMNS_PER_THREAD * thread_row_index_val;
                    float2 bias_val = bias_ptr != nullptr ? __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val))
                                                          : make_float2(0.0f, 0.0f);
                    float2 result_val = make_float2(sums_list[m_tile_index_val][n_tile_index_val][c_index_val] + bias_val.x,
                                                    sums_list[m_tile_index_val][n_tile_index_val][c_index_val + 1] + bias_val.y);
                    nv_bfloat162* output_pair_ptr = reinterpret_cast<nv_bfloat162*>(output_ptr + row_index_val * columns_val + column_index_val);
                    if (add_onto_val) {
                        float2 old_val = __bfloat1622float2(*output_pair_ptr);
                        result_val = make_float2(result_val.x + old_val.x, result_val.y + old_val.y);
                    }
                    *output_pair_ptr = __floats2bfloat162_rn(result_val.x, result_val.y);
                }
            }
        }
    }
}

/// Gives two neighbouring values of k of one line, as one register, the first in the low half.
///
///   tile_ptr          the tile, as TMA laid it
///   line_val, k_val   the first value: its line and its k (even), in the tile
template <typename TILE, bool TURNED>
__device__ __forceinline__ uint32_t mma_tma_matmul_bf16_read_pair(const uint8_t* tile_ptr, int line_val, int k_val) {
    // Not TURNED: tile[line][k]: the 2 sit side by side in one 16-byte cell (the swizzle moves whole cells), one
    // 4-byte read. TURNED: tile[k][line]: they sit a row apart, two reads packed into one register.
    if constexpr (!TURNED) {
        return *reinterpret_cast<const uint32_t*>(tile_ptr + TILE::offset(line_val, k_val));
    } else {
        nv_bfloat16 low_val = *reinterpret_cast<const nv_bfloat16*>(tile_ptr + TILE::offset(line_val, k_val));
        nv_bfloat16 high_val = *reinterpret_cast<const nv_bfloat16*>(tile_ptr + TILE::offset(line_val, k_val + 1));
        return mma_matmul_bf16_pack(low_val, high_val);
    }
}

/// Gives a pointer into shared memory as the 32-bit address the PTX instructions take.
///
///   shared_ptr   a pointer into the block's shared memory
__device__ __forceinline__ uint32_t mma_tma_matmul_bf16_shared_address(const void* shared_ptr) {
    return (uint32_t)__cvta_generic_to_shared(shared_ptr);
}

/// Makes an mbarrier that completes a phase after arrivals_val arrivals (and the bytes it is told to expect).
///
///   barrier_ptr    the barrier, 8 bytes of shared memory
///   arrivals_val   arrivals a phase
__device__ __forceinline__ void mma_tma_matmul_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(mma_tma_matmul_bf16_shared_address(barrier_ptr)), "r"(arrivals_val) : "memory");
}

/// Arrives once at a barrier and tells it bytes_val bytes of TMA copies are coming.
///
///   barrier_ptr   the barrier
///   bytes_val     the bytes the copies asked for next bring
__device__ __forceinline__ void mma_tma_matmul_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val) {
    // Its phase completes when they are in (a box past the edges counts whole: TMA brings its 0s too).
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(mma_tma_matmul_bf16_shared_address(barrier_ptr)), "r"(bytes_val) : "memory");
}

/// Waits until a barrier's phase of this parity is complete.
///
///   barrier_ptr   the barrier
///   parity_val    the phase waited for, 0 or 1
__device__ __forceinline__ void mma_tma_matmul_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val) {
    uint32_t barrier_address_val = mma_tma_matmul_bf16_shared_address(barrier_ptr);
    uint32_t done_val;
    do {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, p;\n"
                     "}\n"
                     : "=r"(done_val) : "r"(barrier_address_val), "r"(parity_val) : "memory");
    } while (done_val == 0);
}

/// Asks TMA for one box of a matrix into this block's shared memory, its bytes counted at a barrier.
///
///   shared_ptr             where the box goes, in shared memory
///   map_ptr                TMA's description of the matrix
///   inner_val, outer_val   the box's corner: its first value along a row, and its first row
///   barrier_ptr            the barrier that expects the bytes
__device__ __forceinline__ void mma_tma_matmul_bf16_copy_box(void* shared_ptr, const CUtensorMap* map_ptr, int inner_val, int outer_val,
                                                             uint64_t* barrier_ptr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(mma_tma_matmul_bf16_shared_address(shared_ptr)), "l"(map_ptr), "r"(inner_val), "r"(outer_val),
                   "r"(mma_tma_matmul_bf16_shared_address(barrier_ptr))
                 : "memory");
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Makes TMA's description of a bfloat16 matrix {outer_val rows, inner_val values a row}, in boxes of box_outer_val x box_inner_val, and gives the driver's result.
///
///   map_ptr                            written: the description, when the result is CUDA_SUCCESS
///   matrix_ptr                         the matrix in GPU memory, on 16 bytes
///   outer_val, inner_val               its shape
///   box_inner_val, box_outer_val       a box
///   swizzle_bytes_val                  32, 64 or 128: TMA's swizzle of that row size; 0: none
inline CUresult mma_tma_matmul_bf16_tensor_map(CUtensorMap* map_ptr, const nv_bfloat16* matrix_ptr, size_t outer_val, size_t inner_val,
                                               int box_inner_val, int box_outer_val, int swizzle_bytes_val) {
    // The driver's cuTensorMapEncodeTiled, found at run time; past the edges: 0s.
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function_ptr, 12000, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: mma_tma_matmul_bf16 could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    CUtensorMapSwizzle swizzle_val = swizzle_bytes_val == 32    ? CU_TENSOR_MAP_SWIZZLE_32B
                                     : swizzle_bytes_val == 64  ? CU_TENSOR_MAP_SWIZZLE_64B
                                     : swizzle_bytes_val == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                                                                : CU_TENSOR_MAP_SWIZZLE_NONE;
    cuuint64_t sizes_list[2] = {inner_val, outer_val};
    cuuint64_t row_bytes_list[1] = {inner_val * sizeof(nv_bfloat16)};
    cuuint32_t box_list[2] = {(cuuint32_t)box_inner_val, (cuuint32_t)box_outer_val};
    cuuint32_t element_steps_list[2] = {1, 1};
    return encode_ptr(map_ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<nv_bfloat16*>(matrix_ptr), sizes_list, row_bytes_list,
                      box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_val, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

#endif // MMA_TMA_MATMUL_BF16_CUH
