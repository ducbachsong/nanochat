// mma_matmul_bf16.cuh — matrix products in bfloat16 on the tensor cores by mma.sync (sm_80 or later), the tiles copied by the threads (cp.async, no TMA) into parts of shared memory.
//
//     tensor   type in memory   shape
//     left     bfloat16         {rows, sum_length}, turned {sum_length, rows}
//     right    bfloat16         {columns, sum_length}, turned {sum_length, columns}
//     bias     bfloat16         {columns}, or none
//     output   bfloat16         {rows, columns}
//     sums     float            in registers
//     a part   bfloat16         one step's left tile {BLOCK_ROWS, BLOCK_K} and right tile {BLOCK_COLUMNS, BLOCK_K} (or turned), a padded row each, in shared memory
#ifndef MMA_MATMUL_BF16_CUH
#define MMA_MATMUL_BF16_CUH

#include "../cuda_common.h"             // cudaCheck, HARDWARE_MMA_*, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"

// The defaults: our choices, swept by tune_mma_matmul_bf16. The hardware's numbers (HARDWARE_MMA_*) are in cuda_common.h.
#define MMA_MATMUL_BF16_BLOCK_ROWS 128                 // each block makes 128 rows x 128 columns of output
#define MMA_MATMUL_BF16_BLOCK_COLUMNS 128
#define MMA_MATMUL_BF16_BLOCK_K 32                     // values of k a step copies: 2 mma steps of 16
#define MMA_MATMUL_BF16_WARPS_DOWN 2                   // the block's warps: 2 down x 4 across
#define MMA_MATMUL_BF16_WARPS_ACROSS 4
#define MMA_MATMUL_BF16_THREADS_PER_BLOCK (MMA_MATMUL_BF16_WARPS_DOWN * MMA_MATMUL_BF16_WARPS_ACROSS * HARDWARE_THREADS_PER_WARP)   // the defaults' block: 256
#define MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP 8         // the block order: rows of squares a group of blocks runs down, as fma_matmul_bf16.cuh's (L2)
#define MMA_MATMUL_BF16_PARTS 2                        // parts of shared memory, a step's tiles each: 1 to MMA_MATMUL_BF16_MAX_PARTS
#define MMA_MATMUL_BF16_MAX_PARTS 8                    // the most parts (cp.async.wait_group takes the steps left in flight as a constant)
#define MMA_MATMUL_BF16_BLOCKS 0                       // blocks a launch: 0, one a square; more: each walks squares b, b + blocks, ...
#define MMA_MATMUL_BF16_MAX_SHARED_BYTES HARDWARE_MAX_SHARED_BYTES_PER_BLOCK   // the parts' shared memory, at most
// The copies: 16-byte copies; a tile row has one spare copy's room, so the rows' reads fall in different banks.
#define MMA_MATMUL_BF16_LOAD_VALUES ((int)(sizeof(uint4) / sizeof(nv_bfloat16)))    // bfloat16s a copy: 8
#define MMA_MATMUL_BF16_LOAD_BYTES ((int)sizeof(uint4))                             // bytes a copy: 16

/// A product's setting, as one type a caller names once: the square, the step of k, the warps; and what follows from them.
template <int BLOCK_ROWS_ARG = MMA_MATMUL_BF16_BLOCK_ROWS, int BLOCK_COLUMNS_ARG = MMA_MATMUL_BF16_BLOCK_COLUMNS,
          int BLOCK_K_ARG = MMA_MATMUL_BF16_BLOCK_K, int WARPS_DOWN_ARG = MMA_MATMUL_BF16_WARPS_DOWN,
          int WARPS_ACROSS_ARG = MMA_MATMUL_BF16_WARPS_ACROSS>
struct MmaMatmulBf16Setting {
    static constexpr int BLOCK_ROWS = BLOCK_ROWS_ARG;                       // rows of output a block makes
    static constexpr int BLOCK_COLUMNS = BLOCK_COLUMNS_ARG;                 // columns of it
    static constexpr int BLOCK_K = BLOCK_K_ARG;                             // values of k a step copies
    static constexpr int WARPS_DOWN = WARPS_DOWN_ARG;                       // the block's warps, down ...
    static constexpr int WARPS_ACROSS = WARPS_ACROSS_ARG;                   // ... and across
    static constexpr int THREADS = WARPS_DOWN * WARPS_ACROSS * HARDWARE_THREADS_PER_WARP;
    static constexpr int WARP_ROWS = BLOCK_ROWS / WARPS_DOWN;               // rows of a warp's part of the square
    static constexpr int WARP_COLUMNS = BLOCK_COLUMNS / WARPS_ACROSS;       // columns of it
    static constexpr int M_TILES = WARP_ROWS / HARDWARE_MMA_M;              // mma tiles down a warp's part
    static constexpr int N_TILES = WARP_COLUMNS / HARDWARE_MMA_N;           // ... across it
    static constexpr int K_STEPS = BLOCK_K / HARDWARE_MMA_K_BF16;           // mma steps of k a step
    static constexpr int SUMS = M_TILES * N_TILES * HARDWARE_MMA_C_VALUES_PER_THREAD_BF16;   // float sums a thread keeps
    static constexpr int K_ROW_LENGTH = BLOCK_K + MMA_MATMUL_BF16_LOAD_VALUES;   // a tile row as it sits {line, k}, with its spare room
    // Built only when true: the warps split the square into whole mma tiles, a step is whole mma steps and whole
    // copies, a block is at most HARDWARE_MAX_THREADS_PER_BLOCK.
    static constexpr bool FITS = WARPS_DOWN >= 1 && WARPS_ACROSS >= 1 && BLOCK_ROWS % (WARPS_DOWN * HARDWARE_MMA_M) == 0 &&
                                 BLOCK_COLUMNS % (WARPS_ACROSS * HARDWARE_MMA_N) == 0 && BLOCK_K % HARDWARE_MMA_K_BF16 == 0 &&
                                 BLOCK_K % MMA_MATMUL_BF16_LOAD_VALUES == 0 && BLOCK_ROWS % MMA_MATMUL_BF16_LOAD_VALUES == 0 &&
                                 BLOCK_COLUMNS % MMA_MATMUL_BF16_LOAD_VALUES == 0 && THREADS <= HARDWARE_MAX_THREADS_PER_BLOCK;
};

/// One operand's tile of a setting (LINES: BLOCK_ROWS for left, BLOCK_COLUMNS for right): its room and its copies.
template <typename SETTING, int LINES>
struct MmaMatmulBf16Tile {
    static constexpr int LINE_ROW_LENGTH = LINES + MMA_MATMUL_BF16_LOAD_VALUES;   // a tile row turned {k, line}, with its spare room
    static constexpr int VALUES = LINES * SETTING::K_ROW_LENGTH > SETTING::BLOCK_K * LINE_ROW_LENGTH
                                      ? LINES * SETTING::K_ROW_LENGTH
                                      : SETTING::BLOCK_K * LINE_ROW_LENGTH;           // room for one tile either way
    static constexpr int PIECES = LINES * SETTING::BLOCK_K / MMA_MATMUL_BF16_LOAD_VALUES;   // 16-byte copies a tile
};

/// Gives the shared memory one part of SETTING takes: a step's left tile and right tile.
template <typename SETTING>
constexpr size_t mma_matmul_bf16_part_bytes() {
    return (size_t)(MmaMatmulBf16Tile<SETTING, SETTING::BLOCK_ROWS>::VALUES + MmaMatmulBf16Tile<SETTING, SETTING::BLOCK_COLUMNS>::VALUES) *
           sizeof(nv_bfloat16);
}

// ── the product ───────────────────────────────────────────────────────────────
void mma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                     const Tensor& bias_tensor = Tensor(), bool add_onto_val = false, bool left_turned_val = false,
                     bool right_turned_val = false);
template <typename SETTING>
void mma_matmul_bf16_at(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor, const Tensor& bias_tensor,
                        bool add_onto_val, bool left_turned_val, bool right_turned_val, int block_rows_per_group_val, int parts_val,
                        int blocks_val);
template <typename SETTING, bool LEFT_TURNED, bool RIGHT_TURNED>
cudaError_t mma_matmul_bf16_launch(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr,
                                   const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                                   bool add_onto_val, int block_rows_per_group_val, int parts_val, int blocks_val);

// ── for it ────────────────────────────────────────────────────────────────────
void mma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                  const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                  size_t* columns_ptr, size_t* sum_length_ptr);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

template <typename SETTING, bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(SETTING::THREADS)
mma_matmul_bf16_kernel(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr, const nv_bfloat16* bias_ptr,
                       size_t rows_val, size_t columns_val, size_t sum_length_val, bool add_onto_val, int block_rows_per_group_val,
                       int parts_val);
template <typename SETTING>
__device__ __forceinline__ void mma_matmul_bf16_square(size_t square_index_val, size_t rows_val, size_t columns_val,
                                                       int block_rows_per_group_val, size_t* first_row_index_ptr,
                                                       size_t* first_column_index_ptr);
template <typename SETTING, int LINES, bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_copy_tile(nv_bfloat16* tile_ptr, const nv_bfloat16* matrix_ptr, size_t lines_val,
                                                          size_t sum_length_val, size_t first_line_index_val, size_t step_first_k_index_val);
__device__ __forceinline__ void mma_matmul_bf16_copy_commit(void);
__device__ __forceinline__ void mma_matmul_bf16_copy_wait(int steps_in_flight_val);
template <typename SETTING, bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_a(uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_row_index_val, int first_k_index_val);
template <typename SETTING, bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_b(uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_column_index_val, int first_k_index_val);
__device__ __forceinline__ void mma_matmul_bf16_multiply_add(float (&c_list)[HARDWARE_MMA_C_VALUES_PER_THREAD_BF16],
                                                             const uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16],
                                                             const uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16]);
__device__ __forceinline__ uint32_t mma_matmul_bf16_pack(nv_bfloat16 low_val, nv_bfloat16 high_val);

/// Writes output = left · rightᵀ (+ bias) (+ output) at the defaults.
///
///   output_tensor      {..., columns}, rows x columns values: written over, or added onto
///   left_tensor        {..., sum_length} (rows = the other dims), or turned {..., rows} (sum_length = the other dims)
///   right_tensor       {..., sum_length} (columns = the other dims), or turned {..., columns}
///   bias_tensor        {columns}, or Tensor() for none
///   add_onto_val       true: added onto what output holds; false: written over
///   left_turned_val, right_turned_val   true: that tensor holds its matrix turned round
inline void mma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                            const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val) {
    // MmaMatmulBf16Setting<>, MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP, MMA_MATMUL_BF16_PARTS, MMA_MATMUL_BF16_BLOCKS. All
    // bfloat16; every tensor's last dim a multiple of MMA_MATMUL_BF16_LOAD_VALUES (8) and its start on 16 bytes.
    mma_matmul_bf16_at<MmaMatmulBf16Setting<>>(output_tensor, left_tensor, right_tensor, bias_tensor, add_onto_val, left_turned_val,
                                               right_turned_val, MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP, MMA_MATMUL_BF16_PARTS,
                                               MMA_MATMUL_BF16_BLOCKS);
}

/// Writes output = left · rightᵀ (+ bias) (+ output) at the setting given.
///
///   output_tensor ... right_turned_val   as in mma_matmul_bf16
///   block_rows_per_group_val             the block order: rows of squares a group of blocks runs down (at least 1)
///   parts_val                            parts of shared memory, a step's tiles each: 1 to MMA_MATMUL_BF16_MAX_PARTS
///   blocks_val                           blocks a launch: 0 one a square; more, each walks squares b, b + blocks, ...
template <typename SETTING>
inline void mma_matmul_bf16_at(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor, const Tensor& bias_tensor,
                               bool add_onto_val, bool left_turned_val, bool right_turned_val, int block_rows_per_group_val, int parts_val,
                               int blocks_val) {
    // Checks, then launches mma_matmul_bf16_kernel made for SETTING and the two turned flags. Any setting gives the
    // same bits: each output value's sum runs over k in the same order, 16 values a mma, whatever the square, the
    // step, the warps, the parts, the blocks or their order.
    static_assert(SETTING::FITS, "the warps split the square into whole mma tiles, a step is whole mma steps and copies, at most 1024 threads");
    // 1. check the tensors and get the sizes; any of them wrong: stop with an error
    size_t rows_val = 0, columns_val = 0, sum_length_val = 0;
    mma_matmul_bf16_check_shapes(output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val,
                                 &rows_val, &columns_val, &sum_length_val);
    if (block_rows_per_group_val < 1 || parts_val < 1 || parts_val > MMA_MATMUL_BF16_MAX_PARTS || blocks_val < 0 ||
        (size_t)parts_val * mma_matmul_bf16_part_bytes<SETTING>() > (size_t)MMA_MATMUL_BF16_MAX_SHARED_BYTES) {
        fprintf(stderr, "error: mma_matmul_bf16 needs block_rows_per_group 1 or more, parts 1 to %d within %d B of shared memory "
                        "(%zu B a part), blocks 0 or more; got %d, %d, %d\n", MMA_MATMUL_BF16_MAX_PARTS, MMA_MATMUL_BF16_MAX_SHARED_BYTES,
                mma_matmul_bf16_part_bytes<SETTING>(), block_rows_per_group_val, parts_val, blocks_val);
        exit(EXIT_FAILURE);
    }
    // 2. launch, for the turned flags
    nv_bfloat16* output_ptr = output_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* left_ptr = left_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* right_ptr = right_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* bias_ptr = bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr;
    cudaError_t error_val;
    if (!left_turned_val && !right_turned_val)
        error_val = mma_matmul_bf16_launch<SETTING, false, false>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                  sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    else if (!left_turned_val && right_turned_val)
        error_val = mma_matmul_bf16_launch<SETTING, false, true>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                 sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    else if (left_turned_val && !right_turned_val)
        error_val = mma_matmul_bf16_launch<SETTING, true, false>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                 sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    else
        error_val = mma_matmul_bf16_launch<SETTING, true, true>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                sum_length_val, add_onto_val, block_rows_per_group_val, parts_val, blocks_val);
    cudaCheck(error_val);
}

/// Launches one mma_matmul_bf16_kernel, and gives the launch's error (cudaSuccess if it started).
///
///   output_ptr, left_ptr, right_ptr, bias_ptr    as in mma_matmul_bf16; bias nullptr for none
///   rows_val, columns_val, sum_length_val        the product's sizes (checked by the caller)
///   add_onto_val                                 as in mma_matmul_bf16
///   block_rows_per_group_val, parts_val, blocks_val   as in mma_matmul_bf16_at (checked by the caller)
template <typename SETTING, bool LEFT_TURNED, bool RIGHT_TURNED>
inline cudaError_t mma_matmul_bf16_launch(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr,
                                          const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                                          bool add_onto_val, int block_rows_per_group_val, int parts_val, int blocks_val) {
    // The parts in dynamic shared memory, the kernel allowed up to MMA_MATMUL_BF16_MAX_SHARED_BYTES the first time (a
    // tune skips the settings that do not fit rather than stopping: the error is given back). Blocks: one a square,
    // or blocks_val, never more than the squares.
    auto kernel_ptr = mma_matmul_bf16_kernel<SETTING, LEFT_TURNED, RIGHT_TURNED>;
    static bool shared_set_val = false;
    if (!shared_set_val) {
        cudaError_t error_val = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, MMA_MATMUL_BF16_MAX_SHARED_BYTES);
        if (error_val != cudaSuccess) return error_val;
        shared_set_val = true;
    }
    size_t squares_val = CEIL_DIV(rows_val, (size_t)SETTING::BLOCK_ROWS) * CEIL_DIV(columns_val, (size_t)SETTING::BLOCK_COLUMNS);
    unsigned launched_val = (unsigned)(blocks_val > 0 && (size_t)blocks_val < squares_val ? (size_t)blocks_val : squares_val);
    size_t shared_bytes_val = (size_t)parts_val * mma_matmul_bf16_part_bytes<SETTING>();
    kernel_ptr<<<launched_val, SETTING::THREADS, shared_bytes_val>>>(output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val,
                                                                     sum_length_val, add_onto_val, block_rows_per_group_val, parts_val);
    return cudaGetLastError();
}

/// Works out this block's squares of output: squares b, b + blocks, ...
///
///   output_ptr, left_ptr, right_ptr, bias_ptr    as in mma_matmul_bf16; bias nullptr for none
///   rows_val, columns_val, sum_length_val        the product's sizes
///   add_onto_val                                 as in mma_matmul_bf16
///   block_rows_per_group_val                     the block order
///   parts_val                                    parts of shared memory, a step's tiles each
template <typename SETTING, bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(SETTING::THREADS)
mma_matmul_bf16_kernel(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr, const nv_bfloat16* bias_ptr,
                       size_t rows_val, size_t columns_val, size_t sum_length_val, bool add_onto_val, int block_rows_per_group_val,
                       int parts_val) {
    // For each square, step by step: the threads copy a step's tile of left and of right into a part of shared
    // memory (cp.async, 16 bytes a copy, no registers on the way), then each warp adds its part of their product
    // into its sums on the tensor cores.
    //
    //     1 part:   copy the step, wait for it, multiply, then the next step
    //     n parts:  the copies of the next n - 1 steps are in flight while a step is multiplied; a step done frees its
    //               part for the step n on
    //
    // LEFT_TURNED, RIGHT_TURNED: how left and right sit, so each copy and each register read is made for its layout.
    using LeftTile = MmaMatmulBf16Tile<SETTING, SETTING::BLOCK_ROWS>;
    using RightTile = MmaMatmulBf16Tile<SETTING, SETTING::BLOCK_COLUMNS>;
    extern __shared__ __align__(16) nv_bfloat16 mma_matmul_bf16_shared_list[];
    nv_bfloat16* left_parts_ptr = mma_matmul_bf16_shared_list;                              // part p at p x LeftTile::VALUES
    nv_bfloat16* right_parts_ptr = mma_matmul_bf16_shared_list + (size_t)parts_val * LeftTile::VALUES;   // ... RightTile::VALUES
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_first_row_index_val = (warp_index_val / SETTING::WARPS_ACROSS) * SETTING::WARP_ROWS;
    int warp_first_column_index_val = (warp_index_val % SETTING::WARPS_ACROSS) * SETTING::WARP_COLUMNS;
    size_t squares_val = CEIL_DIV(rows_val, (size_t)SETTING::BLOCK_ROWS) * CEIL_DIV(columns_val, (size_t)SETTING::BLOCK_COLUMNS);
    int total_steps_val = (int)CEIL_DIV(sum_length_val, (size_t)SETTING::BLOCK_K);
    // asks for a step's two tiles into its part (step % parts), as one group of copies
    auto copy_step_val = [&](int step_index_val, size_t first_row_index_val, size_t first_column_index_val) {
        int part_index_val = step_index_val % parts_val;
        size_t step_first_k_index_val = (size_t)step_index_val * SETTING::BLOCK_K;
        mma_matmul_bf16_copy_tile<SETTING, SETTING::BLOCK_ROWS, LEFT_TURNED>(left_parts_ptr + (size_t)part_index_val * LeftTile::VALUES,
                                                                             left_ptr, rows_val, sum_length_val, first_row_index_val,
                                                                             step_first_k_index_val);
        mma_matmul_bf16_copy_tile<SETTING, SETTING::BLOCK_COLUMNS, RIGHT_TURNED>(right_parts_ptr + (size_t)part_index_val * RightTile::VALUES,
                                                                                 right_ptr, columns_val, sum_length_val, first_column_index_val,
                                                                                 step_first_k_index_val);
    };

    for (size_t square_index_val = blockIdx.x; square_index_val < squares_val; square_index_val += gridDim.x) {
        // 1. the square, and this warp's part of it
        size_t first_row_index_val = 0, first_column_index_val = 0;
        mma_matmul_bf16_square<SETTING>(square_index_val, rows_val, columns_val, block_rows_per_group_val, &first_row_index_val,
                                        &first_column_index_val);
        float sums_list[SETTING::M_TILES][SETTING::N_TILES][HARDWARE_MMA_C_VALUES_PER_THREAD_BF16] = {};

        // 2. the first parts - 1 steps asked for, a group each (empty past the last step, so the count of groups holds)
        for (int step_index_val = 0; step_index_val < parts_val - 1; step_index_val++) {
            if (step_index_val < total_steps_val) copy_step_val(step_index_val, first_row_index_val, first_column_index_val);
            mma_matmul_bf16_copy_commit();
        }

        // 3. walk the sum: wait for the step's part, ask for the step parts - 1 on into the part just freed, multiply
        for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
            if (parts_val == 1) {
                // one part: this step's copies, waited for, before anyone reads them
                copy_step_val(step_index_val, first_row_index_val, first_column_index_val);
                mma_matmul_bf16_copy_commit();
                mma_matmul_bf16_copy_wait(0);
                __syncthreads();
            } else {
                // groups asked for: step + parts - 1; this step's is in once at most parts - 2 are left in flight
                mma_matmul_bf16_copy_wait(parts_val - 2);
                __syncthreads();                    // every thread's copies in, and every warp done with the step before
                int next_step_index_val = step_index_val + parts_val - 1;   // into the part the step before used
                if (next_step_index_val < total_steps_val) copy_step_val(next_step_index_val, first_row_index_val, first_column_index_val);
                mma_matmul_bf16_copy_commit();
            }
            const nv_bfloat16* left_tile_ptr = left_parts_ptr + (size_t)(step_index_val % parts_val) * LeftTile::VALUES;
            const nv_bfloat16* right_tile_ptr = right_parts_ptr + (size_t)(step_index_val % parts_val) * RightTile::VALUES;
            // the tensor cores: for each 16 values of k, this warp's A and B registers, then every A x B
#pragma unroll
            for (int k_step_index_val = 0; k_step_index_val < SETTING::K_STEPS; k_step_index_val++) {
                int first_k_index_val = k_step_index_val * HARDWARE_MMA_K_BF16;
                uint32_t a_list[SETTING::M_TILES][HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16];
                uint32_t b_list[SETTING::N_TILES][HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16];
#pragma unroll
                for (int m_tile_index_val = 0; m_tile_index_val < SETTING::M_TILES; m_tile_index_val++)
                    mma_matmul_bf16_load_a<SETTING, LEFT_TURNED>(a_list[m_tile_index_val], left_tile_ptr,
                                                                 warp_first_row_index_val + m_tile_index_val * HARDWARE_MMA_M, first_k_index_val);
#pragma unroll
                for (int n_tile_index_val = 0; n_tile_index_val < SETTING::N_TILES; n_tile_index_val++)
                    mma_matmul_bf16_load_b<SETTING, RIGHT_TURNED>(b_list[n_tile_index_val], right_tile_ptr,
                                                                  warp_first_column_index_val + n_tile_index_val * HARDWARE_MMA_N, first_k_index_val);
#pragma unroll
                for (int m_tile_index_val = 0; m_tile_index_val < SETTING::M_TILES; m_tile_index_val++)
#pragma unroll
                    for (int n_tile_index_val = 0; n_tile_index_val < SETTING::N_TILES; n_tile_index_val++)
                        mma_matmul_bf16_multiply_add(sums_list[m_tile_index_val][n_tile_index_val], a_list[m_tile_index_val],
                                                     b_list[n_tile_index_val]);
            }
            if (parts_val == 1) __syncthreads();    // every warp done with the one part before the next step's copies
        }
        mma_matmul_bf16_copy_wait(0);               // the empty groups too: none left for the next square
        __syncthreads();                            // every warp done with the parts before the next square's copies

        // 4. output = sums (+ bias) (+ output), two neighbouring columns at a time; nothing past the edges. Each C
        //    tile: rows group and group + 8, columns pair and pair + 1 (the mma layout)
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
                    int c_index_val = HARDWARE_MMA_COLUMNS_PER_THREAD * thread_row_index_val;   // this row's 2 sums in the C tile
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

/// Gives a square's first row and column from its number.
///
///   square_index_val                                  the square's number
///   rows_val, columns_val                             the product's sizes
///   block_rows_per_group_val                          the block order
///   first_row_index_ptr, first_column_index_ptr       written: the square's first row and column
template <typename SETTING>
__device__ __forceinline__ void mma_matmul_bf16_square(size_t square_index_val, size_t rows_val, size_t columns_val,
                                                       int block_rows_per_group_val, size_t* first_row_index_ptr,
                                                       size_t* first_column_index_ptr) {
    // Groups of block_rows_per_group rows of squares, down a group first, so the squares worked on at once share
    // bands of left and right in L2.
    size_t total_square_columns_val = CEIL_DIV(columns_val, (size_t)SETTING::BLOCK_COLUMNS);
    size_t total_square_rows_val = CEIL_DIV(rows_val, (size_t)SETTING::BLOCK_ROWS);
    size_t squares_per_group_val = (size_t)block_rows_per_group_val * total_square_columns_val;
    size_t group_index_val = square_index_val / squares_per_group_val;
    size_t group_first_square_row_index_val = group_index_val * block_rows_per_group_val;
    size_t group_square_rows_val = total_square_rows_val - group_first_square_row_index_val;   // fewer in the last group
    if (group_square_rows_val > (size_t)block_rows_per_group_val) group_square_rows_val = block_rows_per_group_val;
    size_t square_index_in_group_val = square_index_val % squares_per_group_val;
    *first_row_index_ptr = (group_first_square_row_index_val + square_index_in_group_val % group_square_rows_val) * SETTING::BLOCK_ROWS;
    *first_column_index_ptr = (square_index_in_group_val / group_square_rows_val) * SETTING::BLOCK_COLUMNS;
}

/// Asks for one step's tile of left or right, 16 bytes a copy from GPU memory straight into a part of shared memory.
///
///   tile_ptr                       the part's tile of left or right
///   matrix_ptr                     left or right
///   lines_val, sum_length_val      its matrix's sizes
///   first_line_index_val           the square's first line
///   step_first_k_index_val         the step's first k
template <typename SETTING, int LINES, bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_copy_tile(nv_bfloat16* tile_ptr, const nv_bfloat16* matrix_ptr, size_t lines_val,
                                                          size_t sum_length_val, size_t first_line_index_val, size_t step_first_k_index_val) {
    // Thread t asks for copies t, t + THREADS, ...; 0s past the matrix's edges (cp.async reads 0 bytes there and fills
    // the rest with 0s; a copy is all in or all out: the last dim is a multiple of MMA_MATMUL_BF16_LOAD_VALUES). Each
    // line (a row of left, a column of output for right) has sum_length values of k.
    //     not TURNED: the matrix sits as {lines, sum_length}: a copy is 8 neighbouring k of one line, into tile[line][k]
    //     TURNED:     it sits as {sum_length, lines}: a copy is 8 neighbouring lines at one k, into tile[k][line]
    using Tile = MmaMatmulBf16Tile<SETTING, LINES>;
    for (int piece_index_val = threadIdx.x; piece_index_val < Tile::PIECES; piece_index_val += SETTING::THREADS) {
        size_t line_index_val, k_index_val;
        int offset_val;
        if constexpr (TURNED) {
            const int copies_per_k_val = LINES / MMA_MATMUL_BF16_LOAD_VALUES;
            k_index_val = step_first_k_index_val + piece_index_val / copies_per_k_val;
            line_index_val = first_line_index_val + (piece_index_val % copies_per_k_val) * MMA_MATMUL_BF16_LOAD_VALUES;
            offset_val = (piece_index_val / copies_per_k_val) * Tile::LINE_ROW_LENGTH + (piece_index_val % copies_per_k_val) * MMA_MATMUL_BF16_LOAD_VALUES;
        } else {
            const int copies_per_line_val = SETTING::BLOCK_K / MMA_MATMUL_BF16_LOAD_VALUES;
            line_index_val = first_line_index_val + piece_index_val / copies_per_line_val;
            k_index_val = step_first_k_index_val + (piece_index_val % copies_per_line_val) * MMA_MATMUL_BF16_LOAD_VALUES;
            offset_val = (piece_index_val / copies_per_line_val) * SETTING::K_ROW_LENGTH +
                         (piece_index_val % copies_per_line_val) * MMA_MATMUL_BF16_LOAD_VALUES;
        }
        bool inside_val = line_index_val < lines_val && k_index_val < sum_length_val;
        const nv_bfloat16* source_ptr = inside_val ? matrix_ptr + (TURNED ? k_index_val * lines_val + line_index_val
                                                                          : line_index_val * sum_length_val + k_index_val)
                                                   : matrix_ptr;   // a valid address; 0 bytes read from it
        uint32_t shared_address_val = (uint32_t)__cvta_generic_to_shared(tile_ptr + offset_val);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
                     ::"r"(shared_address_val), "l"(source_ptr), "n"(MMA_MATMUL_BF16_LOAD_BYTES), "r"(inside_val ? MMA_MATMUL_BF16_LOAD_BYTES : 0)
                     : "memory");
    }
}

/// Closes this thread's copies asked for since the last group into one group.
__device__ __forceinline__ void mma_matmul_bf16_copy_commit(void) {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

/// Waits until at most steps_in_flight_val of this thread's groups of copies are still on their way.
///
///   steps_in_flight_val   0 to MMA_MATMUL_BF16_MAX_PARTS - 2
__device__ __forceinline__ void mma_matmul_bf16_copy_wait(int steps_in_flight_val) {
    // cp.async.wait_group takes its count as a constant: one case each.
    switch (steps_in_flight_val) {
        case 0: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
        case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
        case 2: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
        case 3: asm volatile("cp.async.wait_group 3;\n" ::: "memory"); break;
        case 4: asm volatile("cp.async.wait_group 4;\n" ::: "memory"); break;
        case 5: asm volatile("cp.async.wait_group 5;\n" ::: "memory"); break;
        default: asm volatile("cp.async.wait_group 6;\n" ::: "memory"); break;
    }
}
static_assert(MMA_MATMUL_BF16_MAX_PARTS - 2 == 6, "mma_matmul_bf16_copy_wait has a case for each count up to MMA_MATMUL_BF16_MAX_PARTS - 2");

/// Reads this thread's A registers of one mma tile out of left's tile in shared memory.
///
///   a_list                                    written: HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16 registers
///   tile_ptr                                  left's tile
///   first_row_index_val, first_k_index_val    the mma tile's {HARDWARE_MMA_M, HARDWARE_MMA_K_BF16} corner in the tile
template <typename SETTING, bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_a(uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_row_index_val, int first_k_index_val) {
    // Reads this thread's A registers of one mma tile out of left's tile in shared memory. With group = lane / 4 and
    // pair = 2 (lane % 4), the registers hold A's (group, pair .. pair+1), (group+8, ..), (group, pair+8 ..),
    // (group+8, pair+8 ..), each 2 neighbouring k.
    //     not TURNED: tile[row][k]: the 2 values sit side by side, one 4-byte read
    //     TURNED:     tile[k][row]: they sit a tile row apart, two reads packed into one register
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int pair_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
    const int half_k_val = HARDWARE_MMA_K_BF16 / 2;               // registers 2 and 3: 8 values of k on
    if constexpr (TURNED) {
        const int row_val = MmaMatmulBf16Tile<SETTING, SETTING::BLOCK_ROWS>::LINE_ROW_LENGTH;   // from one k to the next
        const nv_bfloat16* at_ptr = tile_ptr + (first_k_index_val + pair_index_val) * row_val + first_row_index_val + group_index_val;
        a_list[0] = mma_matmul_bf16_pack(at_ptr[0], at_ptr[row_val]);
        a_list[1] = mma_matmul_bf16_pack(at_ptr[HARDWARE_MMA_ROWS_PER_WARP], at_ptr[row_val + HARDWARE_MMA_ROWS_PER_WARP]);
        a_list[2] = mma_matmul_bf16_pack(at_ptr[half_k_val * row_val], at_ptr[(half_k_val + 1) * row_val]);
        a_list[3] = mma_matmul_bf16_pack(at_ptr[half_k_val * row_val + HARDWARE_MMA_ROWS_PER_WARP],
                                         at_ptr[(half_k_val + 1) * row_val + HARDWARE_MMA_ROWS_PER_WARP]);
    } else {
        const nv_bfloat16* top_ptr = tile_ptr + (first_row_index_val + group_index_val) * SETTING::K_ROW_LENGTH + first_k_index_val + pair_index_val;
        const nv_bfloat16* bottom_ptr = top_ptr + HARDWARE_MMA_ROWS_PER_WARP * SETTING::K_ROW_LENGTH;
        a_list[0] = *reinterpret_cast<const uint32_t*>(top_ptr);
        a_list[1] = *reinterpret_cast<const uint32_t*>(bottom_ptr);
        a_list[2] = *reinterpret_cast<const uint32_t*>(top_ptr + half_k_val);
        a_list[3] = *reinterpret_cast<const uint32_t*>(bottom_ptr + half_k_val);
    }
}

/// Reads this thread's B registers of one mma tile out of right's tile in shared memory.
///
///   b_list                                       written: HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16 registers
///   tile_ptr                                     right's tile
///   first_column_index_val, first_k_index_val    the mma tile's first column (n) and first k in the tile
template <typename SETTING, bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_b(uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_column_index_val, int first_k_index_val) {
    // Reads this thread's B registers of one mma tile out of right's tile in shared memory: B's rows (pair .. pair+1,
    // column group) and (pair+8 .. pair+9, column group), B[k][n] = right[n][k].
    //     not TURNED: tile[column][k]: the 2 values sit side by side, one 4-byte read
    //     TURNED:     tile[k][column]: they sit a tile row apart, two reads packed into one register
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int pair_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
    const int half_k_val = HARDWARE_MMA_K_BF16 / 2;
    if constexpr (TURNED) {
        const int row_val = MmaMatmulBf16Tile<SETTING, SETTING::BLOCK_COLUMNS>::LINE_ROW_LENGTH;
        const nv_bfloat16* at_ptr = tile_ptr + (first_k_index_val + pair_index_val) * row_val + first_column_index_val + group_index_val;
        b_list[0] = mma_matmul_bf16_pack(at_ptr[0], at_ptr[row_val]);
        b_list[1] = mma_matmul_bf16_pack(at_ptr[half_k_val * row_val], at_ptr[(half_k_val + 1) * row_val]);
    } else {
        const nv_bfloat16* at_ptr = tile_ptr + (first_column_index_val + group_index_val) * SETTING::K_ROW_LENGTH + first_k_index_val + pair_index_val;
        b_list[0] = *reinterpret_cast<const uint32_t*>(at_ptr);
        b_list[1] = *reinterpret_cast<const uint32_t*>(at_ptr + half_k_val);
    }
}

/// C += A · B on the tensor cores, for the whole warp: mma.sync m16n8k16, bfloat16 in, float sums.
///
///   c_list   this thread's HARDWARE_MMA_C_VALUES_PER_THREAD_BF16 floats of C
///   a_list   this thread's A registers
///   b_list   this thread's B registers
__device__ __forceinline__ void mma_matmul_bf16_multiply_add(float (&c_list)[HARDWARE_MMA_C_VALUES_PER_THREAD_BF16],
                                                             const uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16],
                                                             const uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c_list[0]), "+f"(c_list[1]), "+f"(c_list[2]), "+f"(c_list[3])
                 : "r"(a_list[0]), "r"(a_list[1]), "r"(a_list[2]), "r"(a_list[3]), "r"(b_list[0]), "r"(b_list[1]));
#endif
}

/// Gives two bfloat16s as one register, the first in the low half.
///
///   low_val, high_val   the two values
__device__ __forceinline__ uint32_t mma_matmul_bf16_pack(nv_bfloat16 low_val, nv_bfloat16 high_val) {
    return (uint32_t)__bfloat16_as_ushort(low_val) | ((uint32_t)__bfloat16_as_ushort(high_val) << 16);
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Stops with a message unless the tensors fit, and gives the product's sizes.
///
///   output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val   as in mma_matmul_bf16
///   rows_ptr, columns_ptr, sum_length_ptr                                                       written: the sizes
inline void mma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                         const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                         size_t* columns_ptr, size_t* sum_length_ptr) {
    // Left's and right's sum_length the same, output rows x columns with columns last, bias {columns}; all
    // bfloat16; every last dim a multiple of MMA_MATMUL_BF16_LOAD_VALUES and every tensor starting on 16 bytes
    // (the 16-byte copies).
    // 1. shapes: each matrix's last dim and the rest, by how it sits
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = left_tensor.dim() >= 1 && right_tensor.dim() >= 1 && output_tensor.dim() >= 1;
    size_t left_last_val = fits_val ? left_tensor.size(left_tensor.dim() - 1) : 1;
    size_t right_last_val = fits_val ? right_tensor.size(right_tensor.dim() - 1) : 1;
    size_t rows_val = left_turned_val ? left_last_val : left_tensor.numel() / left_last_val;
    size_t sum_length_val = left_turned_val ? left_tensor.numel() / left_last_val : left_last_val;
    size_t columns_val = right_turned_val ? right_last_val : right_tensor.numel() / right_last_val;
    size_t right_sum_length_val = right_turned_val ? right_tensor.numel() / right_last_val : right_last_val;
    fits_val = fits_val && sum_length_val == right_sum_length_val && output_tensor.size(output_tensor.dim() - 1) == columns_val &&
               output_tensor.numel() == rows_val * columns_val && (!has_bias_val || bias_tensor.shape() == Shape({columns_val}));
    if (!fits_val) {
        fprintf(stderr, "error: mma_matmul_bf16 needs left {rows, sum_length} (turned {sum_length, rows}), right {columns, "
                        "sum_length} (turned {sum_length, columns}), output {rows, columns}, bias {columns} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16
    if (left_tensor.dtype() != DType::BFloat16 || right_tensor.dtype() != DType::BFloat16 ||
        output_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: mma_matmul_bf16 needs output, left, right and bias all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: whole 16-byte copies, each tensor starting on one
    const size_t load_bytes_val = sizeof(uint4);
    if (left_last_val % MMA_MATMUL_BF16_LOAD_VALUES != 0 || right_last_val % MMA_MATMUL_BF16_LOAD_VALUES != 0 ||
        columns_val % MMA_MATMUL_BF16_LOAD_VALUES != 0 || (size_t)output_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)left_tensor.data_ptr() % load_bytes_val != 0 || (size_t)right_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)bias_tensor.data_ptr() % load_bytes_val != 0) {
        fprintf(stderr, "error: mma_matmul_bf16 reads %zu bytes at a time, so every tensor's last dim must be a multiple of %d "
                        "and every tensor must start on %zu bytes; got last dims %zu (left), %zu (right), %zu (output)\n",
                load_bytes_val, MMA_MATMUL_BF16_LOAD_VALUES, load_bytes_val, left_last_val, right_last_val, columns_val);
        exit(EXIT_FAILURE);
    }
    *rows_ptr = rows_val;
    *columns_ptr = columns_val;
    *sum_length_ptr = sum_length_val;
}

#endif // MMA_MATMUL_BF16_CUH
