// adamw_tma_bf16.cuh — adamw_bf16's step with TMA: TMA brings each tile of the master (or params), grads, m and v into shared memory, the threads do the AdamW math from there, TMA (or the threads) write the results back.
//
//     tensor          type in memory   shape
//     params          bfloat16         {params_in_slice}
//     grads           bfloat16         {params_in_slice}
//     master params   float            {params_in_slice}, or none
//     m, v            bfloat16         {params_in_slice}
//     a tile          the same types   {tile_values} of each in shared memory: master (or params), grads, m, v
#ifndef ADAMW_TMA_BF16_CUH
#define ADAMW_TMA_BF16_CUH

#include "adamw_bf16.cuh"             // adamw_bf16_update_param: its math, copied with every rounding fixed (adamw_tma_bf16_update_param)

// The settings: our choices, swept by tune_adamw_tma_bf16. Defaults until a tune sets them.
#define ADAMW_TMA_BF16_VALUES 8                      // values a thread updates a round: whole float4s (4 values)
#define ADAMW_TMA_BF16_WARPS 8                       // warps a block: 1 to HARDWARE_MAX_WARPS_PER_BLOCK
#define ADAMW_TMA_BF16_TMA_STORE true                // true: TMA writes the results out of shared memory; false: the threads write them
#define ADAMW_TMA_BF16_BLOCKS_PER_SM 2               // blocks a launch, per SM: each takes its share of the SM's shared memory
#define ADAMW_TMA_BF16_STAGES 4                      // parts the share is cut into, a tile each: 1 to ADAMW_TMA_BF16_MAX_STAGES
#define ADAMW_TMA_BF16_MAX_STAGES 8                  // the most parts
#define ADAMW_TMA_BF16_SHARE_ROUND_BYTES 1024        // a share rounded down to this: shared memory is handed out in steps, so B shares still fit B blocks
#define ADAMW_TMA_BF16_MAX_SHARED_BYTES (HARDWARE_MAX_SHARED_BYTES_PER_BLOCK - ADAMW_TMA_BF16_MAX_STAGES * (int)sizeof(uint64_t))   // the dynamic shared memory a block can ask for: the rest is the barriers
#define ADAMW_TMA_BF16_VALUES_PER_COPY (HARDWARE_TMA_BYTES_PER_CELL / (int)sizeof(nv_bfloat16))   // a bulk copy moves whole 16-byte cells: 8 bfloat16s

// ── the step ──────────────────────────────────────────────────────────────────
struct AdamwTmaBf16Step;
struct AdamwTmaBf16Tile;
void adamw_tma_bf16_update(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr, nv_bfloat16* v_ptr,
                           size_t params_in_slice_val, size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val,
                           float learning_rate_val, float beta1_val, float beta2_val, float eps_val, float weight_decay_val,
                           int t_val, float* master_params_ptr);
template <int VALUES, int WARPS, bool TMA_STORE>
bool adamw_tma_bf16_update_at(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr, nv_bfloat16* v_ptr,
                              size_t params_in_slice_val, size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val,
                              float learning_rate_val, float beta1_val, float beta2_val, float eps_val, float weight_decay_val,
                              int t_val, float* master_params_ptr, int blocks_val, int tile_values_val, int stages_val);

// ── for it ────────────────────────────────────────────────────────────────────
void adamw_tma_bf16_check(const nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, const nv_bfloat16* m_ptr,
                          const nv_bfloat16* v_ptr, size_t params_in_slice_val, size_t decayed_count_val,
                          const float* master_params_ptr);
AdamwTmaBf16Step adamw_tma_bf16_step(size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val, float learning_rate_val,
                                     float beta1_val, float beta2_val, float eps_val, float weight_decay_val, int t_val);
bool adamw_tma_bf16_plan(AdamwTmaBf16Tile* tile_ptr, size_t* shared_bytes_ptr, bool has_master_val, int tile_values_val,
                         int stages_val, size_t params_in_slice_val);
int adamw_tma_bf16_share_bytes(int blocks_per_sm_val);
int adamw_tma_bf16_tile_values(int share_bytes_val, int stages_val, bool has_master_val);

// ══ the code, in the order of the list above ══════════════════════════════════

/// This step's numbers, the same for every value: worked out once on the host, as adamw_update_bf16 does.
struct AdamwTmaBf16Step {
    size_t decayed_count;                     // the first this many values get weight decay (the matrices)
    const float* grad_norm_ptr;               // norm of all the gradients, on the GPU
    float max_norm;                           // clip to this norm; 0: no clipping
    float weight_decay_factor;                // 1 - learning_rate x weight_decay
    float beta1, beta2, eps;                  // 0.9, 0.95, 1e-8
    float step_size;                          // learning_rate / (1 - beta1^t)
    float inverse_sqrt_bias_correction2;      // 1 / sqrt(1 - beta2^t)
};

/// One launch's tile: its size, how a set sits in shared memory, and the values TMA moves.
struct AdamwTmaBf16Tile {
    int tile_values;                  // values a tile
    int stages;                       // parts of shared memory, a tile each: TMA fills some while the threads work on one
    int first_bytes;                  // a set's first region: the tile's master (4 bytes a value) or params (2)
    int bf16_bytes;                   // each of the next three regions: grads, m, v (2 bytes a value)
    int set_bytes;                    // the four regions
    size_t copied_values;             // params_in_slice rounded down to ADAMW_TMA_BF16_VALUES_PER_COPY: what TMA moves
    size_t params_in_slice;           // every value; the rest past copied_values (0 or 4) taken from global memory
};

// ── the step ──────────────────────────────────────────────────────────────────

template <int VALUES, int WARPS, bool TMA_STORE>
__global__ void __launch_bounds__(WARPS * HARDWARE_THREADS_PER_WARP, 1)   // one block an SM is enough: all of its registers
adamw_tma_bf16_kernel(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr, nv_bfloat16* v_ptr,
                      float* master_params_ptr, AdamwTmaBf16Step step_val, AdamwTmaBf16Tile tile_val);
__device__ __forceinline__ float adamw_tma_bf16_update_param(float p_val, float g_val, float* m_ptr, float* v_ptr, float clip_val,
                                                             float decay_factor_val, const AdamwTmaBf16Step& step_val);
__device__ __forceinline__ void adamw_tma_bf16_update_four(float4& p_val, float4 g_val, float4& m_val, float4& v_val, float clip_val,
                                                           float decay_factor_val, const AdamwTmaBf16Step& step_val);
__device__ __forceinline__ uint32_t adamw_tma_bf16_shared_address(const void* shared_ptr);
__device__ __forceinline__ void adamw_tma_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val);
__device__ __forceinline__ void adamw_tma_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val);
__device__ __forceinline__ void adamw_tma_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val);
__device__ __forceinline__ void adamw_tma_bf16_copy_in(void* shared_ptr, const void* global_ptr, uint32_t bytes_val, uint64_t* barrier_ptr);
__device__ __forceinline__ void adamw_tma_bf16_copy_out(void* global_ptr, const void* shared_ptr, uint32_t bytes_val);

/// One AdamW step on all the parameters at the default settings: the same arguments as adamw_update_bf16.
///
///   params_ptr, grads_ptr           bfloat16: the parameters (matrices first) and their gradients
///   m_ptr, v_ptr                    bfloat16; all 0 at the start
///   params_in_slice_val             values to update: this GPU's slice
///   decayed_count_val               the first this many values get weight decay (the matrices)
///   grad_norm_ptr                   norm of all the gradients
///   max_norm_val                    clip to this norm; 0: no clipping
///   learning_rate_val               the learning rate of this step
///   beta1_val, beta2_val, eps_val   0.9, 0.95, 1e-8
///   weight_decay_val                0.1
///   t_val                           the step number: 1, 2, 3, ...
///   master_params_ptr               float copy of the params; nullptr: none
inline void adamw_tma_bf16_update(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr, nv_bfloat16* v_ptr,
                                  size_t params_in_slice_val, size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val,
                                  float learning_rate_val, float beta1_val, float beta2_val, float eps_val, float weight_decay_val,
                                  int t_val, float* master_params_ptr) {
    // The ADAMW_TMA_BF16_ settings, ADAMW_TMA_BF16_BLOCKS_PER_SM blocks an SM, each block's share of shared memory cut
    // into ADAMW_TMA_BF16_STAGES parts, a tile each. The same bits as adamw_update_bf16: the same float math on the
    // same values. Needs an H100 (TMA).
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int tile_values_val = adamw_tma_bf16_tile_values(adamw_tma_bf16_share_bytes(ADAMW_TMA_BF16_BLOCKS_PER_SM), ADAMW_TMA_BF16_STAGES,
                                                     master_params_ptr != nullptr);
    bool fits_val = adamw_tma_bf16_update_at<ADAMW_TMA_BF16_VALUES, ADAMW_TMA_BF16_WARPS, ADAMW_TMA_BF16_TMA_STORE>(
        params_ptr, grads_ptr, m_ptr, v_ptr, params_in_slice_val, decayed_count_val, grad_norm_ptr, max_norm_val, learning_rate_val,
        beta1_val, beta2_val, eps_val, weight_decay_val, t_val, master_params_ptr, sms_val * ADAMW_TMA_BF16_BLOCKS_PER_SM,
        tile_values_val, ADAMW_TMA_BF16_STAGES);
    if (!fits_val) {
        fprintf(stderr, "error: adamw_tma_bf16_update's tile of %d values, %d parts does not fit shared memory\n", tile_values_val,
                ADAMW_TMA_BF16_STAGES);
        exit(EXIT_FAILURE);
    }
}

/// One AdamW step on all the parameters at the settings given, or gives false if the tile does not fit.
///
///   params_ptr ... master_params_ptr            as in adamw_tma_bf16_update
///   blocks_val                                  blocks a launch
///   tile_values_val, stages_val                 the tile and its sets
template <int VALUES, int WARPS, bool TMA_STORE>
bool adamw_tma_bf16_update_at(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr, nv_bfloat16* v_ptr,
                              size_t params_in_slice_val, size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val,
                              float learning_rate_val, float beta1_val, float beta2_val, float eps_val, float weight_decay_val,
                              int t_val, float* master_params_ptr, int blocks_val, int tile_values_val, int stages_val) {
    // Checks, plans the tile, works out the step's numbers, then launches adamw_tma_bf16_kernel.
    // 1. check the counts and addresses (wrong: stop with an error), and that the tile fits (else false)
    adamw_tma_bf16_check(params_ptr, grads_ptr, m_ptr, v_ptr, params_in_slice_val, decayed_count_val, master_params_ptr);
    AdamwTmaBf16Tile tile_val;
    size_t shared_bytes_val = 0;
    if (!adamw_tma_bf16_plan(&tile_val, &shared_bytes_val, master_params_ptr != nullptr, tile_values_val, stages_val, params_in_slice_val))
        return false;
    // 2. this step's numbers
    AdamwTmaBf16Step step_val = adamw_tma_bf16_step(decayed_count_val, grad_norm_ptr, max_norm_val, learning_rate_val, beta1_val,
                                                    beta2_val, eps_val, weight_decay_val, t_val);
    // 3. launch: blocks_val blocks, each walking its tiles; the kernel's shared memory raised once
    auto kernel_ptr = adamw_tma_bf16_kernel<VALUES, WARPS, TMA_STORE>;
    static bool shared_set_val = false;
    if (!shared_set_val) {
        cudaCheck(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, ADAMW_TMA_BF16_MAX_SHARED_BYTES));
        shared_set_val = true;
    }
    kernel_ptr<<<blocks_val, WARPS * HARDWARE_THREADS_PER_WARP, shared_bytes_val>>>(params_ptr, grads_ptr, m_ptr, v_ptr,
                                                                                    master_params_ptr, step_val, tile_val);
    cudaCheck(cudaGetLastError());
    return true;
}

/// Walks this block's tiles: TMA brings each tile's four arrays, the threads update them, TMA (or they) write them out.
///
///   params_ptr, grads_ptr, m_ptr, v_ptr   bfloat16, {params_in_slice}: params, m and v written
///   master_params_ptr                     float, {params_in_slice}: written; nullptr: none, the params read instead
///   step_val                              this step's numbers
///   tile_val                              the tile's size and the counts
template <int VALUES, int WARPS, bool TMA_STORE>
__global__ void __launch_bounds__(WARPS * HARDWARE_THREADS_PER_WARP, 1)   // one block an SM is enough: all of its registers
adamw_tma_bf16_kernel(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr, nv_bfloat16* v_ptr,
                      float* master_params_ptr, AdamwTmaBf16Step step_val, AdamwTmaBf16Tile tile_val) {
    // A tile is tile_values values next to each other; tile k holds values k x tile_values on. Block b takes tiles b,
    // b + blocks, ...
    //
    //     a set of tiles in shared memory (one stage):
    //       first region:   the tile's master params (float), or its params (bfloat16) with no master
    //       grads region:   its grads (bfloat16)
    //       m region:       its m
    //       v region:       its v
    //
    // Thread 0 asks TMA for each set: one bulk copy an array (the values are next to each other: no tensor map),
    // counted at the set's mbarrier. Every thread then reads its float4s from the set, does main's AdamW math on
    // them, and writes the results either back into the set (TMA_STORE: the new master over the first region, the
    // new bfloat16 params over the grads region, which nobody reads again; m and v over theirs), for thread 0 to have
    // TMA store, or straight to global memory, as adamw_bf16_kernel does. Thread 0 then asks for the tile STAGES on
    // into the same set.
    //
    // Thread t's float4s in a tile: t, t + THREADS, ... (FOURS of them a round, then THREADS x FOURS on): a warp's
    // reads are next to each other, 8 bytes a lane for a bfloat16 array, 16 for the master: no bank conflict.
    constexpr int THREADS = WARPS * HARDWARE_THREADS_PER_WARP;
    constexpr int FOURS = VALUES / VALUES_PER_FLOAT4;                    // float4s a thread updates a round
    static_assert(VALUES % VALUES_PER_FLOAT4 == 0, "a thread updates whole float4s");
    extern __shared__ __align__(HARDWARE_TMA_BYTES_PER_CELL) uint8_t adamw_tma_bf16_shared_list[];
    __shared__ uint64_t full_barriers_list[ADAMW_TMA_BF16_MAX_STAGES];   // set s's bytes are in
    const bool has_master_val = master_params_ptr != nullptr;
    const int first_value_bytes_val = has_master_val ? (int)sizeof(float) : (int)sizeof(nv_bfloat16);
    const size_t tiles_val = CEIL_DIV(tile_val.copied_values, (size_t)tile_val.tile_values);

    // 1. gradient clipping: scale g down so its norm is at most max_norm; 1 if already below it
    float clip_val = 1.0f;
    if (step_val.max_norm > 0.0f) {
        clip_val = step_val.max_norm / (*step_val.grad_norm_ptr + 1e-6f);
        if (clip_val > 1.0f) clip_val = 1.0f;
    }

    // 2. the barriers, by one thread, visible to TMA
    if (threadIdx.x == 0) {
        for (int set_index_val = 0; set_index_val < tile_val.stages; set_index_val++)
            adamw_tma_bf16_barrier_init(&full_barriers_list[set_index_val], 1);   // thread 0, and the bytes
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");          // the barriers visible to TMA
    }
    __syncthreads();

    // gives a tile's values: tile_values, fewer for the last
    auto tile_count_val = [&](size_t tile_index_val) {
        size_t left_val = tile_val.copied_values - tile_index_val * tile_val.tile_values;
        return left_val < (size_t)tile_val.tile_values ? (int)left_val : tile_val.tile_values;
    };
    // thread 0 asks TMA for a tile's four arrays into set set_index_val: first tells the set's barrier the bytes coming
    auto ask_tile_val = [&](size_t tile_index_val, int set_index_val) {
        size_t first_value_val = tile_index_val * tile_val.tile_values;
        uint32_t count_val = (uint32_t)tile_count_val(tile_index_val);
        uint32_t first_bytes_val = count_val * first_value_bytes_val;
        uint32_t bf16_bytes_val = count_val * (uint32_t)sizeof(nv_bfloat16);
        uint8_t* set_ptr = adamw_tma_bf16_shared_list + (size_t)set_index_val * tile_val.set_bytes;
        uint64_t* barrier_ptr = &full_barriers_list[set_index_val];
        adamw_tma_bf16_barrier_expect(barrier_ptr, first_bytes_val + 3 * bf16_bytes_val);
        const void* first_ptr = has_master_val ? (const void*)(master_params_ptr + first_value_val) : (const void*)(params_ptr + first_value_val);
        adamw_tma_bf16_copy_in(set_ptr, first_ptr, first_bytes_val, barrier_ptr);
        adamw_tma_bf16_copy_in(set_ptr + tile_val.first_bytes, grads_ptr + first_value_val, bf16_bytes_val, barrier_ptr);
        adamw_tma_bf16_copy_in(set_ptr + tile_val.first_bytes + tile_val.bf16_bytes, m_ptr + first_value_val, bf16_bytes_val, barrier_ptr);
        adamw_tma_bf16_copy_in(set_ptr + tile_val.first_bytes + 2 * tile_val.bf16_bytes, v_ptr + first_value_val, bf16_bytes_val, barrier_ptr);
    };

    // 3. the values past the last whole 16-byte cell (0 or 4): block 0's thread 0, from global memory, as adamw_bf16_kernel
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (size_t value_index_val = tile_val.copied_values; value_index_val < tile_val.params_in_slice; value_index_val += VALUES_PER_FLOAT4) {
            float decay_factor_val = value_index_val < step_val.decayed_count ? step_val.weight_decay_factor : 1.0f;
            float4 p_val = has_master_val ? *reinterpret_cast<const float4*>(master_params_ptr + value_index_val)
                                          : read4_as_float(params_ptr + value_index_val);
            float4 m_val = read4_as_float(m_ptr + value_index_val);
            float4 v_val = read4_as_float(v_ptr + value_index_val);
            adamw_tma_bf16_update_four(p_val, read4_as_float(grads_ptr + value_index_val), m_val, v_val, clip_val, decay_factor_val, step_val);
            if (has_master_val) *reinterpret_cast<float4*>(master_params_ptr + value_index_val) = p_val;
            write4_from_float(params_ptr + value_index_val, p_val);
            write4_from_float(m_ptr + value_index_val, m_val);
            write4_from_float(v_ptr + value_index_val, v_val);
        }
    }

    // 4. thread 0 asks for the first STAGES tiles
    size_t my_tiles_val = blockIdx.x < tiles_val ? (tiles_val - 1 - blockIdx.x) / gridDim.x + 1 : 0;
    if (threadIdx.x == 0)
        for (int set_index_val = 0; set_index_val < tile_val.stages && (size_t)set_index_val < my_tiles_val; set_index_val++)
            ask_tile_val(blockIdx.x + (size_t)set_index_val * gridDim.x, set_index_val);

    // 5. each tile: wait for its set, update, write, ask for the tile STAGES on into the same set
    for (size_t walk_index_val = 0; walk_index_val < my_tiles_val; walk_index_val++) {
        int set_index_val = (int)(walk_index_val % tile_val.stages);
        size_t tile_index_val = blockIdx.x + walk_index_val * gridDim.x;
        size_t first_value_val = tile_index_val * tile_val.tile_values;
        int fours_val = tile_count_val(tile_index_val) / VALUES_PER_FLOAT4;
        adamw_tma_bf16_barrier_wait(&full_barriers_list[set_index_val], (uint32_t)((walk_index_val / tile_val.stages) % 2));
        uint8_t* set_ptr = adamw_tma_bf16_shared_list + (size_t)set_index_val * tile_val.set_bytes;
        float* master_tile_ptr = reinterpret_cast<float*>(set_ptr);                  // with a master
        nv_bfloat16* params_tile_ptr = reinterpret_cast<nv_bfloat16*>(set_ptr);      // with none
        nv_bfloat16* grads_tile_ptr = reinterpret_cast<nv_bfloat16*>(set_ptr + tile_val.first_bytes);
        nv_bfloat16* m_tile_ptr = reinterpret_cast<nv_bfloat16*>(set_ptr + tile_val.first_bytes + tile_val.bf16_bytes);
        nv_bfloat16* v_tile_ptr = reinterpret_cast<nv_bfloat16*>(set_ptr + tile_val.first_bytes + 2 * tile_val.bf16_bytes);

        // a. the threads: read FOURS float4s of each array, then update and write each
        for (int first_four_val = threadIdx.x; first_four_val < fours_val; first_four_val += THREADS * FOURS) {
            float4 p_list[FOURS], g_list[FOURS], m_list[FOURS], v_list[FOURS];
#pragma unroll
            for (int four_index_val = 0; four_index_val < FOURS; four_index_val++) {
                int four_val = first_four_val + four_index_val * THREADS;
                if (four_val >= fours_val) continue;
                int value_val = four_val * VALUES_PER_FLOAT4;                       // in the tile
                p_list[four_index_val] = has_master_val ? *reinterpret_cast<const float4*>(master_tile_ptr + value_val)
                                                        : read4_as_float(params_tile_ptr + value_val);
                g_list[four_index_val] = read4_as_float(grads_tile_ptr + value_val);
                m_list[four_index_val] = read4_as_float(m_tile_ptr + value_val);
                v_list[four_index_val] = read4_as_float(v_tile_ptr + value_val);
            }
#pragma unroll
            for (int four_index_val = 0; four_index_val < FOURS; four_index_val++) {
                int four_val = first_four_val + four_index_val * THREADS;
                if (four_val >= fours_val) continue;
                int value_val = four_val * VALUES_PER_FLOAT4;
                size_t global_value_val = first_value_val + value_val;
                float decay_factor_val = global_value_val < step_val.decayed_count ? step_val.weight_decay_factor : 1.0f;   // decay: matrices only
                adamw_tma_bf16_update_four(p_list[four_index_val], g_list[four_index_val], m_list[four_index_val], v_list[four_index_val],
                                           clip_val, decay_factor_val, step_val);
                if constexpr (TMA_STORE) {
                    // into the set: the master exact and the params rounded over the grads, or the params over the first region
                    if (has_master_val) {
                        *reinterpret_cast<float4*>(master_tile_ptr + value_val) = p_list[four_index_val];
                        write4_from_float(grads_tile_ptr + value_val, p_list[four_index_val]);
                    } else {
                        write4_from_float(params_tile_ptr + value_val, p_list[four_index_val]);
                    }
                    write4_from_float(m_tile_ptr + value_val, m_list[four_index_val]);
                    write4_from_float(v_tile_ptr + value_val, v_list[four_index_val]);
                } else {
                    // straight to global memory: the master exact, the rest rounded to bfloat16
                    if (has_master_val) *reinterpret_cast<float4*>(master_params_ptr + global_value_val) = p_list[four_index_val];
                    write4_from_float(params_ptr + global_value_val, p_list[four_index_val]);
                    write4_from_float(m_ptr + global_value_val, m_list[four_index_val]);
                    write4_from_float(v_ptr + global_value_val, v_list[four_index_val]);
                }
            }
        }
        if constexpr (TMA_STORE) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // this thread's results visible to TMA
        __syncthreads();                                                                          // every thread's (and every read of the set done)

        // b. thread 0: TMA stores the results (TMA_STORE), they are read out of the set, then the tile STAGES on comes into it
        if (threadIdx.x == 0) {
            if constexpr (TMA_STORE) {
                uint32_t count_val = (uint32_t)fours_val * VALUES_PER_FLOAT4;
                uint32_t bf16_bytes_val = count_val * (uint32_t)sizeof(nv_bfloat16);
                if (has_master_val) {
                    adamw_tma_bf16_copy_out(master_params_ptr + first_value_val, master_tile_ptr, count_val * (uint32_t)sizeof(float));
                    adamw_tma_bf16_copy_out(params_ptr + first_value_val, grads_tile_ptr, bf16_bytes_val);
                } else {
                    adamw_tma_bf16_copy_out(params_ptr + first_value_val, params_tile_ptr, bf16_bytes_val);
                }
                adamw_tma_bf16_copy_out(m_ptr + first_value_val, m_tile_ptr, bf16_bytes_val);
                adamw_tma_bf16_copy_out(v_ptr + first_value_val, v_tile_ptr, bf16_bytes_val);
                asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
            }
            if (walk_index_val + tile_val.stages < my_tiles_val) {
                if constexpr (TMA_STORE) asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");   // the stores have read the set
                ask_tile_val(tile_index_val + (size_t)tile_val.stages * gridDim.x, set_index_val);
            }
        }
    }
    // 6. the last stores written before the block ends
    if constexpr (TMA_STORE)
        if (threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

/// Moves one parameter one AdamW step, every rounding as adamw_bf16_kernel's compiled code does it; m and v written back.
///
///   p_val              the parameter
///   g_val              its gradient
///   m_ptr, v_ptr       its moments: read, then written back
///   clip_val           g is multiplied by it
///   decay_factor_val   p is multiplied by it
///   step_val           this step's numbers
__device__ __forceinline__ float adamw_tma_bf16_update_param(float p_val, float g_val, float* m_ptr, float* v_ptr, float clip_val,
                                                             float decay_factor_val, const AdamwTmaBf16Step& step_val) {
    // adamw_bf16_update_param's math, but no line left for the compiler to fuse its own way. The compiler may fuse a
    // multiply and an add into one FMA (one rounding, not two), and which of a line's multiplies it fuses depends on
    // the code around it: in adamw_bf16_kernel it fuses m = fma(g, 1 - beta1, beta1 x m), here it chose
    // fma(m, beta1, (1 - beta1) x g), and 0.8% of the masters came out a bit apart. So each step is written with the
    // rounding adamw_bf16_kernel's machine code (sm_90a, nvcc 13.4) has: _rn, round to nearest, one rounding each.
    g_val = __fmul_rn(g_val, clip_val);
    p_val = __fmul_rn(p_val, decay_factor_val);
    // m = fma(g, 1 - beta1, beta1 x m); v = fma(g, (1 - beta2) x g, beta2 x v)
    *m_ptr = __fmaf_rn(g_val, __fsub_rn(1.0f, step_val.beta1), __fmul_rn(step_val.beta1, *m_ptr));
    *v_ptr = __fmaf_rn(g_val, __fmul_rn(__fsub_rn(1.0f, step_val.beta2), g_val), __fmul_rn(step_val.beta2, *v_ptr));
    // sqrt(v) x 1 / sqrt(1 - beta2^t) + eps, one FMA; then p - (step_size x m) / denominator
    float denominator_val = __fmaf_rn(__fsqrt_rn(*v_ptr), step_val.inverse_sqrt_bias_correction2, step_val.eps);
    return __fsub_rn(p_val, __fdiv_rn(__fmul_rn(step_val.step_size, *m_ptr), denominator_val));
}

/// Moves four parameters one AdamW step with adamw_tma_bf16_update_param; m and v written back.
///
///   p_val                       the four parameters: written, the new ones
///   g_val                       their gradients
///   m_val, v_val                their moments: read, then written back
///   clip_val                    g is multiplied by it
///   decay_factor_val            p is multiplied by it
///   step_val                    this step's numbers
__device__ __forceinline__ void adamw_tma_bf16_update_four(float4& p_val, float4 g_val, float4& m_val, float4& v_val, float clip_val,
                                                           float decay_factor_val, const AdamwTmaBf16Step& step_val) {
    p_val.x = adamw_tma_bf16_update_param(p_val.x, g_val.x, &m_val.x, &v_val.x, clip_val, decay_factor_val, step_val);
    p_val.y = adamw_tma_bf16_update_param(p_val.y, g_val.y, &m_val.y, &v_val.y, clip_val, decay_factor_val, step_val);
    p_val.z = adamw_tma_bf16_update_param(p_val.z, g_val.z, &m_val.z, &v_val.z, clip_val, decay_factor_val, step_val);
    p_val.w = adamw_tma_bf16_update_param(p_val.w, g_val.w, &m_val.w, &v_val.w, clip_val, decay_factor_val, step_val);
}

/// Gives a pointer into shared memory as the 32-bit address the PTX instructions take.
///
///   shared_ptr   a pointer into the block's shared memory
__device__ __forceinline__ uint32_t adamw_tma_bf16_shared_address(const void* shared_ptr) {
    return (uint32_t)__cvta_generic_to_shared(shared_ptr);
}

/// Makes an mbarrier that completes a phase after arrivals_val arrivals (and the bytes it is told to expect).
///
///   barrier_ptr    the barrier, 8 bytes of shared memory
///   arrivals_val   arrivals a phase
__device__ __forceinline__ void adamw_tma_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(adamw_tma_bf16_shared_address(barrier_ptr)), "r"(arrivals_val) : "memory");
}

/// Arrives once at a barrier and tells it bytes_val bytes of TMA copies are coming.
///
///   barrier_ptr   the barrier
///   bytes_val     the bytes the copies asked for next bring
__device__ __forceinline__ void adamw_tma_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val) {
    // Its phase completes when they are in.
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(adamw_tma_bf16_shared_address(barrier_ptr)), "r"(bytes_val) : "memory");
}

/// Waits until a barrier's phase of this parity is complete.
///
///   barrier_ptr   the barrier
///   parity_val    the phase waited for, 0 or 1
__device__ __forceinline__ void adamw_tma_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val) {
    // The caller keeps count of the phases (the first is 0, the next 1, then 0 again).
    uint32_t barrier_address_val = adamw_tma_bf16_shared_address(barrier_ptr);
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

/// Asks TMA for bytes_val bytes of global memory, next to each other, into this block's shared memory, counted at a barrier.
///
///   shared_ptr    where they go: on 16 bytes
///   global_ptr    where they are: on 16 bytes
///   bytes_val     a multiple of 16
///   barrier_ptr   the barrier that expects the bytes
__device__ __forceinline__ void adamw_tma_bf16_copy_in(void* shared_ptr, const void* global_ptr, uint32_t bytes_val, uint64_t* barrier_ptr) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(adamw_tma_bf16_shared_address(shared_ptr)), "l"(global_ptr), "r"(bytes_val),
                   "r"(adamw_tma_bf16_shared_address(barrier_ptr))
                 : "memory");
}

/// Has TMA write bytes_val bytes of this block's shared memory into global memory, in the open bulk group.
///
///   global_ptr    where they go: on 16 bytes
///   shared_ptr    where they are: on 16 bytes
///   bytes_val     a multiple of 16
__device__ __forceinline__ void adamw_tma_bf16_copy_out(void* global_ptr, const void* shared_ptr, uint32_t bytes_val) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 ::"l"(global_ptr), "r"(adamw_tma_bf16_shared_address(shared_ptr)), "r"(bytes_val)
                 : "memory");
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Checks the counts and addresses; any of them wrong: stops with an error.
///
///   params_ptr ... master_params_ptr   as in adamw_tma_bf16_update
inline void adamw_tma_bf16_check(const nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, const nv_bfloat16* m_ptr,
                                 const nv_bfloat16* v_ptr, size_t params_in_slice_val, size_t decayed_count_val,
                                 const float* master_params_ptr) {
    // Counts: whole float4s, the decayed ones inside the slice, as adamw_update_bf16. Addresses: a bulk copy starts
    // on 16 bytes (a tile's offset is a whole number of 16-byte cells, so the arrays' starts must be too).
    if (params_in_slice_val % VALUES_PER_FLOAT4 != 0 || decayed_count_val % VALUES_PER_FLOAT4 != 0 || decayed_count_val > params_in_slice_val) {
        fprintf(stderr, "error: adamw_tma_bf16 needs params_in_slice and decayed_count to be multiples of %d, with decayed_count <= "
                        "params_in_slice; got params_in_slice = %zu, decayed_count = %zu\n",
                VALUES_PER_FLOAT4, params_in_slice_val, decayed_count_val);
        exit(EXIT_FAILURE);
    }
    const size_t align_val = HARDWARE_TMA_BYTES_PER_CELL;
    if ((size_t)params_ptr % align_val != 0 || (size_t)grads_ptr % align_val != 0 || (size_t)m_ptr % align_val != 0 ||
        (size_t)v_ptr % align_val != 0 || (size_t)master_params_ptr % align_val != 0) {
        fprintf(stderr, "error: adamw_tma_bf16 needs params, grads, m, v and master_params to start on %zu bytes (bulk copies); got "
                        "%p, %p, %p, %p, %p\n", align_val, (const void*)params_ptr, (const void*)grads_ptr, (const void*)m_ptr,
                (const void*)v_ptr, (const void*)master_params_ptr);
        exit(EXIT_FAILURE);
    }
}

/// Gives this step's numbers, worked out as adamw_update_bf16 does.
///
///   decayed_count_val ... t_val   as in adamw_tma_bf16_update
inline AdamwTmaBf16Step adamw_tma_bf16_step(size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val, float learning_rate_val,
                                            float beta1_val, float beta2_val, float eps_val, float weight_decay_val, int t_val) {
    // The same expressions in the same order, so the same floats: the bits match adamw_update_bf16's.
    // bias corrections: m and v start at 0, so early m and v are too small; dividing by 1 - beta^t undoes it
    float bias_correction1_val = 1.0f - powf(beta1_val, (float)t_val);
    float bias_correction2_val = 1.0f - powf(beta2_val, (float)t_val);
    AdamwTmaBf16Step step_val;
    step_val.decayed_count = decayed_count_val;
    step_val.grad_norm_ptr = grad_norm_ptr;
    step_val.max_norm = max_norm_val;
    step_val.weight_decay_factor = 1.0f - learning_rate_val * weight_decay_val;
    step_val.beta1 = beta1_val;
    step_val.beta2 = beta2_val;
    step_val.eps = eps_val;
    step_val.step_size = learning_rate_val / bias_correction1_val;
    step_val.inverse_sqrt_bias_correction2 = 1.0f / sqrtf(bias_correction2_val);
    return step_val;
}

/// Works out a launch's tile and its shared memory, and tells if it fits.
///
///   tile_ptr                       written: the tile
///   shared_bytes_ptr               written: the dynamic shared memory a block asks for
///   has_master_val                 a float master: the first region 4 bytes a value, else 2
///   tile_values_val, stages_val    the settings
///   params_in_slice_val            values to update
inline bool adamw_tma_bf16_plan(AdamwTmaBf16Tile* tile_ptr, size_t* shared_bytes_ptr, bool has_master_val, int tile_values_val,
                                int stages_val, size_t params_in_slice_val) {
    // Fits: tile_values a positive multiple of ADAMW_TMA_BF16_VALUES_PER_COPY (so every region, and every tile's
    // offset, is whole 16-byte cells), stages within ADAMW_TMA_BF16_MAX_STAGES, the sets within a block's shared memory.
    if (tile_values_val < ADAMW_TMA_BF16_VALUES_PER_COPY || tile_values_val % ADAMW_TMA_BF16_VALUES_PER_COPY != 0 || stages_val < 1 ||
        stages_val > ADAMW_TMA_BF16_MAX_STAGES)
        return false;
    int first_bytes_val = tile_values_val * (has_master_val ? (int)sizeof(float) : (int)sizeof(nv_bfloat16));
    int bf16_bytes_val = tile_values_val * (int)sizeof(nv_bfloat16);
    int set_bytes_val = first_bytes_val + 3 * bf16_bytes_val;
    size_t shared_bytes_val = (size_t)stages_val * set_bytes_val;
    if (shared_bytes_val > (size_t)ADAMW_TMA_BF16_MAX_SHARED_BYTES) return false;
    size_t copied_values_val = params_in_slice_val / ADAMW_TMA_BF16_VALUES_PER_COPY * ADAMW_TMA_BF16_VALUES_PER_COPY;
    *tile_ptr = AdamwTmaBf16Tile{tile_values_val, stages_val, first_bytes_val, bf16_bytes_val, set_bytes_val, copied_values_val,
                                 params_in_slice_val};
    *shared_bytes_ptr = shared_bytes_val;
    return true;
}

/// Gives the shared memory one block can ask for when blocks_per_sm_val blocks share an SM.
///
///   blocks_per_sm_val   blocks on one SM: 1 gives the most a block can have
inline int adamw_tma_bf16_share_bytes(int blocks_per_sm_val) {
    // The SM's shared memory over the blocks, less what CUDA keeps for each block (1 KB on an H100) and the barriers,
    // at most ADAMW_TMA_BF16_MAX_SHARED_BYTES, rounded down to ADAMW_TMA_BF16_SHARE_ROUND_BYTES.
    int sm_bytes_val = 0, reserved_bytes_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sm_bytes_val, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    cudaCheck(cudaDeviceGetAttribute(&reserved_bytes_val, cudaDevAttrReservedSharedMemoryPerBlock, 0));
    int share_val = sm_bytes_val / blocks_per_sm_val - reserved_bytes_val - ADAMW_TMA_BF16_MAX_STAGES * (int)sizeof(uint64_t);
    if (share_val > ADAMW_TMA_BF16_MAX_SHARED_BYTES) share_val = ADAMW_TMA_BF16_MAX_SHARED_BYTES;
    share_val = share_val / ADAMW_TMA_BF16_SHARE_ROUND_BYTES * ADAMW_TMA_BF16_SHARE_ROUND_BYTES;
    return share_val > 0 ? share_val : 0;
}

/// Gives the tile that cuts a share of shared memory into stages_val parts: values a tile, whole 16-byte cells.
///
///   share_bytes_val   the block's shared memory (adamw_tma_bf16_share_bytes)
///   stages_val        the parts: 1, the whole share one tile (the block waits for it, then works); more, the block
///                     works on one part while TMA fills the others
///   has_master_val    a float master: 10 bytes a value (4 + 2 + 2 + 2), else 8
inline int adamw_tma_bf16_tile_values(int share_bytes_val, int stages_val, bool has_master_val) {
    int value_bytes_val = (has_master_val ? (int)sizeof(float) : (int)sizeof(nv_bfloat16)) + 3 * (int)sizeof(nv_bfloat16);
    int tile_values_val = share_bytes_val / stages_val / value_bytes_val;
    return tile_values_val / ADAMW_TMA_BF16_VALUES_PER_COPY * ADAMW_TMA_BF16_VALUES_PER_COPY;
}

#endif // ADAMW_TMA_BF16_CUH
