// sgd_8bit.cuh — gradient clipping and one SGD step with momentum for a bfloat16 model, with the momentum in 8 bits (FP8 E4M3).
//
//     tensor          type in memory
//     params          bfloat16
//     grads           bfloat16
//     master params   float
//     momentum        FP8 E4M3, a float scale per warp

#ifndef SGD_8BIT_CUH
#define SGD_8BIT_CUH

#include <cuda_fp8.h>             // FP8 E4M3
#include "adamw_8bit.cuh"         // ADAMW_VALUES_PER_SCALE, ADAMW_8BIT_SMALLEST_KEPT: the momentum is kept as AdamW8bit keeps m

#define SGD_8BIT_THREADS_PER_BLOCK (4 * HARDWARE_THREADS_PER_WARP)   // the default block size: 4 warps, as AdamW8bit's
#define SGD_8BIT_BLOCKS 40960                                        // blocks in the launch, at most: past that each thread takes several float4s

/// Moves one parameter one SGD step and gives the new p; the momentum is written back.
///
///   p_val               the parameter
///   g_val               its gradient
///   m_ptr               its momentum: read, then written back
///   clip_val            g is multiplied by it: min(1, max_norm / norm)
///   weight_decay_val    weight_decay x p is added to g: weight_decay for a matrix, 0 otherwise
///   momentum_val        how much of the old momentum stays: 0.9
///   nesterov_val        true: step along g + momentum x m; false: along m
///   learning_rate_val   the learning rate of this step
__device__ __forceinline__ float sgd_8bit_update_param(float p_val, float g_val, float* m_ptr, float clip_val,
                                                       float weight_decay_val, float momentum_val, bool nesterov_val,
                                                       float learning_rate_val) {
    // PyTorch's SGD with dampening 0: the decay is L2, added to g (not AdamW's p x (1 - lr wd)); m = momentum x m + g,
    // which at step 1 (m = 0) is g, as PyTorch's buf = g.clone(); then p -= lr x (g + momentum x m) with Nesterov,
    // lr x m without.
    g_val = g_val * clip_val + weight_decay_val * p_val;
    *m_ptr = momentum_val * *m_ptr + g_val;
    float direction_val = nesterov_val ? g_val + momentum_val * *m_ptr : *m_ptr;
    return p_val - learning_rate_val * direction_val;
}

/// Does one SGD step on every value, keeping the momentum in E4M3 with one scale per warp.
///
///   params_ptr                bfloat16: written, the master rounded
///   grads_ptr                 bfloat16
///   m_ptr, m_scales_ptr       momentum / scale, 1 byte per value; one scale per ADAMW_VALUES_PER_SCALE values
///   params_in_slice_val       values to update: this GPU's slice (all of them on one GPU)
///   decayed_count_val         the first this many values get weight decay (the matrices)
///   grad_norm_ptr             norm of all the gradients
///   max_norm_val              clip to this norm; 0: no clipping
///   weight_decay_val          0 .. 5e-4
///   momentum_val              0.9; 0: plain SGD (the momentum is then this step's g)
///   nesterov_val              true: Nesterov's momentum
///   learning_rate_val         the learning rate of this step
///   master_params_ptr         float copy of the params; nullptr: read the bfloat16 params
__global__ void sgd_8bit_kernel(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, __nv_fp8_e4m3* m_ptr,
                                float* m_scales_ptr, size_t params_in_slice_val, size_t decayed_count_val,
                                const float* grad_norm_ptr, float max_norm_val, float weight_decay_val,
                                float momentum_val, bool nesterov_val, float learning_rate_val, float* master_params_ptr) {
    // adamw_8bit_kernel's loop with one moment: one float4 a thread, one warp = ADAMW_VALUES_PER_SCALE values = one
    // scale, so blockDim.x must be a multiple of HARDWARE_THREADS_PER_WARP.
    // Gradient clipping: scale g down so its norm is at most max_norm; 1 if already below it.
    float clip_val = 1.0f;
    if (max_norm_val > 0.0f) {
        clip_val = max_norm_val / (*grad_norm_ptr + 1e-6f);
        if (clip_val > 1.0f) clip_val = 1.0f;
    }
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;   // this thread's place in its warp
    size_t float4_count_val = params_in_slice_val / VALUES_PER_FLOAT4;
    // Loops while the warp's first float4 is in the slice, the same test for all its threads, so the warp stays
    // whole for warp_reduce_max. Each round jumps over the whole grid.
    for (size_t warp_first_index_val = thread_index_val - lane_index_val; warp_first_index_val < float4_count_val;
         warp_first_index_val += (size_t)gridDim.x * blockDim.x) {
        size_t index_val = warp_first_index_val + lane_index_val;   // this thread's float4
        bool in_slice_val = index_val < float4_count_val;           // false: no reads or writes, only the max
        size_t scale_index_val = warp_first_index_val * VALUES_PER_FLOAT4 / ADAMW_VALUES_PER_SCALE;   // one per warp
        float4 m_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // 1. un-zoom and update: last step's bytes x last step's scale = the momentum in float, then SGD
        if (in_slice_val) {
            // decay: matrices only
            float decay_val = index_val * VALUES_PER_FLOAT4 < decayed_count_val ? weight_decay_val : 0.0f;
            // params: float copy, else bfloat16
            float4 p_val = master_params_ptr != nullptr ? reinterpret_cast<float4*>(master_params_ptr)[index_val]
                                                        : read4_as_float(params_ptr + VALUES_PER_FLOAT4 * index_val);
            float4 g_val = read4_as_float(grads_ptr + VALUES_PER_FLOAT4 * index_val);
            // m = stored x scale
            float4 m_stored_val = (float4)reinterpret_cast<__nv_fp8x4_e4m3*>(m_ptr)[index_val];
            float m_scale_val = m_scales_ptr[scale_index_val];
            m_val = make_float4(m_stored_val.x * m_scale_val, m_stored_val.y * m_scale_val,
                                m_stored_val.z * m_scale_val, m_stored_val.w * m_scale_val);
            // SGD
            p_val.x = sgd_8bit_update_param(p_val.x, g_val.x, &m_val.x, clip_val, decay_val, momentum_val, nesterov_val, learning_rate_val);
            p_val.y = sgd_8bit_update_param(p_val.y, g_val.y, &m_val.y, clip_val, decay_val, momentum_val, nesterov_val, learning_rate_val);
            p_val.z = sgd_8bit_update_param(p_val.z, g_val.z, &m_val.z, clip_val, decay_val, momentum_val, nesterov_val, learning_rate_val);
            p_val.w = sgd_8bit_update_param(p_val.w, g_val.w, &m_val.w, clip_val, decay_val, momentum_val, nesterov_val, learning_rate_val);
            // write params: float copy, bfloat16 rounded
            if (master_params_ptr != nullptr) reinterpret_cast<float4*>(master_params_ptr)[index_val] = p_val;
            write4_from_float(params_ptr + VALUES_PER_FLOAT4 * index_val, p_val);
        }
        // 2. the largest |m| of the warp's ADAMW_VALUES_PER_SCALE values: each thread's own 4, then the warp's
        float m_largest_val = fmaxf(fmaxf(fabsf(m_val.x), fabsf(m_val.y)), fmaxf(fabsf(m_val.z), fabsf(m_val.w)));
        m_largest_val = warp_reduce_max(m_largest_val);
        // 3. zoom and store: x 448 / largest, rounded to the nearest E4M3; a group too small for a finite zoom: all 0s, scale 0
        if (m_largest_val < ADAMW_8BIT_SMALLEST_KEPT) m_largest_val = 0.0f;
        if (in_slice_val) {
            float m_to_stored_val = m_largest_val > 0.0f ? HARDWARE_E4M3_LARGEST / m_largest_val : 0.0f;
            reinterpret_cast<__nv_fp8x4_e4m3*>(m_ptr)[index_val] = __nv_fp8x4_e4m3(
                make_float4(m_val.x * m_to_stored_val, m_val.y * m_to_stored_val, m_val.z * m_to_stored_val,
                            m_val.w * m_to_stored_val));
        }
        // 4. the new scale, largest / 448, for the next step's un-zoom: one thread writes it
        if (lane_index_val == 0) m_scales_ptr[scale_index_val] = m_largest_val / HARDWARE_E4M3_LARGEST;
    }
}

/// One SGD step on all the parameters: checks the launch, counts and addresses, then launches sgd_8bit_kernel.
///
///   params_ptr, grads_ptr       bfloat16: the parameters (matrices first) and their gradients
///   m_ptr, m_scales_ptr         momentum / scale, 1 byte per value; one scale per ADAMW_VALUES_PER_SCALE values; all 0 at the start
///   params_in_slice_val         values to update: this GPU's slice
///   decayed_count_val           the first this many values get weight decay (the matrices)
///   grad_norm_ptr               norm of all the gradients
///   max_norm_val                clip to this norm; 0: no clipping
///   learning_rate_val           the learning rate of this step
///   momentum_val, nesterov_val  0.9, true
///   weight_decay_val            0 .. 5e-4, as L2: added to g
///   master_params_ptr           float copy of the params; nullptr: none
///   threads_per_block_val       the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                               SGD_8BIT_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void sgd_update_8bit(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, __nv_fp8_e4m3* m_ptr,
                            float* m_scales_ptr, size_t params_in_slice_val, size_t decayed_count_val,
                            const float* grad_norm_ptr, float max_norm_val, float learning_rate_val, float momentum_val,
                            bool nesterov_val, float weight_decay_val, float* master_params_ptr,
                            int threads_per_block_val = SGD_8BIT_THREADS_PER_BLOCK) {
    // 1. check the launch, counts and addresses, as adamw_update_8bit; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "SGD");
    if (params_in_slice_val % VALUES_PER_FLOAT4 != 0 || decayed_count_val % VALUES_PER_FLOAT4 != 0 || decayed_count_val > params_in_slice_val) {
        fprintf(stderr, "error: SGD needs params_in_slice and decayed_count to be multiples of %d (float4 loads), with "
                        "decayed_count <= params_in_slice; got params_in_slice = %zu, decayed_count = %zu\n",
                VALUES_PER_FLOAT4, params_in_slice_val, decayed_count_val);
        exit(EXIT_FAILURE);
    }
    size_t bfloat16_float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(nv_bfloat16);
    size_t fp8_float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(__nv_fp8_e4m3);
    size_t float_float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(float);
    if ((size_t)params_ptr % bfloat16_float4_bytes_val != 0 || (size_t)grads_ptr % bfloat16_float4_bytes_val != 0 ||
        (size_t)m_ptr % fp8_float4_bytes_val != 0 || (size_t)master_params_ptr % float_float4_bytes_val != 0) {
        fprintf(stderr, "error: SGD needs params and grads to start on %zu bytes, 8-bit momentum on %zu and "
                        "master_params on %zu (float4 loads); got %p, %p, %p, %p\n", bfloat16_float4_bytes_val,
                fp8_float4_bytes_val, float_float4_bytes_val, (void*)params_ptr, (const void*)grads_ptr, (void*)m_ptr,
                (void*)master_params_ptr);
        exit(EXIT_FAILURE);
    }
    // 2. launch: one float4 per thread, the blocks rounded up, but at most SGD_8BIT_BLOCKS; the kernel's warps
    //    then step over the rest
    size_t one_float4_per_thread_blocks_val = CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, (size_t)threads_per_block_val);
    int blocks_val = (int)(one_float4_per_thread_blocks_val < SGD_8BIT_BLOCKS ? one_float4_per_thread_blocks_val : SGD_8BIT_BLOCKS);
    sgd_8bit_kernel<<<blocks_val, threads_per_block_val>>>(
        params_ptr, grads_ptr, m_ptr, m_scales_ptr, params_in_slice_val, decayed_count_val, grad_norm_ptr, max_norm_val,
        weight_decay_val, momentum_val, nesterov_val, learning_rate_val, master_params_ptr);
    cudaCheck(cudaGetLastError());
}

#endif // SGD_8BIT_CUH
