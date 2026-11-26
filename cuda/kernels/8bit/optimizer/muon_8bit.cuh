// muon_8bit.cuh — one Muon step for a bfloat16 model's matrices, with the momentum in 8 bits (FP8 E4M3).
//
//     tensor                    type in memory                     shape
//     param                     bfloat16                           {rows, columns}
//     grad                      bfloat16                           {rows, columns}
//     master param              float                              {rows, columns}
//     momentum                  FP8 E4M3, a float scale per warp   {rows · columns}
//     direction, product        bfloat16                           {rows, columns}
//     gram, gram_squared        bfloat16                           {short, short}, short = min(rows, columns)
//     norm, norm_block_sums     float                              {1}, {2, GRAD_NORM_BF16_BLOCKS}

#ifndef MUON_8BIT_CUH
#define MUON_8BIT_CUH

#include <algorithm>
#include <cmath>
#include "adamw_8bit.cuh"                     // ADAMW_VALUES_PER_SCALE, ADAMW_8BIT_SMALLEST_KEPT: the momentum is kept as AdamW8bit keeps m
#include "../../bf16/grad_norm_bf16.cuh"      // GradNormBf16: the direction's norm, the same bits every run
#include "../../bf16/wgmma_matmul_bf16.cuh"   // Newton–Schulz's products

#define MUON_8BIT_THREADS_PER_BLOCK (4 * HARDWARE_THREADS_PER_WARP)   // AdamW8bit's: a warp is one scale group
#define MUON_8BIT_BLOCKS 40960                                        // blocks in a launch, at most: past that each thread takes several float4s
#define MUON_NEWTON_SCHULZ_A 3.4445f          // the quintic's x, x³ and x⁵ factors (Keller Jordan's Muon): each step pulls every
#define MUON_NEWTON_SCHULZ_B (-4.7750f)       // singular value of the normalized direction toward about 0.7 .. 1.2
#define MUON_NEWTON_SCHULZ_C 2.0315f
#define MUON_NORM_EPS 1e-7f                   // keeps an all-0 direction from dividing by 0
#define MUON_MATCH_ADAMW_RMS 0.2f             // Moonshot's scale: 0.2 · sqrt(max(rows, columns)) gives Muon's update AdamW's RMS

/// Muon's settings for the blocks' matrices; the other params keep AdamW's.
struct MuonConfig {
    float learning_rate = 0.02f;              // at the schedule's peak: a step's is this x AdamW's learning rate / adamw_peak_learning_rate
    float adamw_peak_learning_rate = 6e-4f;   // the schedule's peak for AdamW, for the ratio above
    float momentum = 0.95f;                   // how much of the old momentum stays
    bool nesterov = true;                     // orthogonalize (1 - momentum) g + momentum m, not m
    float weight_decay = 0.0f;                // decoupled, as AdamW's: p x (1 - learning_rate x weight_decay)
    int newton_schulz_steps = 5;
    bool match_adamw = false;                 // false: the step x sqrt(max(1, rows / columns)) (Keller Jordan's, nanochat's);
                                              // true: x 0.2 sqrt(max(rows, columns)) (Moonshot's, AdamW's learning rates)
};

/// The scratch a Muon step works in: made once for the largest matrix, shared by all of them.
struct Muon8bitScratch {
    Tensor direction_tensor;                  // {largest count} bfloat16: the momentum's direction, then Newton–Schulz's X
    Tensor product_tensor;                    // {largest count} bfloat16: (b gram + c gram²) times X, each step
    Tensor gram_tensor;                       // {largest short²} bfloat16: X Xᵀ (or Xᵀ X)
    Tensor gram_squared_tensor;               // {largest short²} bfloat16: the gram times itself, then b gram + c gram²
    Tensor norm_tensor;                       // {1} float: the direction's norm
    Tensor norm_block_sums_tensor;            // {2, GRAD_NORM_BF16_BLOCKS} float: GradNormBf16's

    Muon8bitScratch() = default;
    /// Makes it, all 0.
    ///
    ///   largest_count_val   the most values a matrix has
    ///   largest_short_val   the largest min(rows, columns)
    Muon8bitScratch(size_t largest_count_val, size_t largest_short_val)
        : direction_tensor(Tensor::zeros({largest_count_val}, DType::BFloat16)),
          product_tensor(Tensor::zeros({largest_count_val}, DType::BFloat16)),
          gram_tensor(Tensor::zeros({largest_short_val * largest_short_val}, DType::BFloat16)),
          gram_squared_tensor(Tensor::zeros({largest_short_val * largest_short_val}, DType::BFloat16)),
          norm_tensor(Tensor::zeros({1})),
          norm_block_sums_tensor(Tensor::zeros({2, GRAD_NORM_BF16_BLOCKS})) {}
};

// ── the kernels ───────────────────────────────────────────────────────────────

/// Moves one value's momentum one step and gives the direction Muon orthogonalizes.
///
///   m_val          the old momentum
///   g_val          the gradient, clipped
///   momentum_val   how much of the old momentum stays: 0.95
///   nesterov_val   true: Nesterov's direction
///   m_ptr          the new momentum: written
__device__ __forceinline__ float muon_8bit_momentum_value(float m_val, float g_val, float momentum_val, bool nesterov_val,
                                                          float* m_ptr) {
    // m = momentum m + (1 - momentum) g (Keller Jordan's buf.lerp_(g, 1 - momentum)); the direction is
    // (1 - momentum) g + momentum m with Nesterov (g.lerp_(buf, momentum)), else m itself.
    float new_m_val = momentum_val * m_val + (1.0f - momentum_val) * g_val;
    *m_ptr = new_m_val;
    return nesterov_val ? (1.0f - momentum_val) * g_val + momentum_val * new_m_val : new_m_val;
}

/// Moves one matrix's 8-bit momentum one step and writes the direction to orthogonalize in bfloat16.
///
///   direction_ptr        {count} bfloat16: written
///   grads_ptr            {count} bfloat16
///   m_ptr, m_scales_ptr  momentum / scale, 1 byte a value; one scale per ADAMW_VALUES_PER_SCALE values, the matrix's own
///   count_val            the matrix's values: a multiple of ADAMW_VALUES_PER_SCALE
///   grad_norm_ptr        norm of all the gradients
///   max_norm_val         clip to this norm; 0: no clipping
///   momentum_val         0.95
///   nesterov_val         true: Nesterov's direction
__global__ void muon_8bit_momentum_kernel(nv_bfloat16* direction_ptr, const nv_bfloat16* grads_ptr, __nv_fp8_e4m3* m_ptr,
                                          float* m_scales_ptr, size_t count_val, const float* grad_norm_ptr,
                                          float max_norm_val, float momentum_val, bool nesterov_val) {
    // One float4 a thread, as adamw_8bit_kernel: un-zoom m with last step's scale, move it, zoom it back with the
    // warp's new largest. A warp is one scale group, so blockDim.x must be whole warps.
    // Gradient clipping: scale g down so its norm is at most max_norm; 1 if already below it.
    float clip_val = 1.0f;
    if (max_norm_val > 0.0f) {
        clip_val = max_norm_val / (*grad_norm_ptr + 1e-6f);
        if (clip_val > 1.0f) clip_val = 1.0f;
    }
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;   // this thread's place in its warp
    size_t float4_count_val = count_val / VALUES_PER_FLOAT4;
    // Loops while the warp's first float4 is in the matrix, the same test for its threads, so the warp stays whole
    // for warp_reduce_max. Each round jumps over the whole grid.
    for (size_t warp_first_index_val = thread_index_val - lane_index_val; warp_first_index_val < float4_count_val;
         warp_first_index_val += (size_t)gridDim.x * blockDim.x) {
        size_t index_val = warp_first_index_val + lane_index_val;   // this thread's float4
        bool in_matrix_val = index_val < float4_count_val;          // false: no reads or writes, only the max
        size_t scale_index_val = warp_first_index_val * VALUES_PER_FLOAT4 / ADAMW_VALUES_PER_SCALE;   // one per warp
        float4 m_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // 1. un-zoom and move: stored byte x scale = m in float; then the new m and the direction
        if (in_matrix_val) {
            float4 g_val = read4_as_float(grads_ptr + VALUES_PER_FLOAT4 * index_val);
            float4 m_stored_val = (float4)reinterpret_cast<__nv_fp8x4_e4m3*>(m_ptr)[index_val];
            float m_scale_val = m_scales_ptr[scale_index_val];
            float4 direction_val;
            direction_val.x = muon_8bit_momentum_value(m_stored_val.x * m_scale_val, g_val.x * clip_val, momentum_val, nesterov_val, &m_val.x);
            direction_val.y = muon_8bit_momentum_value(m_stored_val.y * m_scale_val, g_val.y * clip_val, momentum_val, nesterov_val, &m_val.y);
            direction_val.z = muon_8bit_momentum_value(m_stored_val.z * m_scale_val, g_val.z * clip_val, momentum_val, nesterov_val, &m_val.z);
            direction_val.w = muon_8bit_momentum_value(m_stored_val.w * m_scale_val, g_val.w * clip_val, momentum_val, nesterov_val, &m_val.w);
            write4_from_float(direction_ptr + VALUES_PER_FLOAT4 * index_val, direction_val);
        }
        // 2. the warp's largest |m|; a group too small for a finite zoom is stored as 0s, scale 0 (AdamW8bit's NaN fix)
        float m_largest_val = fmaxf(fmaxf(fabsf(m_val.x), fabsf(m_val.y)), fmaxf(fabsf(m_val.z), fabsf(m_val.w)));
        m_largest_val = warp_reduce_max(m_largest_val);
        if (m_largest_val < ADAMW_8BIT_SMALLEST_KEPT) m_largest_val = 0.0f;
        // 3. zoom and store: x 448 / largest, rounded to the nearest E4M3
        if (in_matrix_val) {
            float m_to_stored_val = m_largest_val > 0.0f ? HARDWARE_E4M3_LARGEST / m_largest_val : 0.0f;
            reinterpret_cast<__nv_fp8x4_e4m3*>(m_ptr)[index_val] = __nv_fp8x4_e4m3(
                make_float4(m_val.x * m_to_stored_val, m_val.y * m_to_stored_val, m_val.z * m_to_stored_val,
                            m_val.w * m_to_stored_val));
        }
        // 4. the new scale, largest / 448: one thread writes it
        if (lane_index_val == 0) m_scales_ptr[scale_index_val] = m_largest_val / HARDWARE_E4M3_LARGEST;
    }
}

/// Writes output = first x first_factor + second x second_factor, each value, the factors divided by a norm if given.
///
///   output_ptr          {count} bfloat16: written; may be first_ptr or second_ptr
///   first_ptr           {count} bfloat16
///   first_factor_val    first's factor
///   second_ptr          {count} bfloat16, or nullptr for none
///   second_factor_val   second's factor
///   count_val           a multiple of VALUES_PER_FLOAT4
///   divide_by_ptr       1 float: both factors divided by it + MUON_NORM_EPS; nullptr: not divided
__global__ void muon_8bit_mix_kernel(nv_bfloat16* output_ptr, const nv_bfloat16* first_ptr, float first_factor_val,
                                     const nv_bfloat16* second_ptr, float second_factor_val, size_t count_val,
                                     const float* divide_by_ptr) {
    // One float4 a thread, the sums in float, rounded once to bfloat16. Each value is read before it is written by
    // the same thread, so the output may be one of the inputs.
    float divide_val = divide_by_ptr != nullptr ? 1.0f / (*divide_by_ptr + MUON_NORM_EPS) : 1.0f;
    float first_times_val = first_factor_val * divide_val;
    float second_times_val = second_factor_val * divide_val;
    size_t float4_count_val = count_val / VALUES_PER_FLOAT4;
    for (size_t index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x; index_val < float4_count_val;
         index_val += (size_t)gridDim.x * blockDim.x) {
        float4 first_val = read4_as_float(first_ptr + VALUES_PER_FLOAT4 * index_val);
        float4 output_val = make_float4(first_val.x * first_times_val, first_val.y * first_times_val,
                                        first_val.z * first_times_val, first_val.w * first_times_val);
        if (second_ptr != nullptr) {
            float4 second_val = read4_as_float(second_ptr + VALUES_PER_FLOAT4 * index_val);
            output_val.x += second_val.x * second_times_val;
            output_val.y += second_val.y * second_times_val;
            output_val.z += second_val.z * second_times_val;
            output_val.w += second_val.w * second_times_val;
        }
        write4_from_float(output_ptr + VALUES_PER_FLOAT4 * index_val, output_val);
    }
}

/// Moves one matrix's master by the orthogonalized direction and rounds it into the bfloat16 param.
///
///   params_ptr          {count} bfloat16: written, the master rounded
///   master_params_ptr   {count} float: read and written
///   orthogonal_ptr      {count} bfloat16: Newton–Schulz's output
///   count_val           a multiple of VALUES_PER_FLOAT4
///   decay_factor_val    p is multiplied by it first: 1 - learning_rate x weight_decay
///   step_val            learning_rate x the shape's scale
__global__ void muon_8bit_apply_kernel(nv_bfloat16* params_ptr, float* master_params_ptr, const nv_bfloat16* orthogonal_ptr,
                                       size_t count_val, float decay_factor_val, float step_val) {
    // p = p x decay_factor - step x O, one float4 a thread.
    size_t float4_count_val = count_val / VALUES_PER_FLOAT4;
    for (size_t index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x; index_val < float4_count_val;
         index_val += (size_t)gridDim.x * blockDim.x) {
        float4 p_val = reinterpret_cast<float4*>(master_params_ptr)[index_val];
        float4 o_val = read4_as_float(orthogonal_ptr + VALUES_PER_FLOAT4 * index_val);
        p_val.x = p_val.x * decay_factor_val - step_val * o_val.x;
        p_val.y = p_val.y * decay_factor_val - step_val * o_val.y;
        p_val.z = p_val.z * decay_factor_val - step_val * o_val.z;
        p_val.w = p_val.w * decay_factor_val - step_val * o_val.w;
        reinterpret_cast<float4*>(master_params_ptr)[index_val] = p_val;
        write4_from_float(params_ptr + VALUES_PER_FLOAT4 * index_val, p_val);
    }
}

// ── the launches ──────────────────────────────────────────────────────────────

/// Gives the blocks of an elementwise launch: one float4 a thread, at most MUON_8BIT_BLOCKS.
///
///   count_val               the values
///   threads_per_block_val   the block size
inline int muon_8bit_blocks(size_t count_val, int threads_per_block_val) {
    size_t one_float4_per_thread_blocks_val = CEIL_DIV(count_val / VALUES_PER_FLOAT4, (size_t)threads_per_block_val);
    return (int)std::min(one_float4_per_thread_blocks_val, (size_t)MUON_8BIT_BLOCKS);
}

/// Writes output = first x first_factor + second x second_factor (second may be Tensor() for none).
///
///   output                  bfloat16: written; may be first or second
///   first, second           bfloat16, output's count
///   first_factor_val        first's factor
///   second_factor_val       second's factor
///   divide_by               {1} float: both factors divided by it; Tensor(): not divided
///   threads_per_block_val   the block size
inline void muon_8bit_mix(const Tensor& output, const Tensor& first, float first_factor_val, const Tensor& second,
                          float second_factor_val, const Tensor& divide_by = Tensor(),
                          int threads_per_block_val = MUON_8BIT_THREADS_PER_BLOCK) {
    size_t count_val = output.numel();
    muon_8bit_mix_kernel<<<muon_8bit_blocks(count_val, threads_per_block_val), threads_per_block_val>>>(
        output.data_ptr<nv_bfloat16>(), first.data_ptr<nv_bfloat16>(), first_factor_val,
        second.data_ptr() != nullptr ? second.data_ptr<nv_bfloat16>() : nullptr, second_factor_val, count_val,
        divide_by.data_ptr() != nullptr ? divide_by.data() : nullptr);
    cudaCheck(cudaGetLastError());
}

/// Turns a normalized direction into about its nearest orthogonal matrix: Newton–Schulz's quintic, in bfloat16.
///
///   direction     {rows, columns} bfloat16, norm at most 1: written over with the result
///   scratch_val   its product, gram and gram_squared used
///   steps_val     the quintic's steps: 5
inline void muon_8bit_newton_schulz(const Tensor& direction, const Muon8bitScratch& scratch_val, int steps_val) {
    // Each step, on X = direction (Keller Jordan's zeropower_via_newtonschulz5):
    //
    //     wide (rows <= columns)        tall (rows > columns)
    //     A = X Xᵀ    {rows, rows}      A = Xᵀ X    {columns, columns}
    //     B = b A + c A A               B = b A + c A A
    //     X = a X + B X                 X = a X + X B
    //
    // The gram is the short side's, as Keller's code turns a tall matrix round first; here the products read X
    // turned instead (wgmma_matmul_bf16's turned flags), so nothing is copied. A A is written A Aᵀ and X B is
    // X Bᵀ: equal, A and B are symmetric (but for bfloat16's last bit).
    size_t rows_val = direction.size(0);
    size_t columns_val = direction.size(1);
    bool wide_val = rows_val <= columns_val;
    size_t short_val = wide_val ? rows_val : columns_val;
    Tensor gram = scratch_val.gram_tensor.narrow(0, 0, short_val * short_val).view({short_val, short_val});
    Tensor gram_squared = scratch_val.gram_squared_tensor.narrow(0, 0, short_val * short_val).view({short_val, short_val});
    Tensor product = scratch_val.product_tensor.narrow(0, 0, rows_val * columns_val).view({rows_val, columns_val});
    for (int step_index_val = 0; step_index_val < steps_val; step_index_val++) {
        // A: X Xᵀ (left X, right X); or Xᵀ X (both read turned: X holds {sum, rows} and {sum, columns})
        if (wide_val) wgmma_matmul_bf16(gram, direction, direction);
        else wgmma_matmul_bf16(gram, direction, direction, Tensor(), false, true, true);
        // B = b A + c A Aᵀ, into gram_squared
        wgmma_matmul_bf16(gram_squared, gram, gram);
        muon_8bit_mix(gram_squared, gram, MUON_NEWTON_SCHULZ_B, gram_squared, MUON_NEWTON_SCHULZ_C);
        // B X (right X read turned: {sum = rows, columns}); or X Bᵀ
        if (wide_val) wgmma_matmul_bf16(product, gram_squared, direction, Tensor(), false, false, true);
        else wgmma_matmul_bf16(product, direction, gram_squared);
        // X = a X + that
        muon_8bit_mix(direction, direction, MUON_NEWTON_SCHULZ_A, product, 1.0f);
    }
}

/// Gives the factor a matrix's orthogonal step is multiplied by, for its shape.
///
///   rows_val, columns_val   the matrix's {out, in}
///   match_adamw_val         MuonConfig::match_adamw
inline float muon_8bit_shape_scale(size_t rows_val, size_t columns_val, bool match_adamw_val) {
    // An orthogonal {rows, columns} matrix has RMS 1 / sqrt(max(rows, columns)). Keller's sqrt(max(1, rows / columns))
    // gives a tall matrix a larger step; Moonshot's 0.2 sqrt(max(rows, columns)) gives every matrix RMS 0.2, about
    // AdamW's, so AdamW's learning rate and weight decay carry over.
    if (match_adamw_val) return MUON_MATCH_ADAMW_RMS * sqrtf((float)std::max(rows_val, columns_val));
    return sqrtf(std::max(1.0f, (float)rows_val / (float)columns_val));
}

/// Does one Muon step on one matrix, its momentum in 8 bits: momentum, normalize, Newton–Schulz, then the step.
///
///   param                   {rows, columns} bfloat16: written, the master rounded
///   grad                    {rows, columns} bfloat16
///   master_params_ptr       its float copy, rows x columns values: read and written
///   m_ptr, m_scales_ptr     its momentum / scale, 1 byte a value; one scale per ADAMW_VALUES_PER_SCALE values; all 0 at the start
///   scratch_val             shared by all the matrices: at least this matrix's sizes
///   grad_norm_ptr           norm of all the gradients
///   max_norm_val            clip to this norm; 0: no clipping
///   learning_rate_val       this step's Muon learning rate
///   config_val              momentum, Nesterov, weight decay, the Newton–Schulz steps, the shape's scale
///   threads_per_block_val   the elementwise kernels' block size: whole warps; not given: MUON_8BIT_THREADS_PER_BLOCK
inline void muon_update_8bit(const Tensor& param, const Tensor& grad, float* master_params_ptr, __nv_fp8_e4m3* m_ptr,
                             float* m_scales_ptr, const Muon8bitScratch& scratch_val, const float* grad_norm_ptr,
                             float max_norm_val, float learning_rate_val, const MuonConfig& config_val,
                             int threads_per_block_val = MUON_8BIT_THREADS_PER_BLOCK) {
    // 1. check: a bfloat16 matrix with its gradient; its values whole scale groups (each matrix its own scales);
    //    the scratch big enough; the starts on whole float4s
    check_threads_per_block(threads_per_block_val, "Muon");
    if (param.dim() != 2 || grad.shape() != param.shape() || param.dtype() != DType::BFloat16 ||
        grad.dtype() != DType::BFloat16) {
        fprintf(stderr, "error: Muon needs a bfloat16 matrix and its bfloat16 gradient of the same shape\n");
        exit(EXIT_FAILURE);
    }
    size_t rows_val = param.size(0);
    size_t columns_val = param.size(1);
    size_t count_val = rows_val * columns_val;
    size_t short_val = std::min(rows_val, columns_val);
    if (count_val % ADAMW_VALUES_PER_SCALE != 0) {
        fprintf(stderr, "error: Muon needs a matrix of a multiple of %d values (its momentum's scale groups); got {%zu, %zu}\n",
                ADAMW_VALUES_PER_SCALE, rows_val, columns_val);
        exit(EXIT_FAILURE);
    }
    if (count_val > scratch_val.direction_tensor.numel() || short_val * short_val > scratch_val.gram_tensor.numel()) {
        fprintf(stderr, "error: Muon's scratch is smaller than the matrix {%zu, %zu}\n", rows_val, columns_val);
        exit(EXIT_FAILURE);
    }
    if ((size_t)master_params_ptr % (VALUES_PER_FLOAT4 * sizeof(float)) != 0 ||
        (size_t)m_ptr % (VALUES_PER_FLOAT4 * sizeof(__nv_fp8_e4m3)) != 0) {
        fprintf(stderr, "error: Muon needs master_params on %zu bytes and the momentum on %zu (float4 loads); got %p, %p\n",
                VALUES_PER_FLOAT4 * sizeof(float), VALUES_PER_FLOAT4 * sizeof(__nv_fp8_e4m3), (void*)master_params_ptr,
                (void*)m_ptr);
        exit(EXIT_FAILURE);
    }
    Tensor direction = scratch_val.direction_tensor.narrow(0, 0, count_val).view({rows_val, columns_val});
    int blocks_val = muon_8bit_blocks(count_val, threads_per_block_val);

    // 2. the momentum one step, and the direction to orthogonalize
    muon_8bit_momentum_kernel<<<blocks_val, threads_per_block_val>>>(
        direction.data_ptr<nv_bfloat16>(), grad.data_ptr<nv_bfloat16>(), m_ptr, m_scales_ptr, count_val, grad_norm_ptr,
        max_norm_val, config_val.momentum, config_val.nesterov);
    cudaCheck(cudaGetLastError());

    // 3. the direction / its norm, so every singular value is at most 1: Newton–Schulz's quintic needs it
    GradNormBf16(scratch_val.norm_block_sums_tensor).compute(direction, scratch_val.norm_tensor);
    muon_8bit_mix(direction, direction, 1.0f, Tensor(), 0.0f, scratch_val.norm_tensor, threads_per_block_val);

    // 4. about the nearest orthogonal matrix
    muon_8bit_newton_schulz(direction, scratch_val, config_val.newton_schulz_steps);

    // 5. the step: p x (1 - learning_rate x weight_decay) - learning_rate x the shape's scale x O
    float decay_factor_val = 1.0f - learning_rate_val * config_val.weight_decay;
    float step_val = learning_rate_val * muon_8bit_shape_scale(rows_val, columns_val, config_val.match_adamw);
    muon_8bit_apply_kernel<<<blocks_val, threads_per_block_val>>>(param.data_ptr<nv_bfloat16>(), master_params_ptr,
                                                                  direction.data_ptr<nv_bfloat16>(), count_val,
                                                                  decay_factor_val, step_val);
    cudaCheck(cudaGetLastError());
}

#endif // MUON_8BIT_CUH
