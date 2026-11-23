// test_adamw_tma_bf16.cu — adamw_tma_bf16_update gives the same bits as adamw_update_bf16 (master, params, m, v), at the default and at the edge settings.
#include <torch/types.h>              // torch::randn, torch::equal; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <algorithm>                  // std::count
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../../kernels/bf16/adamw_bf16.cuh"
#include "../../kernels/bf16/adamw_tma_bf16.cuh"

#define GPT2_PARAMS 124439808         // GPT-2 small's values
#define GPT2_DECAYED_PARAMS 124439040 // of them in matrices
#define SMALL_PARAMS 1000004          // a slice that ends 4 values past a 16-byte cell: the values TMA cannot move
#define SMALL_DECAYED_PARAMS 999000
#define LEARNING_RATE 6e-4f
#define BETA1 0.9f
#define BETA2 0.95f
#define EPS 1e-8f
#define WEIGHT_DECAY 0.1f
#define MAX_NORM 1.0f
#define STEP 3                        // a step past the first: m and v not 0, the bias corrections not the first's

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a float tensor with random values, mean 0 and standard deviation scale_val (made positive if asked), on the GPU.
///
///   count_val      its values
///   scale_val      the standard deviation
///   positive_val   true: the values' sizes, for v
static Tensor random_float_tensor(size_t count_val, float scale_val, bool positive_val = false)
{
    Tensor tensor = Tensor::empty({count_val});
    torch::Tensor values_tensor = torch::randn({(int64_t)count_val}, torch::TensorOptions().device(torch::kCUDA)) * scale_val;
    if (positive_val)
        values_tensor = values_tensor.abs();
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), count_val * sizeof(float), cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Tells if two tensors hold the same bits.
///
///   first_tensor, second_tensor   the same dtype and shape, on the GPU
static bool same_bits(const Tensor &first_tensor, const Tensor &second_tensor)
{
    // Each read as ints of its value's width, with no copy into floats: a NaN equals the same NaN.
    bool floats_val = first_tensor.dtype() == DType::Float32;
    torch::TensorOptions ints_val = torch::TensorOptions().dtype(floats_val ? torch::kInt32 : torch::kInt16).device(torch::kCUDA);
    torch::Tensor first_val = torch::from_blob(first_tensor.data_ptr(), {(int64_t)first_tensor.numel()}, ints_val);
    torch::Tensor second_val = torch::from_blob(second_tensor.data_ptr(), {(int64_t)second_tensor.numel()}, ints_val);
    return torch::equal(first_val, second_val);
}

/// AdamW's tensors at one size, their start, and adamw_update_bf16's step on them.
struct AdamwCase
{
    size_t params_in_slice, decayed_count;
    bool has_master;
    Tensor start_master_tensor, start_params_tensor, start_m_tensor, start_v_tensor;      // the step starts from these
    Tensor expected_master_tensor, expected_params_tensor, expected_m_tensor, expected_v_tensor;
    Tensor master_tensor, params_tensor, grads_tensor, m_tensor, v_tensor, grad_norm_tensor;

    /// Makes the tensors and the expected result.
    ///
    ///   params_in_slice_val, decayed_count_val   the counts
    ///   has_master_val                           a float master: as training runs it
    AdamwCase(size_t params_in_slice_val, size_t decayed_count_val, bool has_master_val)
        : params_in_slice(params_in_slice_val), decayed_count(decayed_count_val), has_master(has_master_val)
    {
        // 1. a random master and its bfloat16 params, grads (their norm clipped), m, and v (positive)
        torch::manual_seed(0);
        start_master_tensor = random_float_tensor(params_in_slice, 0.02f);
        start_params_tensor = start_master_tensor.to(DType::BFloat16);
        Tensor grads_float_tensor = random_float_tensor(params_in_slice, 1.0f);
        grads_tensor = grads_float_tensor.to(DType::BFloat16);
        start_m_tensor = random_float_tensor(params_in_slice, 0.1f).to(DType::BFloat16);
        start_v_tensor = random_float_tensor(params_in_slice, 0.01f, true).to(DType::BFloat16);
        float norm_val = torch::from_blob(grads_float_tensor.data(), {(int64_t)params_in_slice},
                                          torch::TensorOptions().device(torch::kCUDA)).norm().item<float>();
        grad_norm_tensor = Tensor::empty({1});
        grad_norm_tensor.copy_from_cpu(&norm_val);
        master_tensor = start_master_tensor.clone();
        params_tensor = start_params_tensor.clone();
        m_tensor = start_m_tensor.clone();
        v_tensor = start_v_tensor.clone();
        // 2. adamw_update_bf16's step, kept
        adamw_update_bf16(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), m_tensor.data_ptr<nv_bfloat16>(),
                          v_tensor.data_ptr<nv_bfloat16>(), params_in_slice, decayed_count, grad_norm_tensor.data(), MAX_NORM,
                          LEARNING_RATE, BETA1, BETA2, EPS, WEIGHT_DECAY, STEP, master());
        expected_master_tensor = master_tensor.clone();
        expected_params_tensor = params_tensor.clone();
        expected_m_tensor = m_tensor.clone();
        expected_v_tensor = v_tensor.clone();
        reset();
    }

    /// Gives the master to pass: nullptr with none.
    float *master(void) { return has_master ? master_tensor.data() : nullptr; }

    /// Puts the master, params, m and v back to the start.
    void reset(void)
    {
        master_tensor.copy_(start_master_tensor);
        params_tensor.copy_(start_params_tensor);
        m_tensor.copy_(start_m_tensor);
        v_tensor.copy_(start_v_tensor);
    }

    /// Tells if the step gave adamw_update_bf16's bits: the master (if any), params, m and v.
    bool same(void)
    {
        cudaCheck(cudaDeviceSynchronize());
        return (!has_master || same_bits(master_tensor, expected_master_tensor)) && same_bits(params_tensor, expected_params_tensor) &&
               same_bits(m_tensor, expected_m_tensor) && same_bits(v_tensor, expected_v_tensor);
    }

    /// Runs adamw_tma_bf16_update_at at one setting from the start, and gives false if it does not fit.
    ///
    ///   blocks_val, tile_values_val, stages_val   the runtime settings
    template <int VALUES, int WARPS, bool TMA_STORE>
    bool run_at(int blocks_val, int tile_values_val, int stages_val)
    {
        reset();
        return adamw_tma_bf16_update_at<VALUES, WARPS, TMA_STORE>(
            params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), m_tensor.data_ptr<nv_bfloat16>(),
            v_tensor.data_ptr<nv_bfloat16>(), params_in_slice, decayed_count, grad_norm_tensor.data(), MAX_NORM, LEARNING_RATE,
            BETA1, BETA2, EPS, WEIGHT_DECAY, STEP, master(), blocks_val, tile_values_val, stages_val);
    }
};

// ── tests ─────────────────────────────────────────────────────────────────────

static void the_default_settings_give_adamw_bf16s_bits(void)
{
    // At GPT-2 small's count with a master (as training runs it) and with none, and at a count whose last 4 values
    // TMA cannot move.
    std::vector<bool> same_list;
    for (bool has_master_val : {true, false})
        for (size_t params_val : {(size_t)GPT2_PARAMS, (size_t)SMALL_PARAMS})
        {
            AdamwCase case_val(params_val, params_val == GPT2_PARAMS ? GPT2_DECAYED_PARAMS : SMALL_DECAYED_PARAMS, has_master_val);
            adamw_tma_bf16_update(case_val.params_tensor.data_ptr<nv_bfloat16>(), case_val.grads_tensor.data_ptr<nv_bfloat16>(),
                                  case_val.m_tensor.data_ptr<nv_bfloat16>(), case_val.v_tensor.data_ptr<nv_bfloat16>(), case_val.params_in_slice,
                                  case_val.decayed_count, case_val.grad_norm_tensor.data(), MAX_NORM, LEARNING_RATE, BETA1, BETA2, EPS,
                                  WEIGHT_DECAY, STEP, case_val.master());
            same_list.push_back(case_val.same());
        }
    print_test_name(__func__);
    ASSERT_EQ((long)std::count(same_list.begin(), same_list.end(), true), (long)same_list.size());
}

static void the_edge_settings_give_adamw_bf16s_bits(void)
{
    // SMALL_PARAMS, with a master and with none, at the sweep's ends: one float4 a thread and 8, one warp and 32, TMA
    // storing and the threads, 96 and 768 threads a block, the smallest tile (8 values) and large ones, one set and the most, few blocks (each
    // walks many tiles) and many (some with none).
    std::vector<bool> fits_list;
    std::vector<bool> same_list;
    for (bool has_master_val : {true, false})
    {
        AdamwCase case_val(SMALL_PARAMS, SMALL_DECAYED_PARAMS, has_master_val);
        auto check_val = [&](bool fits_val) {
            fits_list.push_back(fits_val);
            same_list.push_back(fits_val && case_val.same());
        };
        check_val(case_val.run_at<4, 1, true>(7, 8, 1));
        check_val(case_val.run_at<4, 1, false>(7, 8, 1));
        check_val(case_val.run_at<32, 32, true>(1000, 8192, 2));
        check_val(case_val.run_at<32, 32, false>(100000, 8192, 2));
        check_val(case_val.run_at<8, 4, true>(132, 2048, 8));
        check_val(case_val.run_at<16, 8, false>(264, 4096, 3));
        check_val(case_val.run_at<8, 16, true>(5, 24, 5));
        check_val(case_val.run_at<16, 2, true>(33, 1000, 4));
        check_val(case_val.run_at<8, 3, true>(400, 2048, 3));     // 96 threads: a block size between powers of two
        check_val(case_val.run_at<4, 24, false>(66, 4096, 2));    // 768 threads
    }
    print_test_name(__func__);
    ASSERT_EQ((long)std::count(fits_list.begin(), fits_list.end(), true), (long)fits_list.size());
    ASSERT_EQ((long)std::count(same_list.begin(), same_list.end(), true), (long)same_list.size());
}

static void every_share_and_part_count_gives_adamw_bf16s_bits(void)
{
    // Every blocks an SM B and part count tune_adamw_tma_bf16 sweeps, with a master: B x SMs blocks, each block's share
    // of shared memory cut into the parts. The tile fits, B blocks of it fit on an SM at once (the share's rounding is
    // enough), and the step gives adamw_update_bf16's bits, TMA storing and the threads.
    const int blocks_per_sm_list[] = {1, 2, 3, 4, 6, 8, 16, 32};
    AdamwCase case_val(SMALL_PARAMS, SMALL_DECAYED_PARAMS, true);
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    auto kernel_ptr = adamw_tma_bf16_kernel<8, 1, true>;
    cudaCheck(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, ADAMW_TMA_BF16_MAX_SHARED_BYTES));
    std::vector<bool> fits_list, room_list, same_list;
    for (int blocks_per_sm_val : blocks_per_sm_list)
        for (int parts_val = 1; parts_val <= ADAMW_TMA_BF16_MAX_STAGES; parts_val++)
        {
            int tile_values_val = adamw_tma_bf16_tile_values(adamw_tma_bf16_share_bytes(blocks_per_sm_val), parts_val, true);
            AdamwTmaBf16Tile tile_val;
            size_t shared_bytes_val = 0;
            fits_list.push_back(adamw_tma_bf16_plan(&tile_val, &shared_bytes_val, true, tile_values_val, parts_val, SMALL_PARAMS));
            int blocks_on_sm_val = 0;
            cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_on_sm_val, kernel_ptr, HARDWARE_THREADS_PER_WARP, shared_bytes_val));
            room_list.push_back(blocks_on_sm_val >= blocks_per_sm_val);
            int blocks_val = sms_val * blocks_per_sm_val;
            same_list.push_back(case_val.run_at<8, 1, true>(blocks_val, tile_values_val, parts_val) && case_val.same());
            same_list.push_back(case_val.run_at<8, 1, false>(blocks_val, tile_values_val, parts_val) && case_val.same());
        }
    print_test_name(__func__);
    ASSERT_EQ((long)std::count(fits_list.begin(), fits_list.end(), true), (long)fits_list.size());
    ASSERT_EQ((long)std::count(room_list.begin(), room_list.end(), true), (long)room_list.size());
    ASSERT_EQ((long)std::count(same_list.begin(), same_list.end(), true), (long)same_list.size());
}

static void a_tile_that_does_not_fit_is_refused(void)
{
    // tile_values not whole 16-byte cells of bfloat16 (12), none (0), no sets, more than ADAMW_TMA_BF16_MAX_STAGES, a
    // set too big for shared memory (32,768 values with a master: 320 KB): false, and nothing launched.
    AdamwCase case_val(SMALL_PARAMS, SMALL_DECAYED_PARAMS, true);
    print_test_name(__func__);
    ASSERT_EQ((case_val.run_at<8, 8, true>(132, 12, 1)), false);
    ASSERT_EQ((case_val.run_at<8, 8, true>(132, 0, 1)), false);
    ASSERT_EQ((case_val.run_at<8, 8, true>(132, 2048, 0)), false);
    ASSERT_EQ((case_val.run_at<8, 8, true>(132, 2048, ADAMW_TMA_BF16_MAX_STAGES + 1)), false);
    ASSERT_EQ((case_val.run_at<8, 8, true>(132, 32768, 1)), false);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    the_default_settings_give_adamw_bf16s_bits();
    the_edge_settings_give_adamw_bf16s_bits();
    every_share_and_part_count_gives_adamw_bf16s_bits();
    a_tile_that_does_not_fit_is_refused();

    printf("\nall tests passed\n");
    return 0;
}
