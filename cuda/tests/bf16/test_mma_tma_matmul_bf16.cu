// test_mma_tma_matmul_bf16.cu — mma_tma_matmul_bf16 (tiles brought by TMA) gives the same bits as mma_matmul_bf16 (tiles copied by the threads), at the defaults and at the edge settings.
#include <torch/types.h>              // torch::randn, torch::equal; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <algorithm>                  // std::count
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../../kernels/bf16/mma_matmul_bf16.cuh"
#include "../../kernels/bf16/mma_tma_matmul_bf16.cuh"

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Tells if two bfloat16 tensors hold the same bits.
///
///   first_tensor, second_tensor   the same shape, on the GPU
static bool same_bits(const Tensor &first_tensor, const Tensor &second_tensor)
{
    torch::TensorOptions shorts_val = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA);
    torch::Tensor first_val = torch::from_blob(first_tensor.data_ptr(), {(int64_t)first_tensor.numel()}, shorts_val);
    torch::Tensor second_val = torch::from_blob(second_tensor.data_ptr(), {(int64_t)second_tensor.numel()}, shorts_val);
    return torch::equal(first_val, second_val);
}

/// One product's random operands, and mma_matmul_bf16's output on them (the threads' copies): the expected bits.
struct ProductCase
{
    size_t rows, columns, sum_length;
    bool left_turned, right_turned, add_onto;
    Tensor left_tensor, right_tensor, bias_tensor, start_tensor, expected_tensor, output_tensor;

    /// Makes the operands, output's start (random if added onto, else 0s) and mma_matmul_bf16's result.
    ///
    ///   rows_val, columns_val, sum_length_val   the product's sizes
    ///   with_bias_val                           a bias {columns}
    ///   left_turned_val, right_turned_val       how left and right sit
    ///   add_onto_val                            added onto output
    ProductCase(size_t rows_val, size_t columns_val, size_t sum_length_val, bool with_bias_val, bool left_turned_val, bool right_turned_val,
                bool add_onto_val)
        : rows(rows_val), columns(columns_val), sum_length(sum_length_val), left_turned(left_turned_val), right_turned(right_turned_val),
          add_onto(add_onto_val)
    {
        torch::manual_seed(0);
        left_tensor = random_bf16_tensor(left_turned ? Shape{sum_length, rows} : Shape{rows, sum_length});
        right_tensor = random_bf16_tensor(right_turned ? Shape{sum_length, columns} : Shape{columns, sum_length});
        bias_tensor = with_bias_val ? random_bf16_tensor({columns}) : Tensor();
        start_tensor = add_onto ? random_bf16_tensor({rows, columns}) : Tensor::zeros({rows, columns}, DType::BFloat16);
        expected_tensor = start_tensor.clone();
        mma_matmul_bf16(expected_tensor, left_tensor, right_tensor, bias_tensor, add_onto, left_turned, right_turned);
        output_tensor = start_tensor.clone();
    }

    /// Runs mma_tma_matmul_bf16_at at one setting from output's start, and tells if it gave the expected bits.
    ///
    ///   block_rows_per_group_val, parts_val, blocks_val   the launch's settings
    template <typename SETTING, bool SWIZZLE>
    bool same_at(int block_rows_per_group_val, int parts_val, int blocks_val)
    {
        output_tensor.copy_(start_tensor);
        mma_tma_matmul_bf16_at<SETTING, SWIZZLE>(output_tensor, left_tensor, right_tensor, bias_tensor, add_onto, left_turned, right_turned,
                                                 block_rows_per_group_val, parts_val, blocks_val);
        cudaCheck(cudaDeviceSynchronize());
        return same_bits(output_tensor, expected_tensor);
    }
};

// ── tests ─────────────────────────────────────────────────────────────────────

static void the_defaults_give_mma_matmul_bf16s_bits(void)
{
    // Expect: at the defaults, mma_matmul_bf16's output to the last bit, on a size no square or step divides ({200,
    // 104}, a sum of 136, a bias) in each of the 4 ways left and right sit, and on GPT-2 small's qkv products (the
    // forward with a bias, input_grad with right turned, weight_grad both turned and added onto random values). The
    // same mma on the same values in the same order of k; only who brings the tiles changes, and TMA brings 0s past the
    // edges as the threads' copies do.
    std::vector<bool> same_list;
    for (int way_index_val = 0; way_index_val < 4; way_index_val++)
    {
        ProductCase case_val(200, 104, 136, true, (way_index_val & 1) != 0, (way_index_val & 2) != 0, false);
        case_val.output_tensor.copy_(case_val.start_tensor);
        mma_tma_matmul_bf16(case_val.output_tensor, case_val.left_tensor, case_val.right_tensor, case_val.bias_tensor, false,
                            case_val.left_turned, case_val.right_turned);
        cudaCheck(cudaDeviceSynchronize());
        same_list.push_back(same_bits(case_val.output_tensor, case_val.expected_tensor));
    }
    ProductCase qkv_list[] = {ProductCase(4096, 2304, 768, true, false, false, false), ProductCase(4096, 768, 2304, false, false, true, false),
                              ProductCase(2304, 768, 4096, false, true, true, true)};
    for (ProductCase &case_val : qkv_list)
    {
        case_val.output_tensor.copy_(case_val.start_tensor);
        mma_tma_matmul_bf16(case_val.output_tensor, case_val.left_tensor, case_val.right_tensor, case_val.bias_tensor, case_val.add_onto,
                            case_val.left_turned, case_val.right_turned);
        cudaCheck(cudaDeviceSynchronize());
        same_list.push_back(same_bits(case_val.output_tensor, case_val.expected_tensor));
    }
    print_test_name(__func__);
    ASSERT_EQ((long)std::count(same_list.begin(), same_list.end(), true), (long)same_list.size());
}

static void the_edge_settings_give_mma_matmul_bf16s_bits(void)
{
    // Expect: mma_matmul_bf16's bits at the sweep's ends, plain and both turned, on {200, 104}, a sum of 136, a bias:
    // swizzled (32-, 64- and 128-byte rows; a turned tile in boxes of 64 lines) and not, the smallest square (64 x 64,
    // one warp) and the largest (256 x 256, 32 warps), every step of k, 1 part (wait for each step) and 8 (more than
    // the sum has steps), one block a square and a few blocks walking many squares (the parts used in turn over the
    // squares), the block orders 1 to 64.
    std::vector<bool> same_list;
    for (int way_index_val = 0; way_index_val < 2; way_index_val++)
    {
        bool turned_val = way_index_val == 1;
        ProductCase case_val(200, 104, 136, true, turned_val, turned_val, false);
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<64, 64, 16, 1, 1>, true>(1, 1, 0));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<64, 64, 16, 1, 1>, false>(1, 1, 0));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<256, 256, 64, 4, 8>, true>(64, 3, 0));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<256, 256, 64, 4, 8>, false>(64, 3, 0));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<64, 256, 32, 2, 4>, true>(2, 8, 1));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<256, 64, 32, 4, 2>, false>(3, 4, 2));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<128, 128, 64, 1, 8>, true>(8, 2, 3));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<128, 64, 16, 4, 4>, false>(4, 6, 0));
        same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<64, 64, 32, 2, 2>, true>(16, 1, 5));
    }
    print_test_name(__func__);
    ASSERT_EQ((long)std::count(same_list.begin(), same_list.end(), true), (long)same_list.size());
}

static void the_raced_settings_give_mma_matmul_bf16s_bits(void)
{
    // Expect: mma_matmul_bf16's bits, run after run, at the settings that lost the race between a step's last reads
    // of a part and TMA's write of the step parts on into it (before the fence.proxy.async ahead of the step's
    // __syncthreads): on nanochat d24's attention_output forward ({32768, 1536}, a sum of 1536), swizzled, k 64, warps
    // of 32 x 32, 1 to 4 parts. Without the fence: 64 x 128 and 128 x 64 wrong on every run, 64 x 64 on 1 in 4 at 1 part.
    std::vector<bool> same_list;
    ProductCase case_val(32768, 1536, 1536, false, false, false, false);
    for (int parts_val = 1; parts_val <= 4; parts_val++)
        for (int run_index_val = 0; run_index_val < 25; run_index_val++)
        {
            same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<64, 128, 64, 2, 4>, true>(16, parts_val, 0));
            same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<128, 64, 64, 4, 2>, true>(16, parts_val, 0));
            same_list.push_back(case_val.same_at<MmaMatmulBf16Setting<64, 64, 64, 2, 2>, true>(16, parts_val, 0));
        }
    print_test_name(__func__);
    ASSERT_EQ((long)std::count(same_list.begin(), same_list.end(), true), (long)same_list.size());
}

static void the_default_kernels_fit_their_block(void)
{
    // Expect: the defaults' 4 kernels (one per turned flags) take MMA_MATMUL_BF16_THREADS_PER_BLOCK threads and spill nothing.
    const void *kernels_list[] = {(const void *)mma_tma_matmul_bf16_kernel<MmaMatmulBf16Setting<>, MMA_TMA_MATMUL_BF16_SWIZZLE, false, false>,
                                  (const void *)mma_tma_matmul_bf16_kernel<MmaMatmulBf16Setting<>, MMA_TMA_MATMUL_BF16_SWIZZLE, false, true>,
                                  (const void *)mma_tma_matmul_bf16_kernel<MmaMatmulBf16Setting<>, MMA_TMA_MATMUL_BF16_SWIZZLE, true, false>,
                                  (const void *)mma_tma_matmul_bf16_kernel<MmaMatmulBf16Setting<>, MMA_TMA_MATMUL_BF16_SWIZZLE, true, true>};
    for (const void *kernel_ptr : kernels_list)
    {
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernel_ptr));
        print_test_name(__func__);
        printf("    %d registers a thread, largest block %d threads, %zu bytes local memory\n", attributes_val.numRegs,
               attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= MMA_MATMUL_BF16_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    the_defaults_give_mma_matmul_bf16s_bits();
    the_edge_settings_give_mma_matmul_bf16s_bits();
    the_raced_settings_give_mma_matmul_bf16s_bits();
    the_default_kernels_fit_their_block();

    printf("\nall tests passed\n");
    return 0;
}
