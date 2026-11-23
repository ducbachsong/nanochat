// tune_cutlass_mma_bf16.cu — the timings of CUTLASS's mma.sync products (its Ampere kernels: mma.sync m16n8k16, cp.async stages, no wgmma) on a model's 15 linear products, every setting below, against PyTorch's matmul (cuBLAS: wgmma on an H100); each checked against cuBLAS's values.
#include "matmul_products.h"          // the 15 products, their operands, cuBLAS; libtorch before any CUDA header
#include <algorithm>                  // std::sort
#include <memory>                     // std::unique_ptr: the settings
#include <string>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"                       // CUTLASS 2's device-wide product: its Sm80 kernels are mma.sync
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_bf16_flops, gpu_folder_name

#define FIRST_PASS_RUNS 5             // timed runs of each setting, averaged
#define FINALISTS 5                   // the fastest settings, timed again over SWEEP_ROUNDS rounds with cuBLAS
#define MOST_RELATIVE_ERROR 1e-2f     // largest |CUTLASS - cuBLAS| over largest |cuBLAS|: the same product, another order of k
static constexpr int SPLIT_K_LIST[] = {1, 2, 4, 8};   // slices of k (CUTLASS's serial split-k), tried on the products whose sum is longer than their output is wide

// ── the settings ──────────────────────────────────────────────────────────────

/// One CUTLASS product built for one layout of left and right: prepared on a product's operands, then run.
struct CutlassSetting
{
    std::string name;                 // its block's square and step, its warp's, its stages
    int block_rows = 0, block_columns = 0, block_k = 0, stages = 0;
    virtual ~CutlassSetting() = default;

    /// Prepares it on a product's operands at split_k_val slices of k: gives why not, or "" if it can run.
    ///
    ///   operands_val   the product's operands and output
    ///   split_k_val    slices of k
    virtual std::string prepare(MatmulOperands &operands_val, int split_k_val) = 0;

    /// Runs it as prepared: gives why not, or "" if it started.
    virtual std::string run(void) = 0;
};

/// A CutlassSetting for one cutlass::gemm::device::Gemm.
template <typename GEMM>
struct CutlassGemmSetting : CutlassSetting
{
    GEMM gemm;
    torch::Tensor workspace;          // split-k's semaphores

    std::string prepare(MatmulOperands &operands_val, int split_k_val) override
    {
        // output = left · rightᵀ: A is left {rows, sum_length} (row-major) or turned {sum_length, rows} (column-major); B is
        // rightᵀ: right {columns, sum_length} is B column-major, turned {sum_length, columns} B row-major. C and D are
        // output, row-major; beta 1 adds onto it (weight_grad).
        const MatmulProduct &product_val = operands_val.product;
        using ElementA = typename GEMM::ElementA;
        using ElementB = typename GEMM::ElementB;
        using ElementC = typename GEMM::ElementC;
        int rows_val = (int)product_val.rows, columns_val = (int)product_val.columns, sum_length_val = (int)product_val.sum_length;
        auto *left_ptr = reinterpret_cast<ElementA *>(operands_val.left_tensor.data_ptr());
        auto *right_ptr = reinterpret_cast<ElementB *>(operands_val.right_tensor.data_ptr());
        auto *output_ptr = reinterpret_cast<ElementC *>(operands_val.output_tensor.data_ptr());
        typename GEMM::Arguments arguments_val({rows_val, columns_val, sum_length_val},
                                               {left_ptr, product_val.left_turned ? rows_val : sum_length_val},
                                               {right_ptr, product_val.right_turned ? columns_val : sum_length_val},
                                               {output_ptr, columns_val}, {output_ptr, columns_val},
                                               {1.0f, product_val.add_onto ? 1.0f : 0.0f}, split_k_val);
        cutlass::Status status_val = gemm.can_implement(arguments_val);
        if (status_val != cutlass::Status::kSuccess)
            return std::string("can not: ") + cutlassGetStatusString(status_val);
        size_t workspace_bytes_val = GEMM::get_workspace_size(arguments_val);
        workspace = torch::zeros({(int64_t)std::max(workspace_bytes_val, (size_t)1)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        status_val = gemm.initialize(arguments_val, workspace.data_ptr());
        if (status_val != cutlass::Status::kSuccess)
            return std::string("not set up: ") + cutlassGetStatusString(status_val);
        return "";
    }

    std::string run(void) override
    {
        cutlass::Status status_val = gemm.run();
        return status_val == cutlass::Status::kSuccess ? "" : cutlassGetStatusString(status_val);
    }
};

/// Adds one CUTLASS product to a list: its block's square and step, its warp's, its stages, for one layout of left and right.
template <typename LAYOUT_A, typename LAYOUT_B, int BLOCK_ROWS, int BLOCK_COLUMNS, int BLOCK_K, int WARP_ROWS, int WARP_COLUMNS, int STAGES>
static void add_setting(std::vector<std::unique_ptr<CutlassSetting>> &settings_list)
{
    // bfloat16 in and out, float sums, mma.sync m16n8k16 (Sm80, OpClassTensorOp), 8 values a load, the blocks in
    // CUTLASS's usual swizzle of 8, serial split-k allowed
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::bfloat16_t, LAYOUT_A, cutlass::bfloat16_t, LAYOUT_B, cutlass::bfloat16_t, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K>,
        cutlass::gemm::GemmShape<WARP_ROWS, WARP_COLUMNS, BLOCK_K>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, STAGES, 8, 8, true>;
    auto setting_ptr = std::make_unique<CutlassGemmSetting<Gemm>>();
    char name_list[96];
    snprintf(name_list, sizeof(name_list), "block %3d x %3d x %2d, warp %3d x %3d, %2d stages", BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, WARP_ROWS,
             WARP_COLUMNS, STAGES);
    setting_ptr->name = name_list;
    setting_ptr->block_rows = BLOCK_ROWS;
    setting_ptr->block_columns = BLOCK_COLUMNS;
    setting_ptr->block_k = BLOCK_K;
    setting_ptr->stages = STAGES;
    settings_list.push_back(std::move(setting_ptr));
}

/// Gives every CUTLASS product the tune tries for one layout of left and right: CUTLASS's profiler's usual Sm80 bfloat16 tiles.
template <typename LAYOUT_A, typename LAYOUT_B>
static std::vector<std::unique_ptr<CutlassSetting>> settings_for(void)
{
    std::vector<std::unique_ptr<CutlassSetting>> settings_list;
    add_setting<LAYOUT_A, LAYOUT_B, 256, 128, 32, 64, 64, 3>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 256, 32, 64, 64, 3>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 256, 64, 32, 64, 64, 4>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 64, 256, 32, 64, 64, 4>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 128, 32, 64, 64, 4>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 128, 32, 64, 64, 5>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 64, 32, 64, 32, 6>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 64, 128, 32, 32, 64, 6>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 64, 64, 32, 32, 32, 10>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 128, 64, 64, 64, 3>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 128, 64, 64, 64, 4>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 256, 128, 64, 64, 64, 3>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 256, 64, 64, 64, 3>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 64, 128, 64, 32, 64, 4>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 128, 64, 64, 64, 32, 4>(settings_list);
    add_setting<LAYOUT_A, LAYOUT_B, 64, 64, 64, 32, 32, 5>(settings_list);
    return settings_list;
}

// ══ the tune ══════════════════════════════════════════════════════════════════

/// One setting tried on one product, and its times.
struct Trial
{
    CutlassSetting *setting_ptr;
    int split_k_val;
    std::string skipped_val;          // empty: it ran; else why not
    float error_val = 0.0f;           // largest |CUTLASS - cuBLAS| over largest |cuBLAS|
    float first_ms_val = 0.0f, final_ms_val = 0.0f;
};

/// One product's result, for the table at the end.
struct ProductResult
{
    std::string name;
    float cublas_ms = 0.0f, best_ms = 0.0f, best_unsplit_ms = 0.0f;   // the second pass's
    std::string best_setting, best_unsplit_setting;
    size_t wrong = 0;                 // settings past MOST_RELATIVE_ERROR
};

/// Times every setting on one product, prints its fastest, and writes every trial to the .csv.
///
///   product_val    the product
///   settings_ptr   the settings built for its layout of left and right
///   csv_ptr        the .csv every trial's row goes to; nullptr: none
static ProductResult time_every_setting(const MatmulProduct &product_val, std::vector<std::unique_ptr<CutlassSetting>> *settings_ptr, FILE *csv_ptr)
{
    // First pass: each setting (and slices of k, for a sum longer than the output is wide) run once from 0s and checked
    // against cuBLAS's values within MOST_RELATIVE_ERROR (not the bits: CUTLASS sums k in another order), then timed
    // FIRST_PASS_RUNS times. Second pass: the FINALISTS fastest and the fastest without split-k, with cuBLAS, over
    // SWEEP_ROUNDS rounds taking turns. cuBLAS is warmed up first: its first call is not the product's time.
    //
    // Expect: every setting that runs within MOST_RELATIVE_ERROR of cuBLAS.
    MatmulOperands operands_val(product_val);
    operands_val.cublas_output.zero_();
    operands_val.run_cublas();
    cudaCheck(cudaDeviceSynchronize());
    torch::Tensor expected_val = operands_val.cublas_output.to(torch::kFloat).clone();
    float expected_largest_val = expected_val.abs().max().item<float>();
    warm_up_gpu([&] { operands_val.run_cublas(); });
    bool long_sum_val = product_val.sum_length > std::max(product_val.rows, product_val.columns);
    double flops_val = product_val.flops();
    auto tflops_val = [&](float ms_val) { return flops_val / ms_val / TERA_IN_MILLISECONDS; };
    printf("\n%s: output {%zu, %zu}, sum of %zu%s%s%s: %.2f TFLOP\n", product_val.name.c_str(), product_val.rows, product_val.columns,
           product_val.sum_length, product_val.left_turned ? ", left turned" : "", product_val.right_turned ? ", right turned" : "",
           product_val.add_onto ? ", added onto" : "", flops_val / 1e12);

    // 1. first pass
    std::vector<Trial> trials_list;
    for (const std::unique_ptr<CutlassSetting> &setting_ptr : *settings_ptr)
        for (int split_k_val : SPLIT_K_LIST)
        {
            if (split_k_val > 1 && !long_sum_val)
                continue;
            Trial trial_val{setting_ptr.get(), split_k_val};
            operands_val.output_tensor.zero_();
            trial_val.skipped_val = setting_ptr->prepare(operands_val, split_k_val);
            if (trial_val.skipped_val.empty())
                trial_val.skipped_val = setting_ptr->run();
            if (trial_val.skipped_val.empty())
            {
                cudaError_t error_val = cudaDeviceSynchronize();
                if (error_val != cudaSuccess)
                {
                    printf("    stopped: %s, split %d failed while running: %s\n", setting_ptr->name.c_str(), split_k_val, cudaGetErrorString(error_val));
                    exit(EXIT_FAILURE);
                }
                torch::Tensor got_val = torch_view_of(operands_val.output_tensor).to(torch::kFloat);
                trial_val.error_val = (got_val - expected_val).abs().max().item<float>() / expected_largest_val;
                float total_ms_val = 0.0f;
                for (int run_index_val = 0; run_index_val < FIRST_PASS_RUNS; run_index_val++)
                    total_ms_val += gpu_milliseconds([&] { setting_ptr->run(); });
                trial_val.first_ms_val = total_ms_val / FIRST_PASS_RUNS;
            }
            trials_list.push_back(trial_val);
        }

    // 2. second pass: the fastest within the error, and the fastest without split-k, taking turns with cuBLAS
    std::vector<int> order_list;
    for (size_t index_val = 0; index_val < trials_list.size(); index_val++)
        if (trials_list[index_val].skipped_val.empty() && trials_list[index_val].error_val <= MOST_RELATIVE_ERROR)
            order_list.push_back((int)index_val);
    std::sort(order_list.begin(), order_list.end(), [&](int first_val, int second_val) {
        return trials_list[first_val].first_ms_val < trials_list[second_val].first_ms_val;
    });
    std::vector<int> finalists_list(order_list.begin(), order_list.begin() + std::min((int)order_list.size(), FINALISTS));
    for (int index_val : order_list)
        if (trials_list[index_val].split_k_val == 1)
        {
            if (std::find(finalists_list.begin(), finalists_list.end(), index_val) == finalists_list.end())
                finalists_list.push_back(index_val);
            break;
        }
    std::vector<float> total_ms_list(finalists_list.size(), 0.0f);
    float cublas_total_ms_val = 0.0f;
    for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
    {
        cublas_total_ms_val += gpu_milliseconds([&] { operands_val.run_cublas(); });
        for (size_t finalist_index_val = 0; finalist_index_val < finalists_list.size(); finalist_index_val++)
        {
            Trial &trial_val = trials_list[finalists_list[finalist_index_val]];
            trial_val.setting_ptr->prepare(operands_val, trial_val.split_k_val);
            total_ms_list[finalist_index_val] += gpu_milliseconds([&] { trial_val.setting_ptr->run(); });
        }
    }
    for (size_t finalist_index_val = 0; finalist_index_val < finalists_list.size(); finalist_index_val++)
        trials_list[finalists_list[finalist_index_val]].final_ms_val = total_ms_list[finalist_index_val] / SWEEP_ROUNDS;
    float cublas_ms_val = cublas_total_ms_val / SWEEP_ROUNDS;

    // 3. the finalists, fastest first
    std::sort(finalists_list.begin(), finalists_list.end(),
              [&](int first_val, int second_val) { return trials_list[first_val].final_ms_val < trials_list[second_val].final_ms_val; });
    printf("    PyTorch (cuBLAS, wgmma): %.4f ms, %.0f TFLOPS\n", cublas_ms_val, tflops_val(cublas_ms_val));
    printf("         ms  TFLOPS  x cuBLAS   error    setting\n");
    ProductResult result_val;
    result_val.name = product_val.name;
    result_val.cublas_ms = cublas_ms_val;
    for (int index_val : finalists_list)
    {
        const Trial &trial_val = trials_list[index_val];
        printf("    %7.4f  %6.0f    %5.2fx  %.1e   %s, split %d\n", trial_val.final_ms_val, tflops_val(trial_val.final_ms_val),
               cublas_ms_val / trial_val.final_ms_val, trial_val.error_val, trial_val.setting_ptr->name.c_str(), trial_val.split_k_val);
        std::string setting_val = trial_val.setting_ptr->name + ", split " + std::to_string(trial_val.split_k_val);
        if (result_val.best_ms == 0.0f)
        {
            result_val.best_ms = trial_val.final_ms_val;
            result_val.best_setting = setting_val;
        }
        if (trial_val.split_k_val == 1 && result_val.best_unsplit_ms == 0.0f)
        {
            result_val.best_unsplit_ms = trial_val.final_ms_val;
            result_val.best_unsplit_setting = setting_val;
        }
    }

    // 4. the ones past the error, and every trial into the .csv
    for (const Trial &trial_val : trials_list)
    {
        bool wrong_val = trial_val.skipped_val.empty() && trial_val.error_val > MOST_RELATIVE_ERROR;
        if (wrong_val)
        {
            printf("    PAST THE ERROR (%.1e): %s, split %d\n", trial_val.error_val, trial_val.setting_ptr->name.c_str(), trial_val.split_k_val);
            result_val.wrong++;
        }
        if (csv_ptr != nullptr)
            fprintf(csv_ptr, "\"%s\",%d,%d,%d,%d,%d,%.5f,%.5f,%.1f,%.2e,\"%s\"\n", product_val.name.c_str(), trial_val.setting_ptr->block_rows,
                    trial_val.setting_ptr->block_columns, trial_val.setting_ptr->block_k, trial_val.setting_ptr->stages, trial_val.split_k_val,
                    trial_val.first_ms_val, trial_val.final_ms_val, trial_val.skipped_val.empty() ? tflops_val(trial_val.first_ms_val) : 0.0,
                    trial_val.error_val, trial_val.skipped_val.empty() ? (wrong_val ? "past the error" : "ran") : trial_val.skipped_val.c_str());
    }
    return result_val;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    // ./tune_cutlass_mma_bf16: nanochat d24's 15 products; gpt2: GPT-2 small's (biases: not taken, it stops); a layer's
    // name (qkv, attention_output, mlp_up, mlp_down, head): only those layers' products.
    ModelSize size_val;
    std::vector<MatmulProduct> products_list = products_from_arguments(argc, argv, &size_val);
    std::string folder_val = "cuda/tests/results/" + gpu_folder_name();
    std::string log_path_val = start_run_log(folder_val.c_str(), __FILE__, "tune_", "txt");
    print_gpu();
    printf("%s: %zu products; CUTLASS's Sm80 products (mma.sync m16n8k16, no wgmma, no TMA) against cuBLAS\n", size_val.name_ptr, products_list.size());
    for (const MatmulProduct &product_val : products_list)
        if (product_val.has_bias)
        {
            printf("error: %s has a bias: this tune runs products without one\n", product_val.name.c_str());
            return EXIT_FAILURE;
        }

    // the settings for each way left and right sit: forward (left, right as they sit), input_grad (right turned),
    // weight_grad (both turned)
    std::vector<std::unique_ptr<CutlassSetting>> plain_list = settings_for<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>();
    std::vector<std::unique_ptr<CutlassSetting>> right_turned_list = settings_for<cutlass::layout::RowMajor, cutlass::layout::RowMajor>();
    std::vector<std::unique_ptr<CutlassSetting>> both_turned_list = settings_for<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>();

    std::string csv_path_val = log_path_val.substr(0, log_path_val.size() - 3) + "csv";
    FILE *csv_ptr = fopen(csv_path_val.c_str(), "w");
    if (csv_ptr != nullptr)
        fprintf(csv_ptr, "product,block_rows,block_columns,block_k,stages,split_k,first_ms,final_ms,tflops,error,status\n");
    std::vector<ProductResult> results_list;
    for (const MatmulProduct &product_val : products_list)
    {
        std::vector<std::unique_ptr<CutlassSetting>> *settings_ptr = !product_val.left_turned && !product_val.right_turned ? &plain_list
                                                                    : !product_val.left_turned                            ? &right_turned_list
                                                                                                                         : &both_turned_list;
        if (product_val.left_turned && !product_val.right_turned)
        {
            printf("error: %s: left turned alone is not built\n", product_val.name.c_str());
            return EXIT_FAILURE;
        }
        results_list.push_back(time_every_setting(product_val, settings_ptr, csv_ptr));
        if (csv_ptr != nullptr)
            fflush(csv_ptr);
    }
    if (csv_ptr != nullptr)
    {
        fclose(csv_ptr);
        printf("\nevery setting: %s\n", csv_path_val.c_str());
    }

    // every product side by side: cuBLAS (wgmma), CUTLASS's best (mma.sync), and its best without split-k
    print_test_name("every_product");
    printf("    %-28s  %8s  %8s  %8s  %8s   %s\n", "product", "cuBLAS", "CUTLASS", "x cuBLAS", "no split", "best setting");
    double cublas_total_ms_val = 0.0, best_total_ms_val = 0.0, unsplit_total_ms_val = 0.0;
    size_t wrong_val = 0;
    for (const ProductResult &result_val : results_list)
    {
        printf("    %-28s  %8.4f  %8.4f  %7.2fx  %8.4f   %s\n", result_val.name.c_str(), result_val.cublas_ms, result_val.best_ms,
               result_val.cublas_ms / result_val.best_ms, result_val.best_unsplit_ms, result_val.best_setting.c_str());
        cublas_total_ms_val += result_val.cublas_ms;
        best_total_ms_val += result_val.best_ms;
        unsplit_total_ms_val += result_val.best_unsplit_ms;
        wrong_val += result_val.wrong;
    }
    printf("    %-28s  %8.4f  %8.4f  %7.2fx  %8.4f\n", "all, once each", cublas_total_ms_val, best_total_ms_val,
           cublas_total_ms_val / best_total_ms_val, unsplit_total_ms_val);

    // every setting that ran was within MOST_RELATIVE_ERROR of cuBLAS (the ones that were not are printed above)
    ASSERT_EQ((long)wrong_val, 0L);
    printf("\nall tunes done\n");
    return 0;
}
