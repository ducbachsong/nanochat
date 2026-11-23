// tune_mma_tma_matmul_bf16.cu — the timings of cuda/kernels/bf16/mma_tma_matmul_bf16.cuh (tiles brought by TMA, mma.sync): every setting, father to son, on a model's 15 linear products, against the defaults and PyTorch's matmul (cuBLAS); each checked to the bit against mma_matmul_bf16 (tiles copied by the threads). With serve: only the settings tune_mma_tma_matmul_bf16.py asks for.
#include "matmul_products.h"          // the 15 products, their operands, cuBLAS; libtorch before any CUDA header
#include <unistd.h>                   // dup, dup2: serve's replies on the real stdout
#include <algorithm>                  // std::sort
#include <chrono>                     // the sweep's progress
#include <iostream>                   // std::cin: serve's requests
#include <map>                        // the skipped settings, counted by reason
#include <memory>                     // std::unique_ptr: serve's product
#include <sstream>                    // a request's fields
#include <string>
#include <vector>
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_bf16_flops, gpu_folder_name, for_every_setting
#include "../../kernels/bf16/mma_matmul_bf16.cuh"       // the threads' copies: the expected bits
#include "../../kernels/bf16/mma_tma_matmul_bf16.cuh"

// The sweep, every one with every other, father to son: the kernel built (the square's rows and columns, the step of
// k, the threads a block, the warps down: the rest go across, the swizzle TMA lays the tiles in), then what a launch
// is given (the parts of shared memory, the blocks, their order)
static constexpr int BLOCK_ROWS_LIST[] = {64, 128, 256};
static constexpr int BLOCK_COLUMNS_LIST[] = {64, 128, 256};
static constexpr int BLOCK_K_LIST[] = {16, 32, 64};
static constexpr int THREADS_LIST[] = {32, 64, 128, 256, 512, 1024};   // threads a block: whole warps
static constexpr int WARPS_DOWN_LIST[] = {1, 2, 4, 8};                // the block's warps down; across: threads / 32 / down
static constexpr bool SWIZZLE_LIST[] = {true, false};                 // TMA's swizzle (a tile row's 32, 64 or 128 bytes), or the rows as they sit
static constexpr int PARTS_LIST[] = {1, 2, 3, 4, 5, 6, 7, 8};         // parts of shared memory, a step's tiles each
static constexpr int BLOCKS_PER_SM_LIST[] = {0, 1, 2, 3, 4, 8};       // blocks a launch, per SM; 0: one a square
static constexpr int BLOCK_ORDER_LIST[] = {1, 4, 16, 64};             // rows of squares a group of blocks runs down
static_assert(sizeof(PARTS_LIST) / sizeof(PARTS_LIST[0]) == MMA_TMA_MATMUL_BF16_MAX_PARTS, "every part count the kernel takes");
enum SweepList { SWEEP_BLOCK_ROWS, SWEEP_BLOCK_COLUMNS, SWEEP_BLOCK_K, SWEEP_THREADS, SWEEP_WARPS_DOWN, SWEEP_SWIZZLE, SWEEP_PARTS,
                 SWEEP_BLOCKS, SWEEP_BLOCK_ORDER };   // each list's place in time_every_setting's tunes_list, father to son: the same order there
#define COUNT(list) ((int)(sizeof(list) / sizeof((list)[0])))
#define MAX_REGISTERS_PER_THREAD 255  // the most registers a thread can have
#define FIRST_PASS_RUNS 3             // timed runs of each setting in the first pass, averaged
#define SLOW_CUT 20.0f                // a setting this many times the defaults: timed once in the first pass
#define FINALISTS 50                  // the first pass's fastest, timed again over SWEEP_ROUNDS rounds taking turns
#define TOP_PRINTED 10                // finalists printed
#define PROGRESS_EVERY 2000           // settings between progress lines

/// Gives where value_val sits in a list, or -1.
///
///   list_list   the list
///   value_val   the value looked for
template <size_t N>
constexpr int index_of(const int (&list_list)[N], int value_val)
{
    for (size_t index_val = 0; index_val < N; index_val++)
        if (list_list[index_val] == value_val)
            return (int)index_val;
    return -1;
}

// ── the product's data ────────────────────────────────────────────────────────

/// One product's operands (matmul_products.h) and mma_matmul_bf16's result on them (the threads' copies): the expected bits.
struct ProductData : MatmulOperands
{
    torch::Tensor expected_bits;      // mma_matmul_bf16's output, as 16-bit ints

    /// Makes the operands and mma_matmul_bf16's result.
    ///
    ///   product_val   the product
    explicit ProductData(const MatmulProduct &product_val) : MatmulOperands(product_val)
    {
        // onto 0s, as the check runs each setting
        mma_matmul_bf16(output_tensor, left_tensor, right_tensor, bias_tensor, product.add_onto, product.left_turned, product.right_turned);
        cudaCheck(cudaDeviceSynchronize());
        expected_bits = torch_view_of(output_tensor, torch::kInt16).clone();
    }

    /// Runs the product at the defaults: mma_tma_matmul_bf16.
    void run_defaults(void)
    {
        mma_tma_matmul_bf16(output_tensor, left_tensor, right_tensor, bias_tensor, product.add_onto, product.left_turned, product.right_turned);
    }

    /// Tells if output holds mma_matmul_bf16's bits.
    bool same(void) const { return torch::equal(torch_view_of(output_tensor, torch::kInt16), expected_bits); }
};

// ── the kernels ───────────────────────────────────────────────────────────────

using LaunchProduct = cudaError_t (*)(nv_bfloat16 *, const nv_bfloat16 *, const nv_bfloat16 *, const nv_bfloat16 *, size_t, size_t,
                                      size_t, bool, int, int, int);
using KernelPointer = const void *;

/// One built kernel for one pair of turned flags: how to launch it, its registers; and why its settings are skipped if they are.
struct KernelInfo
{
    LaunchProduct launch_ptr = nullptr;   // mma_tma_matmul_bf16_launch for this setting, swizzle and turned flags
    KernelPointer kernel_ptr = nullptr;   // the kernel itself, for the blocks an SM holds at each part count
    int registers = 0;
    size_t local_bytes = 0;               // > 0: registers spilled to local memory, so its settings are skipped
    std::string error;                    // not empty: CUDA refused it, so its settings are skipped
};

/// One kernel the sweep can build: its warps, a part's shared memory, its sums a thread, and its info for each pair of turned flags.
struct KernelEntry
{
    std::string not_built;            // not empty: why it was not built (its warps do not split the square, or its sums need more registers than a thread has)
    int warps_across = 0;
    size_t part_bytes = 0;            // one part: a step's left and right tiles
    int sums = 0;                     // float sums a thread keeps
    KernelInfo infos_list[2][2];      // [left turned][right turned]: the products use (plain, plain), (plain, turned), (turned, turned)
};
static KernelEntry kernels_list[COUNT(BLOCK_ROWS_LIST)][COUNT(BLOCK_COLUMNS_LIST)][COUNT(BLOCK_K_LIST)][COUNT(THREADS_LIST)][COUNT(WARPS_DOWN_LIST)]
                               [COUNT(SWIZZLE_LIST)];

/// Reads one built kernel's registers, for one swizzle and pair of turned flags.
template <typename SETTING, bool SWIZZLE, bool LEFT_TURNED, bool RIGHT_TURNED>
static KernelInfo read_kernel_info(void)
{
    // Not stopping on an error: it is kept in error, and the kernel's settings are skipped with it.
    auto kernel_ptr = mma_tma_matmul_bf16_kernel<SETTING, SWIZZLE, LEFT_TURNED, RIGHT_TURNED>;
    KernelInfo info_val;
    info_val.launch_ptr = mma_tma_matmul_bf16_launch<SETTING, SWIZZLE, LEFT_TURNED, RIGHT_TURNED>;
    info_val.kernel_ptr = (KernelPointer)kernel_ptr;
    cudaFuncAttributes attributes_val = {};
    cudaError_t error_val = cudaFuncGetAttributes(&attributes_val, kernel_ptr);
    if (error_val == cudaSuccess)
        error_val = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES);
    cudaGetLastError();   // cleared: the next call does not see it
    info_val.registers = attributes_val.numRegs;
    info_val.local_bytes = attributes_val.localSizeBytes;
    if (error_val != cudaSuccess)
        info_val.error = cudaGetErrorString(error_val);
    return info_val;
}

/// Fills one entry of kernels_list: built only if its warps split the square and its sums fit a thread's registers.
template <int BLOCK_ROWS, int BLOCK_COLUMNS, int BLOCK_K, int THREADS, int WARPS_DOWN, bool SWIZZLE>
static void fill_kernel(void)
{
    // The warps: THREADS / 32, WARPS_DOWN of them down, the rest across; each warp's part of the square whole mma
    // tiles. A thread keeps its sums in registers, at most MAX_REGISTERS_PER_THREAD, and fewer the more threads a
    // block has (an SM's registers over the block): past that the kernel can only spill. Either way it is never built,
    // and its settings are skipped without the compile time. The products use three pairs of turned flags: only
    // those are built.
    constexpr int WARPS = THREADS / HARDWARE_THREADS_PER_WARP;
    constexpr int WARPS_ACROSS = WARPS % WARPS_DOWN == 0 ? WARPS / WARPS_DOWN : 0;
    KernelEntry &entry_val = kernels_list[index_of(BLOCK_ROWS_LIST, BLOCK_ROWS)][index_of(BLOCK_COLUMNS_LIST, BLOCK_COLUMNS)]
                                         [index_of(BLOCK_K_LIST, BLOCK_K)][index_of(THREADS_LIST, THREADS)][index_of(WARPS_DOWN_LIST, WARPS_DOWN)]
                                         [SWIZZLE ? 0 : 1];   // SWIZZLE_LIST's order
    entry_val.warps_across = WARPS_ACROSS;
    char reason_list[160];
    if constexpr (WARPS_ACROSS == 0)
    {
        snprintf(reason_list, sizeof(reason_list), "warps do not split: %d warps, %d down", WARPS, WARPS_DOWN);
        entry_val.not_built = reason_list;
    }
    else
    {
        using Setting = MmaMatmulBf16Setting<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, WARPS_DOWN, WARPS_ACROSS>;
        constexpr int MOST_REGISTERS = HARDWARE_REGISTERS_PER_SM / THREADS < MAX_REGISTERS_PER_THREAD ? HARDWARE_REGISTERS_PER_SM / THREADS
                                                                                                      : MAX_REGISTERS_PER_THREAD;
        if constexpr (!Setting::FITS)
        {
            snprintf(reason_list, sizeof(reason_list), "warps do not split the square: %d x %d warps on %d x %d (whole mma tiles of 16 x 8)",
                     WARPS_DOWN, WARPS_ACROSS, BLOCK_ROWS, BLOCK_COLUMNS);
            entry_val.not_built = reason_list;
        }
        else
        {
            entry_val.part_bytes = mma_tma_matmul_bf16_part_bytes<Setting>();
            entry_val.sums = Setting::SUMS;
            if constexpr (Setting::SUMS >= MOST_REGISTERS)
            {
                snprintf(reason_list, sizeof(reason_list), "too many sums a thread: %d floats, a thread has at most %d registers at %d threads",
                         Setting::SUMS, MOST_REGISTERS, THREADS);
                entry_val.not_built = reason_list;
            }
            else
            {
                entry_val.infos_list[0][0] = read_kernel_info<Setting, SWIZZLE, false, false>();
                entry_val.infos_list[0][1] = read_kernel_info<Setting, SWIZZLE, false, true>();
                entry_val.infos_list[1][1] = read_kernel_info<Setting, SWIZZLE, true, true>();
            }
        }
    }
}

/// Fills the entries of one square, step of k, block size and warps down: swizzled and not.
template <int BLOCK_ROWS, int BLOCK_COLUMNS, int BLOCK_K, int THREADS, int WARPS_DOWN>
static void fill_swizzles(void)
{
    static_assert(COUNT(SWIZZLE_LIST) == 2, "one line a swizzle below, in SWIZZLE_LIST's order");
    fill_kernel<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, THREADS, WARPS_DOWN, true>();
    fill_kernel<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, THREADS, WARPS_DOWN, false>();
}

/// Fills the entries of one square, step of k and block size: every count of warps down.
template <int BLOCK_ROWS, int BLOCK_COLUMNS, int BLOCK_K, int THREADS>
static void fill_warps_down(void)
{
    static_assert(COUNT(WARPS_DOWN_LIST) == 4, "one line a count of warps down below");
    fill_swizzles<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, THREADS, 1>();
    fill_swizzles<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, THREADS, 2>();
    fill_swizzles<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, THREADS, 4>();
    fill_swizzles<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, THREADS, 8>();
}

/// Fills the entries of one square and step of k: every block size.
template <int BLOCK_ROWS, int BLOCK_COLUMNS, int BLOCK_K>
static void fill_threads(void)
{
    static_assert(COUNT(THREADS_LIST) == 6, "one line a block size below");
    fill_warps_down<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, 32>();
    fill_warps_down<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, 64>();
    fill_warps_down<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, 128>();
    fill_warps_down<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, 256>();
    fill_warps_down<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, 512>();
    fill_warps_down<BLOCK_ROWS, BLOCK_COLUMNS, BLOCK_K, 1024>();
}

/// Fills the entries of one square: every step of k.
template <int BLOCK_ROWS, int BLOCK_COLUMNS>
static void fill_steps(void)
{
    static_assert(COUNT(BLOCK_K_LIST) == 3, "one line a step of k below");
    fill_threads<BLOCK_ROWS, BLOCK_COLUMNS, 16>();
    fill_threads<BLOCK_ROWS, BLOCK_COLUMNS, 32>();
    fill_threads<BLOCK_ROWS, BLOCK_COLUMNS, 64>();
}

/// Fills the entries of one count of rows: every count of columns.
template <int BLOCK_ROWS>
static void fill_columns(void)
{
    static_assert(COUNT(BLOCK_COLUMNS_LIST) == 3, "one line a count of columns below");
    fill_steps<BLOCK_ROWS, 64>();
    fill_steps<BLOCK_ROWS, 128>();
    fill_steps<BLOCK_ROWS, 256>();
}

/// Fills kernels_list: every kernel the sweep can build.
static void fill_kernels(void)
{
    static_assert(COUNT(BLOCK_ROWS_LIST) == 3, "one line a count of rows below");
    fill_columns<64>();
    fill_columns<128>();
    fill_columns<256>();
}

/// One setting of the sweep for one product, and its times.
struct Setting
{
    int block_rows_index_val, block_columns_index_val, block_k_index_val, threads_index_val, warps_down_index_val, swizzle_index_val;   // the kernel
    int parts_val, blocks_per_sm_val, block_order_val;   // given at launch; blocks_per_sm 0: one a square
    int resident_val = 0;             // its blocks an SM holds at once, at its parts' shared memory
    std::string skipped_val;          // empty: it ran; else why it was skipped
    float first_ms_val = 0.0f;
    int first_runs_val = 0;
    bool same_val = false;            // mma_matmul_bf16's bits
    float final_ms_val = 0.0f;

    /// Gives its kernel's entry.
    const KernelEntry &entry(void) const
    {
        return kernels_list[block_rows_index_val][block_columns_index_val][block_k_index_val][threads_index_val][warps_down_index_val]
                           [swizzle_index_val];
    }
};

/// Gives a setting as one line: its square, step of k, threads and warps, swizzle, parts, blocks and their order.
///
///   setting_val   the setting
static std::string describe(const Setting &setting_val)
{
    char blocks_list[24];
    if (setting_val.blocks_per_sm_val == 0)
        snprintf(blocks_list, sizeof(blocks_list), "one a square");
    else
        snprintf(blocks_list, sizeof(blocks_list), "%d an SM", setting_val.blocks_per_sm_val);
    char line_list[220];
    snprintf(line_list, sizeof(line_list), "square %3d x %3d, k %2d, %4d threads (warps %d x %2d), %-10s, %d parts, blocks %-12s, order %2d  (%2d on an SM)",
             BLOCK_ROWS_LIST[setting_val.block_rows_index_val], BLOCK_COLUMNS_LIST[setting_val.block_columns_index_val],
             BLOCK_K_LIST[setting_val.block_k_index_val], THREADS_LIST[setting_val.threads_index_val],
             WARPS_DOWN_LIST[setting_val.warps_down_index_val], setting_val.entry().warps_across,
             SWIZZLE_LIST[setting_val.swizzle_index_val] ? "swizzled" : "no swizzle", setting_val.parts_val, blocks_list,
             setting_val.block_order_val, setting_val.resident_val);
    return line_list;
}

/// Makes one point of the sweep a setting for one product, and why it is skipped if it is.
///
///   point_val     the point: a value of each list, read by SweepList (SWEEP_BLOCK_ROWS, ...)
///   product_val   the product
///   sms_val       this GPU's SMs
static Setting plan_setting(const TunePoint &point_val, const MatmulProduct &product_val, int sms_val)
{
    // Skipped (skipped_val says why): the kernel was not built (its warps do not split the square, or its sums need
    // more registers than a thread has), CUDA refused it, it spills, its parts do not fit a block's shared memory, no
    // block of it fits on an SM, or its blocks reach the squares (the same as one a square). Each reason names only
    // what caused it, so the log can count the settings under each. Else skipped_val stays empty.
    Setting setting_val;
    setting_val.block_rows_index_val = point_val.index(SWEEP_BLOCK_ROWS);
    setting_val.block_columns_index_val = point_val.index(SWEEP_BLOCK_COLUMNS);
    setting_val.block_k_index_val = point_val.index(SWEEP_BLOCK_K);
    setting_val.threads_index_val = point_val.index(SWEEP_THREADS);
    setting_val.warps_down_index_val = point_val.index(SWEEP_WARPS_DOWN);
    setting_val.swizzle_index_val = point_val.index(SWEEP_SWIZZLE);
    setting_val.parts_val = point_val.value(SWEEP_PARTS);
    setting_val.blocks_per_sm_val = point_val.value(SWEEP_BLOCKS);
    setting_val.block_order_val = point_val.value(SWEEP_BLOCK_ORDER);
    const KernelEntry &entry_val = setting_val.entry();
    const KernelInfo &info_val = entry_val.infos_list[product_val.left_turned][product_val.right_turned];
    char kernel_list[96];
    snprintf(kernel_list, sizeof(kernel_list), "square %d x %d, k %d, %d threads, %d down, %s", point_val.value(SWEEP_BLOCK_ROWS),
             point_val.value(SWEEP_BLOCK_COLUMNS), point_val.value(SWEEP_BLOCK_K), point_val.value(SWEEP_THREADS),
             point_val.value(SWEEP_WARPS_DOWN), point_val.value(SWEEP_SWIZZLE) ? "swizzled" : "no swizzle");
    char reason_list[240] = "";
    size_t shared_bytes_val = (size_t)setting_val.parts_val * entry_val.part_bytes + MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES;   // the parts, and room to start them on an atom
    size_t squares_val = product_val.squares(point_val.value(SWEEP_BLOCK_ROWS), point_val.value(SWEEP_BLOCK_COLUMNS));
    if (!entry_val.not_built.empty())
        snprintf(reason_list, sizeof(reason_list), "%s", entry_val.not_built.c_str());
    else if (!info_val.error.empty())
        snprintf(reason_list, sizeof(reason_list), "kernel refused: %s: %s", kernel_list, info_val.error.c_str());
    else if (info_val.local_bytes > 0)
        snprintf(reason_list, sizeof(reason_list), "kernel spills: %s: %zu bytes a thread", kernel_list, info_val.local_bytes);
    else if (shared_bytes_val > (size_t)MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES)
        snprintf(reason_list, sizeof(reason_list), "does not fit shared memory: %d parts of %zu B, a block has %d", setting_val.parts_val,
                 entry_val.part_bytes, MMA_TMA_MATMUL_BF16_MAX_SHARED_BYTES);
    else if (setting_val.blocks_per_sm_val > 0 && (size_t)setting_val.blocks_per_sm_val * sms_val >= squares_val)
        snprintf(reason_list, sizeof(reason_list), "blocks reach the squares: %d an SM is %d blocks, %zu squares (the same as one a square)",
                 setting_val.blocks_per_sm_val, setting_val.blocks_per_sm_val * sms_val, squares_val);
    else
    {
        // a block of it fits on an SM, at its parts' shared memory; how many at once
        int blocks_on_sm_val = 0;
        cudaError_t error_val = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_on_sm_val, info_val.kernel_ptr,
                                                                              point_val.value(SWEEP_THREADS), shared_bytes_val);
        cudaGetLastError();
        if (error_val != cudaSuccess || blocks_on_sm_val == 0)
            snprintf(reason_list, sizeof(reason_list), "no room on an SM: %s, %d registers a thread, %zu B shared%s%s", kernel_list,
                     info_val.registers, shared_bytes_val, error_val != cudaSuccess ? ": " : "",
                     error_val != cudaSuccess ? cudaGetErrorString(error_val) : "");
        setting_val.resident_val = setting_val.blocks_per_sm_val > 0 ? std::min(blocks_on_sm_val, setting_val.blocks_per_sm_val)
                                                                     : blocks_on_sm_val;
    }
    setting_val.skipped_val = reason_list;
    return setting_val;
}

/// Gives the sweep's lists, father to son, in SweepList's order.
static std::vector<TuneList> sweep_lists(void)
{
    return {
        {"block_rows", BLOCK_ROWS_LIST},         // SWEEP_BLOCK_ROWS: rows of output a block makes
        {"block_columns", BLOCK_COLUMNS_LIST},   // SWEEP_BLOCK_COLUMNS: columns of it
        {"block_k", BLOCK_K_LIST},               // SWEEP_BLOCK_K: values of k a step copies
        {"threads", THREADS_LIST},               // SWEEP_THREADS: threads a block
        {"warps_down", WARPS_DOWN_LIST},         // SWEEP_WARPS_DOWN: the rest across
        {"swizzle", SWIZZLE_LIST},               // SWEEP_SWIZZLE: 1 TMA's swizzle, 0 the rows as they sit
        {"parts", PARTS_LIST},                   // SWEEP_PARTS: parts of shared memory, a step's tiles each
        {"blocks_per_sm", BLOCKS_PER_SM_LIST},   // SWEEP_BLOCKS: blocks a launch per SM; 0 one a square
        {"block_order", BLOCK_ORDER_LIST},       // SWEEP_BLOCK_ORDER: rows of squares a group of blocks runs down
    };
}

// ── one setting, run ──────────────────────────────────────────────────────────

/// Launches one setting on a product's operands, into its output.
///
///   data_val      the product's operands and output
///   setting_val   the setting
///   sms_val       this GPU's SMs
static cudaError_t launch_setting(ProductData &data_val, const Setting &setting_val, int sms_val)
{
    const MatmulProduct &product_val = data_val.product;
    LaunchProduct launch_ptr = setting_val.entry().infos_list[product_val.left_turned][product_val.right_turned].launch_ptr;
    return launch_ptr(data_val.output_tensor.data_ptr<nv_bfloat16>(), data_val.left_tensor.data_ptr<nv_bfloat16>(),
                      data_val.right_tensor.data_ptr<nv_bfloat16>(), data_val.bias_ptr(), product_val.rows, product_val.columns,
                      product_val.sum_length, product_val.add_onto, setting_val.block_order_val, setting_val.parts_val,
                      setting_val.blocks_per_sm_val * sms_val);
}

/// What came of running one setting: checked and timed, its launch refused, or it failed while running.
enum RunOutcome { RUN_TIMED, RUN_REFUSED, RUN_FAILED };

/// Runs one setting from 0s, checks it against mma_matmul_bf16's bits, then times it: the first pass's way.
///
///   data_val                the product's operands, output and expected bits
///   setting_val             the setting: same_val, first_ms_val and first_runs_val written, or skipped_val with why not
///   sms_val                 this GPU's SMs
///   defaults_first_ms_val   the defaults' time: a setting SLOW_CUT times it is timed once
static RunOutcome check_and_time(ProductData &data_val, Setting &setting_val, int sms_val, float defaults_first_ms_val)
{
    // Checked to the last bit (the same mma on the same values in the same order of k), then timed FIRST_PASS_RUNS
    // times, averaged. A launch CUDA refuses: RUN_REFUSED, nothing broken. An error while it runs (a bad address)
    // breaks the GPU for every later call: RUN_FAILED, and the caller can only stop.
    data_val.output_tensor.zero_();
    cudaError_t error_val = launch_setting(data_val, setting_val, sms_val);
    if (error_val != cudaSuccess)
    {
        setting_val.skipped_val = std::string("launch refused: ") + cudaGetErrorString(error_val);
        return RUN_REFUSED;
    }
    error_val = cudaDeviceSynchronize();
    if (error_val != cudaSuccess)
    {
        setting_val.skipped_val = std::string("failed while running: ") + cudaGetErrorString(error_val);
        return RUN_FAILED;
    }
    setting_val.same_val = data_val.same();
    float ms_val = gpu_milliseconds([&] { launch_setting(data_val, setting_val, sms_val); });
    setting_val.first_runs_val = 1;
    if (ms_val < SLOW_CUT * defaults_first_ms_val)
        for (; setting_val.first_runs_val < FIRST_PASS_RUNS; setting_val.first_runs_val++)
            ms_val += gpu_milliseconds([&] { launch_setting(data_val, setting_val, sms_val); });
    setting_val.first_ms_val = ms_val / setting_val.first_runs_val;
    return RUN_TIMED;
}

/// Times settings again over SWEEP_ROUNDS rounds, taking turns with the defaults and cuBLAS: the second pass.
///
///   data_val          the product's operands and output
///   finalists_list    the settings: each one's final_ms_val written
///   sms_val           this GPU's SMs
///   defaults_ms_ptr   written: the defaults' time
///   cublas_ms_ptr     written: PyTorch's (cuBLAS)
static void time_finalists(ProductData &data_val, const std::vector<Setting *> &finalists_list, int sms_val, float *defaults_ms_ptr,
                           float *cublas_ms_ptr)
{
    // cuBLAS warmed up first: its first call (its handle, its workspace) is not a product's time
    warm_up_gpu([&] { data_val.run_cublas(); });
    std::vector<float> total_ms_list(finalists_list.size(), 0.0f);
    float defaults_total_ms_val = 0.0f, cublas_total_ms_val = 0.0f;
    for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
    {
        defaults_total_ms_val += gpu_milliseconds([&] { data_val.run_defaults(); });
        cublas_total_ms_val += gpu_milliseconds([&] { data_val.run_cublas(); });
        for (size_t finalist_index_val = 0; finalist_index_val < finalists_list.size(); finalist_index_val++)
            total_ms_list[finalist_index_val] += gpu_milliseconds([&] { launch_setting(data_val, *finalists_list[finalist_index_val], sms_val); });
    }
    for (size_t finalist_index_val = 0; finalist_index_val < finalists_list.size(); finalist_index_val++)
        finalists_list[finalist_index_val]->final_ms_val = total_ms_list[finalist_index_val] / SWEEP_ROUNDS;
    *defaults_ms_ptr = defaults_total_ms_val / SWEEP_ROUNDS;
    *cublas_ms_ptr = cublas_total_ms_val / SWEEP_ROUNDS;
}

// ══ the tune ══════════════════════════════════════════════════════════════════

/// One product's result, for the table at the end.
struct ProductResult
{
    std::string name;
    float defaults_ms = 0.0f, cublas_ms = 0.0f, best_ms = 0.0f;   // the second pass's
    double flops = 0.0;
    std::string best_setting;
    size_t not_same = 0;
};

/// Sweeps every setting of one product, father to son, prints its tables and finalists, and writes its rows to the .csv.
///
///   product_val   the product
///   sms_val       this GPU's SMs
///   csv_ptr       the .csv every setting's row goes to; nullptr: none
static ProductResult time_every_setting(const MatmulProduct &product_val, int sms_val, FILE *csv_ptr)
{
    // Every one with every other, father to son: the kernel built (the square's rows and columns, the step of k, the
    // threads a block, the warps down, the rest across, the swizzle), then what a launch is given: the parts of shared
    // memory (1: TMA brings a step, wait, multiply; n: n steps asked for ahead, a step done frees its part), the
    // blocks (one a square, or B an SM each walking squares) and their order.
    //
    // A setting that cannot run is skipped, never the end of the tune (plan_setting says why; a launch CUDA refuses
    // too). The log counts the settings under each reason; the .csv has every setting with its own.
    //
    // First pass, in that order: each setting run once from 0s and checked against mma_matmul_bf16's output (the
    // threads' copies), to the last bit (the same mma on the same values in the same order of k), then timed
    // FIRST_PASS_RUNS times (once if slower than SLOW_CUT x the defaults). Second pass: the FINALISTS fastest, the
    // defaults and PyTorch's matmul (cuBLAS), timed over SWEEP_ROUNDS rounds taking turns.
    //
    // Expect: every setting gives mma_matmul_bf16's bits.

    // 1. the tensors, mma_matmul_bf16's bits, the defaults' time
    ProductData data_val(product_val);
    warm_up_gpu([&] { data_val.run_defaults(); });
    float defaults_first_ms_val = 0.0f;
    for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
        defaults_first_ms_val += gpu_milliseconds([&] { data_val.run_defaults(); });
    defaults_first_ms_val /= SWEEP_ROUNDS;

    // 2. every setting, father to son (for_every_setting); the ones that cannot run skipped, with why
    const std::vector<TuneList> tunes_list = sweep_lists();
    std::vector<Setting> settings_list;
    std::vector<Setting> skipped_list;
    for_every_setting(tunes_list, [&](const TunePoint &point_val) {
        Setting setting_val = plan_setting(point_val, product_val, sms_val);
        (setting_val.skipped_val.empty() ? settings_list : skipped_list).push_back(setting_val);
    });
    printf("\n%s: output {%zu, %zu}, sum of %zu%s%s%s: %.2f TFLOP\n", product_val.name.c_str(), product_val.rows, product_val.columns,
           product_val.sum_length, product_val.left_turned ? ", left turned" : "", product_val.right_turned ? ", right turned" : "",
           product_val.add_onto ? ", added onto" : (product_val.has_bias ? ", + bias" : ""), product_val.flops() / 1e12);
    printf("    %zu settings in the sweep, %zu to run, %zu skipped\n", count_settings(tunes_list), settings_list.size(), skipped_list.size());

    // 3. first pass: check, then time; a launch CUDA refuses skipped, with its error
    auto start_val = std::chrono::steady_clock::now();
    std::vector<Setting> ran_list;
    for (size_t setting_index_val = 0; setting_index_val < settings_list.size(); setting_index_val++)
    {
        Setting &setting_val = settings_list[setting_index_val];
        RunOutcome outcome_val = check_and_time(data_val, setting_val, sms_val, defaults_first_ms_val);
        if (outcome_val == RUN_REFUSED)
        {
            skipped_list.push_back(setting_val);
            continue;
        }
        // an error while it runs breaks the GPU for every later call: nothing to skip to, so stop here
        if (outcome_val == RUN_FAILED)
        {
            printf("    stopped: %s %s; the GPU takes no more work after this\n", describe(setting_val).c_str(), setting_val.skipped_val.c_str());
            exit(EXIT_FAILURE);
        }
        ran_list.push_back(setting_val);
        if ((setting_index_val + 1) % PROGRESS_EVERY == 0)
            printf("    ... %zu of %zu settings, %.0f s\n", setting_index_val + 1, settings_list.size(),
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - start_val).count());
    }
    settings_list = ran_list;   // from here on, the settings that ran

    // 4. second pass: the finalists, the defaults and cuBLAS, taking turns
    std::vector<int> order_list(settings_list.size());
    for (size_t index_val = 0; index_val < order_list.size(); index_val++)
        order_list[index_val] = (int)index_val;
    std::sort(order_list.begin(), order_list.end(), [&](int first_val, int second_val) {
        return settings_list[first_val].first_ms_val < settings_list[second_val].first_ms_val;
    });
    int finalists_val = std::min((int)order_list.size(), FINALISTS);
    std::vector<Setting *> finalists_list;
    for (int finalist_index_val = 0; finalist_index_val < finalists_val; finalist_index_val++)
        finalists_list.push_back(&settings_list[order_list[finalist_index_val]]);
    float defaults_ms_val = 0.0f, cublas_ms_val = 0.0f;
    time_finalists(data_val, finalists_list, sms_val, &defaults_ms_val, &cublas_ms_val);
    double flops_val = product_val.flops();
    double peak_val = peak_bf16_flops();
    auto tflops_val = [&](float ms_val) { return flops_val / ms_val / TERA_IN_MILLISECONDS; };
    printf("    the defaults: %.4f ms, %.0f TFLOPS (MFU %.0f%%); PyTorch (cuBLAS): %.4f ms, %.0f TFLOPS (MFU %.0f%%)\n", defaults_ms_val,
           tflops_val(defaults_ms_val), 100.0 * flops_val / (defaults_ms_val / 1e3) / peak_val, cublas_ms_val, tflops_val(cublas_ms_val),
           100.0 * flops_val / (cublas_ms_val / 1e3) / peak_val);

    // 5. the settings skipped: each reason, with the settings under it
    printf("    %zu settings ran, %zu skipped\n", settings_list.size(), skipped_list.size());
    if (!skipped_list.empty())
    {
        std::map<std::string, long> reasons_list;
        for (const Setting &setting_val : skipped_list)
            reasons_list[setting_val.skipped_val]++;
        printf("    skipped, each reason (settings under it: why)\n");
        for (const auto &reason_val : reasons_list)
            printf("      %6ld  %s\n", reason_val.second, reason_val.first.c_str());
    }

    // 6. five tables, the fastest setting in each cell (first pass) as x the defaults (above 1: faster)
    auto fastest_val = [&](auto matches_val) {
        int best_val = -1;
        for (size_t index_val = 0; index_val < settings_list.size(); index_val++)
            if (matches_val(settings_list[index_val]) && (best_val < 0 || settings_list[index_val].first_ms_val < settings_list[best_val].first_ms_val))
                best_val = (int)index_val;
        return best_val;
    };
    auto print_cell_val = [&](int best_val) {
        if (best_val < 0)
            printf("  %7s", "-");
        else
            printf("  %6.2fx", defaults_first_ms_val / settings_list[best_val].first_ms_val);
    };
    //    a. the square's rows against its columns
    printf("    the square: rows \\ columns (first pass, x the defaults)\n               ");
    for (int block_columns_val : BLOCK_COLUMNS_LIST)
        printf("  %7d", block_columns_val);
    printf("\n");
    for (int rows_index_val = 0; rows_index_val < COUNT(BLOCK_ROWS_LIST); rows_index_val++)
    {
        printf("    %11d", BLOCK_ROWS_LIST[rows_index_val]);
        for (int columns_index_val = 0; columns_index_val < COUNT(BLOCK_COLUMNS_LIST); columns_index_val++)
            print_cell_val(fastest_val([&](const Setting &s) {
                return s.block_rows_index_val == rows_index_val && s.block_columns_index_val == columns_index_val;
            }));
        printf("\n");
    }
    //    b. the threads a block against the warps down
    printf("    the block: threads \\ warps down (first pass, x the defaults)\n               ");
    for (int warps_down_val : WARPS_DOWN_LIST)
        printf("  %7d", warps_down_val);
    printf("\n");
    for (int threads_index_val = 0; threads_index_val < COUNT(THREADS_LIST); threads_index_val++)
    {
        printf("    %11d", THREADS_LIST[threads_index_val]);
        for (int down_index_val = 0; down_index_val < COUNT(WARPS_DOWN_LIST); down_index_val++)
            print_cell_val(fastest_val([&](const Setting &s) {
                return s.threads_index_val == threads_index_val && s.warps_down_index_val == down_index_val;
            }));
        printf("\n");
    }
    //    c. the blocks an SM against the parts: 1 part waits for each step; n keep n - 1 steps' copies in flight
    printf("    blocks an SM \\ parts (first pass, x the defaults; blocks 0: one a square)\n               ");
    for (int parts_val : PARTS_LIST)
        printf("  %7d", parts_val);
    printf("\n");
    for (int blocks_per_sm_val : BLOCKS_PER_SM_LIST)
    {
        printf("    %11d", blocks_per_sm_val);
        for (int parts_val : PARTS_LIST)
            print_cell_val(fastest_val([&](const Setting &s) { return s.blocks_per_sm_val == blocks_per_sm_val && s.parts_val == parts_val; }));
        printf("\n");
    }
    //    d. the swizzle against the parts
    printf("    swizzle \\ parts (first pass, x the defaults)\n               ");
    for (int parts_val : PARTS_LIST)
        printf("  %7d", parts_val);
    printf("\n");
    for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
    {
        printf("    %11s", SWIZZLE_LIST[swizzle_index_val] ? "swizzled" : "none");
        for (int parts_val : PARTS_LIST)
            print_cell_val(fastest_val([&](const Setting &s) { return s.swizzle_index_val == swizzle_index_val && s.parts_val == parts_val; }));
        printf("\n");
    }
    //    e. the step of k against the block order
    printf("    the step of k \\ the block order (first pass, x the defaults)\n               ");
    for (int block_order_val : BLOCK_ORDER_LIST)
        printf("  %7d", block_order_val);
    printf("\n");
    for (int k_index_val = 0; k_index_val < COUNT(BLOCK_K_LIST); k_index_val++)
    {
        printf("    %11d", BLOCK_K_LIST[k_index_val]);
        for (int block_order_val : BLOCK_ORDER_LIST)
            print_cell_val(fastest_val([&](const Setting &s) { return s.block_k_index_val == k_index_val && s.block_order_val == block_order_val; }));
        printf("\n");
    }

    // 7. the finalists, timed again taking turns with the defaults and cuBLAS
    std::vector<int> final_order_list(order_list.begin(), order_list.begin() + finalists_val);
    std::sort(final_order_list.begin(), final_order_list.end(),
              [&](int first_val, int second_val) { return settings_list[first_val].final_ms_val < settings_list[second_val].final_ms_val; });
    printf("    the %d fastest, timed again over %d rounds with the defaults and cuBLAS\n", TOP_PRINTED, SWEEP_ROUNDS);
    printf("         ms  TFLOPS  x defaults  x cuBLAS   setting\n");
    for (int rank_val = 0; rank_val < std::min(TOP_PRINTED, finalists_val); rank_val++)
    {
        const Setting &setting_val = settings_list[final_order_list[rank_val]];
        printf("    %7.4f  %6.0f     %5.2fx    %5.2fx   %s\n", setting_val.final_ms_val, tflops_val(setting_val.final_ms_val),
               defaults_ms_val / setting_val.final_ms_val, cublas_ms_val / setting_val.final_ms_val, describe(setting_val).c_str());
    }

    // 8. every setting into the .csv: the ones that ran, then the skipped ones (no times; status: why, in quotes)
    if (csv_ptr != nullptr)
        for (const std::vector<Setting> *list_ptr : {&settings_list, &skipped_list})
            for (const Setting &setting_val : *list_ptr)
            {
                const KernelEntry &entry_val = setting_val.entry();
                const KernelInfo &info_val = entry_val.infos_list[product_val.left_turned][product_val.right_turned];
                bool ran_val = setting_val.skipped_val.empty();
                fprintf(csv_ptr, "\"%s\",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%.5f,%d,%.5f,%.1f,%d,\"%s\"\n", product_val.name.c_str(),
                        BLOCK_ROWS_LIST[setting_val.block_rows_index_val], BLOCK_COLUMNS_LIST[setting_val.block_columns_index_val],
                        BLOCK_K_LIST[setting_val.block_k_index_val], THREADS_LIST[setting_val.threads_index_val],
                        WARPS_DOWN_LIST[setting_val.warps_down_index_val], entry_val.warps_across,
                        SWIZZLE_LIST[setting_val.swizzle_index_val] ? 1 : 0, setting_val.parts_val,
                        setting_val.blocks_per_sm_val, setting_val.resident_val, setting_val.block_order_val, info_val.registers,
                        entry_val.sums, info_val.local_bytes, (size_t)setting_val.parts_val * entry_val.part_bytes + MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES,
                        setting_val.first_ms_val,
                        setting_val.first_runs_val, setting_val.final_ms_val, ran_val ? tflops_val(setting_val.first_ms_val) : 0.0,
                        setting_val.same_val ? 1 : 0, ran_val ? "ran" : setting_val.skipped_val.c_str());
            }

    // 9. any setting that did not give mma_matmul_bf16's bits, by name (checked once, at the end of the tune)
    ProductResult result_val;
    for (const Setting &setting_val : settings_list)
        if (!setting_val.same_val)
        {
            printf("    NOT mma_matmul_bf16's bits: %s\n", describe(setting_val).c_str());
            result_val.not_same++;
        }
    result_val.name = product_val.name;
    result_val.flops = flops_val;
    result_val.defaults_ms = defaults_ms_val;
    result_val.cublas_ms = cublas_ms_val;
    if (finalists_val > 0)
    {
        const Setting &best_setting_val = settings_list[final_order_list[0]];
        result_val.best_ms = best_setting_val.final_ms_val;
        result_val.best_setting = describe(best_setting_val);
    }
    return result_val;
}

/// Prints each kernel built: its warps, a part's shared memory, its sums, and its registers for each pair of turned flags.
static void print_kernels(void)
{
    printf("each kernel: registers a thread for each pair of turned flags (S: spills; X: refused; its settings skipped)\n");
    printf("    square     k  threads  warps   swizzle    part B  sums   neither turned  right turned  both turned\n");
    for (int rows_index_val = 0; rows_index_val < COUNT(BLOCK_ROWS_LIST); rows_index_val++)
        for (int columns_index_val = 0; columns_index_val < COUNT(BLOCK_COLUMNS_LIST); columns_index_val++)
            for (int k_index_val = 0; k_index_val < COUNT(BLOCK_K_LIST); k_index_val++)
                for (int threads_index_val = 0; threads_index_val < COUNT(THREADS_LIST); threads_index_val++)
                    for (int down_index_val = 0; down_index_val < COUNT(WARPS_DOWN_LIST); down_index_val++)
                        for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
                    {
                        const KernelEntry &entry_val = kernels_list[rows_index_val][columns_index_val][k_index_val][threads_index_val][down_index_val]
                                                                   [swizzle_index_val];
                        printf("    %3d x %3d  %2d  %7d  %d x %2d  %-8s  %7zu  %4d", BLOCK_ROWS_LIST[rows_index_val], BLOCK_COLUMNS_LIST[columns_index_val],
                               BLOCK_K_LIST[k_index_val], THREADS_LIST[threads_index_val], WARPS_DOWN_LIST[down_index_val], entry_val.warps_across,
                               SWIZZLE_LIST[swizzle_index_val] ? "swizzled" : "none", entry_val.part_bytes, entry_val.sums);
                        if (!entry_val.not_built.empty())
                        {
                            printf("   not built: %s\n", entry_val.not_built.c_str());
                            continue;
                        }
                        const KernelInfo *infos_list[3] = {&entry_val.infos_list[0][0], &entry_val.infos_list[0][1], &entry_val.infos_list[1][1]};
                        for (const KernelInfo *info_ptr : infos_list)
                            printf("   %3d regs %c   ", info_ptr->registers, !info_ptr->error.empty() ? 'X' : info_ptr->local_bytes > 0 ? 'S' : ' ');
                        printf("\n");
                    }
}

// ══ serve: the settings tune_mma_tma_matmul_bf16.py asks for ═════════════════

/// Splits a line at its tabs.
///
///   line_val   the line
static std::vector<std::string> split_tabs(const std::string &line_val)
{
    std::vector<std::string> fields_list;
    std::stringstream stream_val(line_val);
    std::string field_val;
    while (std::getline(stream_val, field_val, '\t'))
        fields_list.push_back(field_val);
    return fields_list;
}

/// Answers tune_mma_tma_matmul_bf16.py's requests on stdin, one a line, until quit or stdin ends.
///
///   products_list   the products it can pick
///   sms_val         this GPU's SMs
///   replies_ptr     where the replies go: the real stdout (all else printed goes to stderr)
static void serve(const std::vector<MatmulProduct> &products_list, int sms_val, FILE *replies_ptr)
{
    // A line each way, its fields split by tabs, the first saying what it is:
    //
    //     asked              replied
    //     (at the start)     ready  sms  gpu_folder  products; a product line each; end
    //                        product  index  name  rows  columns  sum_length  left_turned  right_turned  add_onto  has_bias  flops
    //     select  index      a space line for each setting that can run; selected  index  defaults_ms  can_run  skipped
    //                        space  id  block_rows  block_columns  block_k  threads  warps_down  warps_across  swizzle  parts
    //                               blocks_per_sm  resident_per_sm  block_order  registers  sums  shared_bytes  squares
    //     run  id            ran  id  first_ms  first_runs  same; or refused  id  why; or failed  id  why, and the end
    //     final  id,id,...   final  id  final_ms for each; final_end  defaults_ms  cublas_ms
    //     quit               none: the end
    //
    // id: the setting's place in the sweep, father to son (for_every_setting), the same on every run of one build.
    // run is check_and_time, final time_finalists: the same check and timings as the sweep. A setting that fails
    // while running leaves the GPU taking no more work: the reply says so and the server ends; the driver starts
    // another. A request it cannot answer: error  why.
    const std::vector<TuneList> tunes_list = sweep_lists();
    fprintf(replies_ptr, "ready\t%d\t%s\t%zu\n", sms_val, gpu_folder_name().c_str(), products_list.size());
    for (size_t product_index_val = 0; product_index_val < products_list.size(); product_index_val++)
    {
        const MatmulProduct &product_val = products_list[product_index_val];
        fprintf(replies_ptr, "product\t%zu\t%s\t%zu\t%zu\t%zu\t%d\t%d\t%d\t%d\t%.0f\n", product_index_val, product_val.name.c_str(),
                product_val.rows, product_val.columns, product_val.sum_length, product_val.left_turned ? 1 : 0, product_val.right_turned ? 1 : 0,
                product_val.add_onto ? 1 : 0, product_val.has_bias ? 1 : 0, product_val.flops());
    }
    fprintf(replies_ptr, "end\n");
    fflush(replies_ptr);

    std::unique_ptr<ProductData> data_ptr;   // the product selected: its operands, output and expected bits
    std::vector<Setting> space_list;         // its every setting, by id; skipped_val empty: it can run
    float defaults_first_ms_val = 0.0f;
    std::string line_val;
    while (std::getline(std::cin, line_val))
    {
        std::vector<std::string> fields_list = split_tabs(line_val);
        if (fields_list.empty())
            continue;
        const std::string &request_val = fields_list[0];
        if (request_val == "quit")
            return;
        if (request_val == "select" && fields_list.size() == 2)
        {
            // its tensors (the last product's freed first), the defaults' time, every setting planned
            size_t product_index_val = std::stoul(fields_list[1]);
            if (product_index_val >= products_list.size())
            {
                fprintf(replies_ptr, "error\tno product %zu\n", product_index_val);
                fflush(replies_ptr);
                continue;
            }
            const MatmulProduct &product_val = products_list[product_index_val];
            data_ptr.reset();
            data_ptr = std::make_unique<ProductData>(product_val);
            warm_up_gpu([&] { data_ptr->run_defaults(); });
            defaults_first_ms_val = 0.0f;
            for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
                defaults_first_ms_val += gpu_milliseconds([&] { data_ptr->run_defaults(); });
            defaults_first_ms_val /= SWEEP_ROUNDS;
            space_list.clear();
            size_t can_run_val = 0;
            for_every_setting(tunes_list, [&](const TunePoint &point_val) {
                Setting setting_val = plan_setting(point_val, product_val, sms_val);
                if (setting_val.skipped_val.empty())
                {
                    const KernelEntry &entry_val = setting_val.entry();
                    const KernelInfo &info_val = entry_val.infos_list[product_val.left_turned][product_val.right_turned];
                    fprintf(replies_ptr, "space\t%zu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%zu\t%zu\n", space_list.size(),
                            point_val.value(SWEEP_BLOCK_ROWS), point_val.value(SWEEP_BLOCK_COLUMNS), point_val.value(SWEEP_BLOCK_K),
                            point_val.value(SWEEP_THREADS), point_val.value(SWEEP_WARPS_DOWN), entry_val.warps_across,
                            point_val.value(SWEEP_SWIZZLE), setting_val.parts_val, setting_val.blocks_per_sm_val, setting_val.resident_val,
                            setting_val.block_order_val, info_val.registers, entry_val.sums,
                            (size_t)setting_val.parts_val * entry_val.part_bytes + MMA_TMA_MATMUL_BF16_PART_ALIGN_BYTES,
                            product_val.squares(point_val.value(SWEEP_BLOCK_ROWS), point_val.value(SWEEP_BLOCK_COLUMNS)));
                    can_run_val++;
                }
                space_list.push_back(setting_val);
            });
            fprintf(replies_ptr, "selected\t%zu\t%.5f\t%zu\t%zu\n", product_index_val, defaults_first_ms_val, can_run_val,
                    space_list.size() - can_run_val);
        }
        else if (request_val == "run" && fields_list.size() == 2)
        {
            // a copy: each run starts from the plan, not from an earlier run's times
            size_t setting_id_val = std::stoul(fields_list[1]);
            if (data_ptr == nullptr || setting_id_val >= space_list.size() || !space_list[setting_id_val].skipped_val.empty())
            {
                fprintf(replies_ptr, "error\tno setting %zu that can run\n", setting_id_val);
                fflush(replies_ptr);
                continue;
            }
            Setting setting_val = space_list[setting_id_val];
            RunOutcome outcome_val = check_and_time(*data_ptr, setting_val, sms_val, defaults_first_ms_val);
            if (outcome_val == RUN_TIMED)
                fprintf(replies_ptr, "ran\t%zu\t%.5f\t%d\t%d\n", setting_id_val, setting_val.first_ms_val, setting_val.first_runs_val,
                        setting_val.same_val ? 1 : 0);
            else
            {
                fprintf(replies_ptr, "%s\t%zu\t%s\n", outcome_val == RUN_REFUSED ? "refused" : "failed", setting_id_val,
                        setting_val.skipped_val.c_str());
                if (outcome_val == RUN_FAILED)
                {
                    fflush(replies_ptr);
                    return;   // the GPU takes no more work
                }
            }
        }
        else if (request_val == "final" && fields_list.size() == 2)
        {
            // the settings named, copies, timed taking turns with the defaults and cuBLAS
            std::vector<Setting> finalists_list;
            std::vector<size_t> setting_ids_list;
            std::stringstream ids_stream_val(fields_list[1]);
            std::string id_val;
            bool known_val = data_ptr != nullptr;
            while (known_val && std::getline(ids_stream_val, id_val, ','))
            {
                size_t setting_id_val = std::stoul(id_val);
                known_val = setting_id_val < space_list.size() && space_list[setting_id_val].skipped_val.empty();
                if (known_val)
                {
                    setting_ids_list.push_back(setting_id_val);
                    finalists_list.push_back(space_list[setting_id_val]);
                }
            }
            if (!known_val)
            {
                fprintf(replies_ptr, "error\tfinal: a setting that cannot run, or no product selected\n");
                fflush(replies_ptr);
                continue;
            }
            std::vector<Setting *> finalist_ptrs_list;
            for (Setting &setting_val : finalists_list)
                finalist_ptrs_list.push_back(&setting_val);
            float defaults_ms_val = 0.0f, cublas_ms_val = 0.0f;
            time_finalists(*data_ptr, finalist_ptrs_list, sms_val, &defaults_ms_val, &cublas_ms_val);
            for (size_t finalist_index_val = 0; finalist_index_val < finalists_list.size(); finalist_index_val++)
                fprintf(replies_ptr, "final\t%zu\t%.5f\n", setting_ids_list[finalist_index_val], finalists_list[finalist_index_val].final_ms_val);
            fprintf(replies_ptr, "final_end\t%.5f\t%.5f\n", defaults_ms_val, cublas_ms_val);
        }
        else
            fprintf(replies_ptr, "error\tnot a request: %s\n", line_val.c_str());
        fflush(replies_ptr);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    // ./tune_mma_tma_matmul_bf16: nanochat d24's 15 products; gpt2: GPT-2 small's; a layer's name (qkv,
    // attention_output, mlp_up, mlp_down, head): only those layers' products. Any order: ./tune_mma_tma_matmul_bf16 gpt2 head.
    // serve, with any of those: no sweep; the settings tune_mma_tma_matmul_bf16.py asks for (serve's comment).
    bool serve_val = false;
    std::vector<char *> arguments_list;
    for (int arg_index_val = 0; arg_index_val < argc; arg_index_val++)
        if (arg_index_val > 0 && std::string(argv[arg_index_val]) == "serve")
            serve_val = true;
        else
            arguments_list.push_back(argv[arg_index_val]);
    ModelSize size_val;
    std::vector<MatmulProduct> products_list = products_from_arguments((int)arguments_list.size(), arguments_list.data(), &size_val);
    if (serve_val)
    {
        // the replies on the real stdout; all else printed (the GPU, the kernels' table, CUDA's errors) on stderr
        int replies_fd_val = dup(STDOUT_FILENO);
        dup2(STDERR_FILENO, STDOUT_FILENO);
        FILE *replies_ptr = fdopen(replies_fd_val, "w");
        print_gpu();
        int serve_sms_val = 0;
        cudaCheck(cudaDeviceGetAttribute(&serve_sms_val, cudaDevAttrMultiProcessorCount, 0));
        fill_kernels();
        print_kernels();
        fflush(stdout);
        serve(products_list, serve_sms_val, replies_ptr);
        fclose(replies_ptr);
        return 0;
    }
    std::string folder_val = "cuda/tests/results/" + gpu_folder_name();
    std::string log_path_val = start_run_log(folder_val.c_str(), __FILE__, "tune_", "txt");
    print_gpu();
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    printf("%s: %zu rows, embed_dim %zu, qkv %zu, mlp %zu, vocab %zu, %s; %zu products\n", size_val.name_ptr, size_val.rows,
           size_val.embed_dim, size_val.qkv_dim, size_val.mlp_dim, size_val.vocab_size,
           size_val.linear_bias ? "linear biases" : "no linear bias", products_list.size());
    fill_kernels();
    print_kernels();

    // every product, one after the other: its tensors made, swept, then freed
    std::string csv_path_val = log_path_val.substr(0, log_path_val.size() - 3) + "csv";
    FILE *csv_ptr = fopen(csv_path_val.c_str(), "w");
    if (csv_ptr != nullptr)
        fprintf(csv_ptr, "product,block_rows,block_columns,block_k,threads,warps_down,warps_across,swizzle,parts,blocks_per_sm,resident_per_sm,"
                         "block_order,registers,sums,spill_bytes,shared_bytes,first_ms,first_runs,final_ms,tflops,same,status\n");
    std::vector<ProductResult> results_list;
    for (const MatmulProduct &product_val : products_list)
        results_list.push_back(time_every_setting(product_val, sms_val, csv_ptr));
    if (csv_ptr != nullptr)
    {
        fclose(csv_ptr);
        printf("\nevery setting: %s\n", csv_path_val.c_str());
    }

    // every product side by side: the defaults, cuBLAS and the best setting (the second pass's times)
    print_test_name("every_product");
    printf("    %-28s  %8s  %8s  %8s  %10s  %8s   %s\n", "product", "defaults", "cuBLAS", "best", "x defaults", "x cuBLAS", "best setting");
    double defaults_total_ms_val = 0.0, cublas_total_ms_val = 0.0, best_total_ms_val = 0.0;
    size_t not_same_val = 0;
    for (const ProductResult &result_val : results_list)
    {
        printf("    %-28s  %8.4f  %8.4f  %8.4f  %9.2fx  %7.2fx   %s\n", result_val.name.c_str(), result_val.defaults_ms, result_val.cublas_ms,
               result_val.best_ms, result_val.defaults_ms / result_val.best_ms, result_val.cublas_ms / result_val.best_ms,
               result_val.best_setting.c_str());
        defaults_total_ms_val += result_val.defaults_ms;
        cublas_total_ms_val += result_val.cublas_ms;
        best_total_ms_val += result_val.best_ms;
        not_same_val += result_val.not_same;
    }
    printf("    %-28s  %8.4f  %8.4f  %8.4f  %9.2fx  %7.2fx\n", "all, once each", defaults_total_ms_val, cublas_total_ms_val, best_total_ms_val,
           defaults_total_ms_val / best_total_ms_val, cublas_total_ms_val / best_total_ms_val);

    // every setting that ran gave mma_matmul_bf16's bits (the ones that did not are printed above, by product)
    ASSERT_EQ((long)not_same_val, 0L);
    printf("\nall tunes done\n");
    return 0;
}
