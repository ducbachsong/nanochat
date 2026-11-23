// tune_adamw_tma_bf16.cu — the timings of cuda/kernels/bf16/adamw_tma_bf16.cuh: every setting, father to son, against adamw_bf16_kernel.
#include <torch/types.h>              // torch::randn, torch::equal; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <algorithm>                  // std::sort
#include <chrono>                     // the sweep's progress
#include <string>
#include <vector>
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gpu_folder_name
#include "../../kernels/bf16/adamw_bf16.cuh"
#include "../../kernels/bf16/adamw_tma_bf16.cuh"

/// AdamW's sizes.
struct AdamwSize
{
    const char *name_ptr;
    size_t params_in_slice, decayed_count;
};
static const AdamwSize GPT2_SIZE = {"GPT-2 small", 124439808, 124439040};   // tune_adamw_bf16's: the matrices' 124,439,040 decayed, then 768
static const AdamwSize D24_SIZE = {"nanochat d24", 733105152, 733105152};   // the README's count, all decayed: the split moves one multiply, not the time

#define LEARNING_RATE 6e-4f
#define BETA1 0.9f
#define BETA2 0.95f
#define EPS 1e-8f
#define WEIGHT_DECAY 0.1f
#define MAX_NORM 1.0f
#define STEP 1

// The sweep, every one with every other, father to son: blocks an SM (which sets each block's share of shared memory),
// threads a block, values a thread, who stores, the parts the share is cut into (a tile each: 1, the whole share one tile)
static constexpr int BLOCKS_PER_SM_LIST[] = {1, 2, 3, 4, 6, 8, 16, 32};
static constexpr int THREADS_LIST[] = {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024};   // threads a block: whole warps
#define WARPS_PACK 1, 2, 3, 4, 6, 8, 12, 16, 24, 32           // the kernels compiled, one a THREADS_LIST entry: its warps
static constexpr int WARPS_LIST[] = {WARPS_PACK};
static constexpr int VALUES_LIST[] = {4, 8, 16, 32};
static constexpr bool TMA_STORE_LIST[] = {true, false};
static constexpr int PARTS_LIST[] = {1, 2, 3, 4, 5, 6, 7, 8};
static_assert(sizeof(PARTS_LIST) / sizeof(PARTS_LIST[0]) == ADAMW_TMA_BF16_MAX_STAGES, "every part count the kernel takes");
enum SweepList { SWEEP_BLOCKS, SWEEP_THREADS, SWEEP_VALUES, SWEEP_STORE, SWEEP_PARTS };   // each list's place in time_every_setting's tunes_list, father to son: the same order there
static constexpr int BASELINE_BLOCKS_PER_SM_LIST[] = {0, 1, 2, 4, 8, 16, 32};   // adamw_bf16_kernel's launches; 0: one float4 a thread
static constexpr int BASELINE_THREADS_LIST[] = {128, 256, 512, 1024};
#define COUNT(list) ((int)(sizeof(list) / sizeof((list)[0])))

/// Tells if THREADS_LIST and WARPS_LIST name the same blocks, entry by entry: a threads index picks its warps' kernel.
constexpr bool threads_match_warps(void)
{
    if (COUNT(THREADS_LIST) != COUNT(WARPS_LIST))
        return false;
    for (int index_val = 0; index_val < COUNT(THREADS_LIST); index_val++)
        if (THREADS_LIST[index_val] != WARPS_LIST[index_val] * HARDWARE_THREADS_PER_WARP)
            return false;
    return true;
}
static_assert(threads_match_warps(), "THREADS_LIST[i] must be WARPS_LIST[i] x 32");
#define FIRST_PASS_RUNS 3             // timed runs of each setting in the first pass, averaged
#define SLOW_CUT 20.0f                // a setting this many times the current kernel's best: timed once in the first pass
#define FINALISTS 100                 // the first pass's fastest, timed again over SWEEP_ROUNDS rounds taking turns
#define TOP_PRINTED 20                // finalists printed
#define PROGRESS_EVERY 1000           // settings between progress lines
#define MEGA 1e6

using AdamwTmaKernel = void (*)(nv_bfloat16 *, const nv_bfloat16 *, nv_bfloat16 *, nv_bfloat16 *, float *, AdamwTmaBf16Step,
                                AdamwTmaBf16Tile);
static AdamwTmaKernel kernels_list[COUNT(VALUES_LIST)][COUNT(TMA_STORE_LIST)][COUNT(WARPS_LIST)];

/// One kernel's registers, and why its settings are skipped if they are.
struct KernelInfo
{
    int registers;
    size_t local_bytes;               // > 0: registers spilled to local memory, so its settings are skipped
    std::string error;                // not empty: CUDA would not raise its shared memory, so its settings are skipped
};
static KernelInfo infos_list[COUNT(VALUES_LIST)][COUNT(TMA_STORE_LIST)][COUNT(WARPS_LIST)];

/// One setting of the sweep, its tile, and its times.
struct Setting
{
    int blocks_per_sm_val, warps_index_val, values_index_val, store_index_val, stages_val;
    int share_bytes_val, tile_values_val;   // the block's share of shared memory, cut into stages_val tiles of tile_values_val
    int resident_val = 0;                   // blocks of it on an SM at once: blocks_per_sm_val, or fewer if threads or registers run out
    AdamwTmaBf16Tile tile_val = {};
    size_t shared_bytes_val = 0;
    std::string skipped_val;                // empty: it ran; else why it was skipped
    float first_ms_val = 0.0f;
    int first_runs_val = 0;
    bool same_val = false;
    float final_ms_val = 0.0f;
};

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
    bool floats_val = first_tensor.dtype() == DType::Float32;
    torch::TensorOptions ints_val = torch::TensorOptions().dtype(floats_val ? torch::kInt32 : torch::kInt16).device(torch::kCUDA);
    torch::Tensor first_val = torch::from_blob(first_tensor.data_ptr(), {(int64_t)first_tensor.numel()}, ints_val);
    torch::Tensor second_val = torch::from_blob(second_tensor.data_ptr(), {(int64_t)second_tensor.numel()}, ints_val);
    return torch::equal(first_val, second_val);
}

/// AdamW's tensors at one size, their start, this step's numbers, and adamw_update_bf16's step on them.
struct AdamwData
{
    AdamwSize size_val;
    bool has_master;
    Tensor start_master_tensor, start_params_tensor, start_m_tensor, start_v_tensor;
    Tensor expected_master_tensor, expected_params_tensor, expected_m_tensor, expected_v_tensor;
    Tensor master_tensor, params_tensor, grads_tensor, m_tensor, v_tensor, grad_norm_tensor;
    AdamwTmaBf16Step step_val;

    /// Makes the tensors and the expected result.
    ///
    ///   size_val         the counts
    ///   has_master_val   a float master: as training runs it
    AdamwData(const AdamwSize &size_val, bool has_master_val) : size_val(size_val), has_master(has_master_val)
    {
        // 1. a random master and its bfloat16 params, grads (their norm clipped), m and v 0 (step 1)
        size_t count_val = size_val.params_in_slice;
        torch::manual_seed(0);
        start_master_tensor = random_float_tensor(count_val, 0.02f);
        start_params_tensor = start_master_tensor.to(DType::BFloat16);
        Tensor grads_float_tensor = random_float_tensor(count_val, 1.0f);
        grads_tensor = grads_float_tensor.to(DType::BFloat16);
        start_m_tensor = Tensor::zeros({count_val}, DType::BFloat16);
        start_v_tensor = Tensor::zeros({count_val}, DType::BFloat16);
        float norm_val = torch::from_blob(grads_float_tensor.data(), {(int64_t)count_val},
                                          torch::TensorOptions().device(torch::kCUDA)).norm().item<float>();
        grad_norm_tensor = Tensor::empty({1});
        grad_norm_tensor.copy_from_cpu(&norm_val);
        master_tensor = start_master_tensor.clone();
        params_tensor = start_params_tensor.clone();
        m_tensor = start_m_tensor.clone();
        v_tensor = start_v_tensor.clone();
        step_val = adamw_tma_bf16_step(size_val.decayed_count, grad_norm_tensor.data(), MAX_NORM, LEARNING_RATE, BETA1, BETA2, EPS,
                                       WEIGHT_DECAY, STEP);
        // 2. adamw_update_bf16's step, kept
        adamw_update_bf16(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), m_tensor.data_ptr<nv_bfloat16>(),
                          v_tensor.data_ptr<nv_bfloat16>(), count_val, size_val.decayed_count, grad_norm_tensor.data(), MAX_NORM,
                          LEARNING_RATE, BETA1, BETA2, EPS, WEIGHT_DECAY, STEP, master());
        expected_master_tensor = master_tensor.clone();
        expected_params_tensor = params_tensor.clone();
        expected_m_tensor = m_tensor.clone();
        expected_v_tensor = v_tensor.clone();
        reset();
    }

    /// Gives the master to pass: nullptr with none.
    float *master(void) const { return has_master ? master_tensor.data() : nullptr; }

    /// Puts the master, params, m and v back to the start.
    void reset(void)
    {
        master_tensor.copy_(start_master_tensor);
        params_tensor.copy_(start_params_tensor);
        m_tensor.copy_(start_m_tensor);
        v_tensor.copy_(start_v_tensor);
    }

    /// Tells if the step gave adamw_update_bf16's bits: the master (if any), params, m and v.
    bool same(void) const
    {
        return (!has_master || same_bits(master_tensor, expected_master_tensor)) && same_bits(params_tensor, expected_params_tensor) &&
               same_bits(m_tensor, expected_m_tensor) && same_bits(v_tensor, expected_v_tensor);
    }

    /// Gives the bytes a step moves: 20 a value with a master, 14 with none.
    double bytes(void) const
    {
        // With a master: it, grads, m, v read (4 + 2 + 2 + 2), it, params, m, v written (4 + 2 + 2 + 2). With none:
        // params, grads, m, v read (8), params, m, v written (6).
        return (has_master ? 20.0 : 14.0) * size_val.params_in_slice;
    }
};

/// Fills one row of kernels_list: one value count and store, every warp count.
///
///   row_list   kernels_list[values][store]
template <int VALUES, bool TMA_STORE, int... WARPS>
static void fill_warps(AdamwTmaKernel (&row_list)[sizeof...(WARPS)])
{
    int index_val = 0;
    ((row_list[index_val++] = adamw_tma_bf16_kernel<VALUES, WARPS, TMA_STORE>), ...);
}

/// Fills kernels_list, every kernel the sweep can launch, and reads each one's registers into infos_list.
static void fill_kernels(void)
{
    static_assert(COUNT(VALUES_LIST) == 4 && COUNT(TMA_STORE_LIST) == 2, "one line a value count and store below");
    fill_warps<4, true, WARPS_PACK>(kernels_list[0][0]);
    fill_warps<4, false, WARPS_PACK>(kernels_list[0][1]);
    fill_warps<8, true, WARPS_PACK>(kernels_list[1][0]);
    fill_warps<8, false, WARPS_PACK>(kernels_list[1][1]);
    fill_warps<16, true, WARPS_PACK>(kernels_list[2][0]);
    fill_warps<16, false, WARPS_PACK>(kernels_list[2][1]);
    fill_warps<32, true, WARPS_PACK>(kernels_list[3][0]);
    fill_warps<32, false, WARPS_PACK>(kernels_list[3][1]);
    for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        for (int store_index_val = 0; store_index_val < COUNT(TMA_STORE_LIST); store_index_val++)
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                // a kernel CUDA refuses is not the end of the tune: its settings are skipped and the error reported
                AdamwTmaKernel kernel_ptr = kernels_list[values_index_val][store_index_val][warps_index_val];
                KernelInfo &info_val = infos_list[values_index_val][store_index_val][warps_index_val];
                cudaFuncAttributes attributes_val = {};
                cudaError_t error_val = cudaFuncGetAttributes(&attributes_val, kernel_ptr);
                if (error_val == cudaSuccess)
                    error_val = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, ADAMW_TMA_BF16_MAX_SHARED_BYTES);
                info_val = {attributes_val.numRegs, attributes_val.localSizeBytes, error_val == cudaSuccess ? "" : cudaGetErrorString(error_val)};
                cudaGetLastError();   // cleared: the next call does not see it
            }
}

/// Launches one setting, and gives the launch's error (cudaSuccess if it started).
///
///   setting_val   the setting
///   data_val      the tensors
///   sms_val       this GPU's SMs
static cudaError_t launch_setting(const Setting &setting_val, const AdamwData &data_val, int sms_val)
{
    AdamwTmaKernel kernel_ptr = kernels_list[setting_val.values_index_val][setting_val.store_index_val][setting_val.warps_index_val];
    kernel_ptr<<<sms_val * setting_val.blocks_per_sm_val, THREADS_LIST[setting_val.warps_index_val],
                 setting_val.shared_bytes_val>>>(data_val.params_tensor.data_ptr<nv_bfloat16>(), data_val.grads_tensor.data_ptr<nv_bfloat16>(),
                                                 data_val.m_tensor.data_ptr<nv_bfloat16>(), data_val.v_tensor.data_ptr<nv_bfloat16>(),
                                                 data_val.master(), data_val.step_val, setting_val.tile_val);
    return cudaGetLastError();
}

/// Gives a setting as one line: its threads, values, who stores, parts and tile, registers, shared memory, blocks on an SM.
///
///   setting_val   the setting
static std::string describe(const Setting &setting_val)
{
    const KernelInfo &info_val = infos_list[setting_val.values_index_val][setting_val.store_index_val][setting_val.warps_index_val];
    char line_list[200];
    snprintf(line_list, sizeof(line_list),
             "%4d threads, %2d values, %-7s store, %d parts of %5d values  (%3d registers%s, %6zu B shared, %2d of %2d blocks on an SM)",
             THREADS_LIST[setting_val.warps_index_val], VALUES_LIST[setting_val.values_index_val],
             TMA_STORE_LIST[setting_val.store_index_val] ? "TMA" : "threads", setting_val.stages_val, setting_val.tile_values_val,
             info_val.registers, info_val.local_bytes > 0 ? " SPILLS" : "", setting_val.shared_bytes_val, setting_val.resident_val,
             setting_val.blocks_per_sm_val);
    return line_list;
}

/// Makes one point of the sweep a setting: its share, tile and shared memory, and why it is skipped if it is.
///
///   point_val             the point: a value of each list, read by SweepList (SWEEP_BLOCKS, ...)
///   has_master_val        a float master: 10 bytes a value, else 8
///   params_in_slice_val   values to update
static Setting plan_setting(const TunePoint &point_val, bool has_master_val, size_t params_in_slice_val)
{
    // Skipped (skipped_val says why): the kernel was refused or spills, the tile does not fit the block's share of
    // shared memory, or no block of it fits on an SM. Else skipped_val stays empty and the setting runs.
    // 1. the values, and the tile: the block's share cut into the parts
    Setting setting_val;
    setting_val.blocks_per_sm_val = point_val.value(SWEEP_BLOCKS);
    setting_val.warps_index_val = point_val.index(SWEEP_THREADS);   // the same place in WARPS_LIST: its kernel
    setting_val.values_index_val = point_val.index(SWEEP_VALUES);
    setting_val.store_index_val = point_val.index(SWEEP_STORE);
    setting_val.stages_val = point_val.value(SWEEP_PARTS);
    setting_val.share_bytes_val = adamw_tma_bf16_share_bytes(setting_val.blocks_per_sm_val);
    setting_val.tile_values_val = adamw_tma_bf16_tile_values(setting_val.share_bytes_val, setting_val.stages_val, has_master_val);
    AdamwTmaKernel kernel_ptr = kernels_list[setting_val.values_index_val][setting_val.store_index_val][setting_val.warps_index_val];
    const KernelInfo &info_val = infos_list[setting_val.values_index_val][setting_val.store_index_val][setting_val.warps_index_val];
    int threads_val = point_val.value(SWEEP_THREADS);
    char reason_list[160];
    // 2. the kernel: refused, or it spills
    if (!info_val.error.empty())
    {
        snprintf(reason_list, sizeof(reason_list), "kernel refused: %s", info_val.error.c_str());
        setting_val.skipped_val = reason_list;
        return setting_val;
    }
    if (info_val.local_bytes > 0)
    {
        snprintf(reason_list, sizeof(reason_list), "kernel spills %zu bytes a thread", info_val.local_bytes);
        setting_val.skipped_val = reason_list;
        return setting_val;
    }
    // 3. the tile: it fits the block's share of shared memory
    if (!adamw_tma_bf16_plan(&setting_val.tile_val, &setting_val.shared_bytes_val, has_master_val, setting_val.tile_values_val,
                             setting_val.stages_val, params_in_slice_val))
    {
        snprintf(reason_list, sizeof(reason_list), "does not fit shared memory: %d parts of a %d-byte share give a tile of %d values",
                 setting_val.stages_val, setting_val.share_bytes_val, setting_val.tile_values_val);
        setting_val.skipped_val = reason_list;
        return setting_val;
    }
    // 4. a block of it fits on an SM; how many at once
    int blocks_on_sm_val = 0;
    cudaError_t error_val = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_on_sm_val, kernel_ptr, threads_val, setting_val.shared_bytes_val);
    cudaGetLastError();   // cleared: the next call does not see it
    if (error_val != cudaSuccess)
    {
        snprintf(reason_list, sizeof(reason_list), "no room on an SM: %s", cudaGetErrorString(error_val));
        setting_val.skipped_val = reason_list;
    }
    else if (blocks_on_sm_val == 0)
    {
        snprintf(reason_list, sizeof(reason_list), "no room on an SM: %zu B shared, %d threads, %d registers a thread",
                 setting_val.shared_bytes_val, threads_val, info_val.registers);
        setting_val.skipped_val = reason_list;
    }
    setting_val.resident_val = std::min(blocks_on_sm_val, setting_val.blocks_per_sm_val);
    return setting_val;
}

// ══ the tune ══════════════════════════════════════════════════════════════════

/// Launches adamw_bf16_kernel, main's, at one launch.
///
///   data_val                 the tensors and this step's numbers
///   blocks_val, threads_val  the launch
static void launch_the_current_kernel(const AdamwData &data_val, int blocks_val, int threads_val)
{
    const AdamwTmaBf16Step &step_val = data_val.step_val;
    adamw_bf16_kernel<<<blocks_val, threads_val>>>(
        data_val.params_tensor.data_ptr<nv_bfloat16>(), data_val.grads_tensor.data_ptr<nv_bfloat16>(), data_val.m_tensor.data_ptr<nv_bfloat16>(),
        data_val.v_tensor.data_ptr<nv_bfloat16>(), data_val.size_val.params_in_slice, step_val.decayed_count, step_val.grad_norm_ptr,
        step_val.max_norm, step_val.weight_decay_factor, step_val.beta1, step_val.beta2, step_val.eps, step_val.step_size,
        step_val.inverse_sqrt_bias_correction2, data_val.master());
    cudaCheck(cudaGetLastError());
}

/// Times adamw_bf16_kernel at every launch of the baseline lists, and gives its fastest.
///
///   data_val                            the tensors
///   sms_val                             this GPU's SMs
///   best_blocks_ptr, best_threads_ptr   written: its fastest launch
static float time_the_current_kernel(const AdamwData &data_val, int sms_val, int *best_blocks_ptr, int *best_threads_ptr)
{
    // SWEEP_ROUNDS rounds, each timing every launch once. Blocks 0 in the list: one float4 a thread, as
    // adamw_update_bf16 launches.
    const int launches_val = COUNT(BASELINE_BLOCKS_PER_SM_LIST) * COUNT(BASELINE_THREADS_LIST);
    std::vector<float> total_ms_list(launches_val, 0.0f);
    auto blocks_of_val = [&](int launch_index_val) {
        int blocks_per_sm_val = BASELINE_BLOCKS_PER_SM_LIST[launch_index_val / COUNT(BASELINE_THREADS_LIST)];
        int threads_val = BASELINE_THREADS_LIST[launch_index_val % COUNT(BASELINE_THREADS_LIST)];
        return blocks_per_sm_val > 0 ? sms_val * blocks_per_sm_val
                                     : (int)CEIL_DIV(data_val.size_val.params_in_slice / VALUES_PER_FLOAT4, (size_t)threads_val);
    };
    for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
            total_ms_list[launch_index_val] += gpu_milliseconds([&] {
                launch_the_current_kernel(data_val, blocks_of_val(launch_index_val),
                                          BASELINE_THREADS_LIST[launch_index_val % COUNT(BASELINE_THREADS_LIST)]);
            });
    int best_index_val = (int)(std::min_element(total_ms_list.begin(), total_ms_list.end()) - total_ms_list.begin());
    *best_blocks_ptr = blocks_of_val(best_index_val);
    *best_threads_ptr = BASELINE_THREADS_LIST[best_index_val % COUNT(BASELINE_THREADS_LIST)];
    return total_ms_list[best_index_val] / SWEEP_ROUNDS;
}

/// Sweeps every setting, father to son, and prints the tree of the fastest, the finalists, and every setting in a .csv.
///
///   size_val         the counts
///   has_master_val   a float master: as training runs it
///   csv_path         where every setting's row goes
static void time_every_setting(const AdamwSize &size_val, bool has_master_val, const std::string &csv_path)
{
    // Every one with every other, father to son: blocks an SM B (the launch is B x this GPU's SMs, and each block
    // takes 1 / B of an SM's shared memory: its share), threads a block, values a thread, who stores (TMA or the
    // threads), then the parts the share is cut into, a tile each. 1 part: TMA fills the whole share and the block
    // waits for it before working on it. n parts: the block works on one while TMA fills the others, and each part
    // done takes the next tile.
    //
    // A setting that cannot run is skipped, never the end of the tune: its kernel spills registers (or CUDA would not
    // raise its shared memory), its tile does not fit shared memory, no block of it fits on an SM, or its launch is
    // refused. Each is reported with its reason, in the log and in the .csv.
    //
    // First pass, in that order: each setting run once from the start and its master, params, m and v checked against
    // adamw_update_bf16's, to the last bit, then timed FIRST_PASS_RUNS times (once if slower than SLOW_CUT x the current
    // kernel's best). The timed runs keep stepping the same tensors: the values move, the time does not. Second pass:
    // the FINALISTS fastest and the current kernel's best launch, timed over SWEEP_ROUNDS rounds taking turns.
    //
    // Expect: every setting gives the same bits.

    // 1. the tensors, the kernels, the current kernel's fastest launch
    AdamwData data_val(size_val, has_master_val);
    fill_kernels();
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    warm_up_gpu([&] { launch_the_current_kernel(data_val, (int)CEIL_DIV(size_val.params_in_slice / VALUES_PER_FLOAT4, (size_t)256), 256); });
    int baseline_blocks_val = 0, baseline_threads_val = 0;
    float baseline_ms_val = time_the_current_kernel(data_val, sms_val, &baseline_blocks_val, &baseline_threads_val);

    // 2. every setting, father to son (for_every_setting); the ones that cannot run skipped, with why
    const std::vector<TuneList> tunes_list = {
        {"blocks_per_sm", BLOCKS_PER_SM_LIST},   // SWEEP_BLOCKS: blocks an SM, which sets each block's share of shared memory
        {"threads", THREADS_LIST},               // SWEEP_THREADS: threads a block
        {"values", VALUES_LIST},                 // SWEEP_VALUES: values a thread a round
        {"tma_store", TMA_STORE_LIST},           // SWEEP_STORE: 1 TMA stores, 0 the threads
        {"parts", PARTS_LIST},                   // SWEEP_PARTS: the share cut into this many tiles
    };
    std::vector<Setting> settings_list;
    std::vector<Setting> skipped_list;
    printf("    %zu settings in the sweep: %s", count_settings(tunes_list), tunes_list[0].name_ptr);
    for (size_t tune_index_val = 1; tune_index_val < tunes_list.size(); tune_index_val++)
        printf(" x %s", tunes_list[tune_index_val].name_ptr);
    printf("\n");
    for_every_setting(tunes_list, [&](const TunePoint &point_val) {
        Setting setting_val = plan_setting(point_val, has_master_val, size_val.params_in_slice);
        (setting_val.skipped_val.empty() ? settings_list : skipped_list).push_back(setting_val);
    });
    long fewer_resident_val = (long)std::count_if(settings_list.begin(), settings_list.end(),
                                                  [](const Setting &s) { return s.resident_val < s.blocks_per_sm_val; });
    printf("    %zu settings to run (%ld with fewer than B blocks on an SM at once: they run in waves), %zu skipped (listed below)\n",
           settings_list.size(), fewer_resident_val, skipped_list.size());

    // 3. first pass: check, then time; a launch CUDA refuses (too many resources, a bad launch) skipped, with its error
    auto start_val = std::chrono::steady_clock::now();
    std::vector<Setting> ran_list;
    for (size_t setting_index_val = 0; setting_index_val < settings_list.size(); setting_index_val++)
    {
        Setting &setting_val = settings_list[setting_index_val];
        data_val.reset();
        cudaError_t error_val = launch_setting(setting_val, data_val, sms_val);
        if (error_val != cudaSuccess)
        {
            setting_val.skipped_val = std::string("launch refused: ") + cudaGetErrorString(error_val);
            skipped_list.push_back(setting_val);
            continue;
        }
        // an error while it runs (a bad address) breaks the GPU for every later call: nothing to skip to, so stop here
        error_val = cudaDeviceSynchronize();
        if (error_val != cudaSuccess)
        {
            printf("    stopped: %s failed while running: %s; the GPU takes no more work after this\n", describe(setting_val).c_str(),
                   cudaGetErrorString(error_val));
            exit(EXIT_FAILURE);
        }
        setting_val.same_val = data_val.same();
        float ms_val = gpu_milliseconds([&] { launch_setting(setting_val, data_val, sms_val); });
        setting_val.first_runs_val = 1;
        if (ms_val < SLOW_CUT * baseline_ms_val)
            for (; setting_val.first_runs_val < FIRST_PASS_RUNS; setting_val.first_runs_val++)
                ms_val += gpu_milliseconds([&] { launch_setting(setting_val, data_val, sms_val); });
        setting_val.first_ms_val = ms_val / setting_val.first_runs_val;
        ran_list.push_back(setting_val);
        if ((setting_index_val + 1) % PROGRESS_EVERY == 0)
            printf("    ... %zu of %zu settings, %.0f s\n", setting_index_val + 1, settings_list.size(),
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - start_val).count());
    }
    settings_list = ran_list;   // from here on, the settings that ran

    // 4. second pass: the finalists and the current kernel's best, taking turns
    std::vector<int> order_list(settings_list.size());
    for (size_t index_val = 0; index_val < order_list.size(); index_val++)
        order_list[index_val] = (int)index_val;
    std::sort(order_list.begin(), order_list.end(), [&](int first_val, int second_val) {
        return settings_list[first_val].first_ms_val < settings_list[second_val].first_ms_val;
    });
    int finalists_val = std::min((int)order_list.size(), FINALISTS);
    std::vector<float> total_ms_list(finalists_val, 0.0f);
    float baseline_total_ms_val = 0.0f;
    for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
    {
        baseline_total_ms_val += gpu_milliseconds([&] { launch_the_current_kernel(data_val, baseline_blocks_val, baseline_threads_val); });
        for (int finalist_index_val = 0; finalist_index_val < finalists_val; finalist_index_val++)
            total_ms_list[finalist_index_val] +=
                gpu_milliseconds([&] { launch_setting(settings_list[order_list[finalist_index_val]], data_val, sms_val); });
    }
    for (int finalist_index_val = 0; finalist_index_val < finalists_val; finalist_index_val++)
        settings_list[order_list[finalist_index_val]].final_ms_val = total_ms_list[finalist_index_val] / SWEEP_ROUNDS;
    float baseline_final_ms_val = baseline_total_ms_val / SWEEP_ROUNDS;

    // 5. the registers of each kernel
    double bytes_val = data_val.bytes();
    double peak_val = peak_gigabytes_per_second();
    print_test_name(__func__);
    printf("    %s: %zu values (%zu decayed), %s: %.1f MB a step; this GPU's peak %.0f GB/s\n", size_val.name_ptr,
           size_val.params_in_slice, size_val.decayed_count, has_master_val ? "a float master" : "no master", bytes_val / MEGA, peak_val);
    printf("    registers a thread (S: spills to local memory, its settings skipped; X: CUDA refused it, its settings skipped)\n");
    printf("    threads  store   ");
    for (int values_val : VALUES_LIST)
        printf("  %3d values", values_val);
    printf("\n");
    for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        for (int store_index_val = 0; store_index_val < COUNT(TMA_STORE_LIST); store_index_val++)
        {
            printf("    %7d  %-7s ", THREADS_LIST[warps_index_val], TMA_STORE_LIST[store_index_val] ? "TMA" : "threads");
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                const KernelInfo &info_val = infos_list[values_index_val][store_index_val][warps_index_val];
                printf("  %8d %c", info_val.registers, !info_val.error.empty() ? 'X' : info_val.local_bytes > 0 ? 'S' : ' ');
            }
            printf("\n");
        }

    // 5b. the settings skipped: how many for each kind of reason, then each one with its own
    printf("    %zu settings ran, %zu skipped\n", settings_list.size(), skipped_list.size());
    if (!skipped_list.empty())
    {
        const char *kinds_list[] = {"kernel spills", "kernel refused", "does not fit shared memory", "no room on an SM", "launch refused"};
        for (const char *kind_ptr : kinds_list)
        {
            long count_val = (long)std::count_if(skipped_list.begin(), skipped_list.end(),
                                                 [&](const Setting &s) { return s.skipped_val.rfind(kind_ptr, 0) == 0; });
            if (count_val > 0)
                printf("      %6ld  %s\n", count_val, kind_ptr);
        }
        printf("    skipped, each one (blocks an SM, then the setting: why)\n");
        for (const Setting &setting_val : skipped_list)
            printf("      %2d/SM  %s: %s\n", setting_val.blocks_per_sm_val, describe(setting_val).c_str(), setting_val.skipped_val.c_str());
    }

    // 6. the tree: the fastest setting under each blocks, each blocks x threads, each blocks x threads x values (first pass)
    printf("    the current kernel: fastest %d blocks x %d threads, %.4f ms, %.0f GB/s\n", baseline_blocks_val, baseline_threads_val,
           baseline_ms_val, gigabytes_per_second(bytes_val, baseline_ms_val));
    printf("    the tree, first pass: the fastest under each father (ms, GB/s, x the current kernel's best)\n");
    auto fastest_val = [&](auto matches_val) {
        int best_val = -1;
        for (size_t index_val = 0; index_val < settings_list.size(); index_val++)
            if (matches_val(settings_list[index_val]) && (best_val < 0 || settings_list[index_val].first_ms_val < settings_list[best_val].first_ms_val))
                best_val = (int)index_val;
        return best_val;
    };
    auto print_node_val = [&](int depth_val, const char *label_ptr, int best_val) {
        if (best_val < 0)
        {
            printf("    %*s%-18s nothing runs\n", 4 * depth_val, "", label_ptr);
            return;
        }
        const Setting &setting_val = settings_list[best_val];
        printf("    %*s%-18s %8.4f  %5.0f  %5.2fx   %s\n", 4 * depth_val, "", label_ptr, setting_val.first_ms_val,
               gigabytes_per_second(bytes_val, setting_val.first_ms_val), baseline_ms_val / setting_val.first_ms_val,
               describe(setting_val).c_str());
    };
    char label_list[32];
    for (int blocks_per_sm_val : BLOCKS_PER_SM_LIST)
    {
        snprintf(label_list, sizeof(label_list), "blocks %d (%dx SMs)", sms_val * blocks_per_sm_val, blocks_per_sm_val);
        print_node_val(0, label_list, fastest_val([&](const Setting &s) { return s.blocks_per_sm_val == blocks_per_sm_val; }));
        for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        {
            snprintf(label_list, sizeof(label_list), "threads %d", THREADS_LIST[warps_index_val]);
            print_node_val(1, label_list, fastest_val([&](const Setting &s) {
                               return s.blocks_per_sm_val == blocks_per_sm_val && s.warps_index_val == warps_index_val;
                           }));
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                snprintf(label_list, sizeof(label_list), "values %d", VALUES_LIST[values_index_val]);
                print_node_val(2, label_list, fastest_val([&](const Setting &s) {
                                   return s.blocks_per_sm_val == blocks_per_sm_val && s.warps_index_val == warps_index_val &&
                                          s.values_index_val == values_index_val;
                               }));
            }
        }
    }

    // 7. blocks an SM against parts, once for each store: the fastest setting in each cell, as x the current kernel's
    //    best (above 1: faster), so 1 part (wait for the whole share) and n parts (work while TMA fills) side by side
    for (int store_index_val = 0; store_index_val < COUNT(TMA_STORE_LIST); store_index_val++)
    {
        printf("    %s store, the fastest at each blocks an SM and parts (first pass, x the current kernel's best)\n",
               TMA_STORE_LIST[store_index_val] ? "TMA" : "threads");
        printf("    blocks/SM   share KB");
        for (int parts_val : PARTS_LIST)
            printf("  %d parts", parts_val);
        printf("\n");
        for (int blocks_per_sm_val : BLOCKS_PER_SM_LIST)
        {
            printf("    %9d   %8.1f", blocks_per_sm_val, adamw_tma_bf16_share_bytes(blocks_per_sm_val) / 1024.0);
            for (int parts_val : PARTS_LIST)
            {
                int best_val = fastest_val([&](const Setting &s) {
                    return s.blocks_per_sm_val == blocks_per_sm_val && s.store_index_val == store_index_val && s.stages_val == parts_val;
                });
                if (best_val < 0)
                    printf("  %7s", "-");
                else
                    printf("  %6.2fx", baseline_ms_val / settings_list[best_val].first_ms_val);
            }
            printf("\n");
        }
    }

    // 8. the finalists, timed again taking turns with the current kernel's best
    printf("    the %d fastest, timed again over %d rounds with the current kernel (%d x %d: %.4f ms, %.0f GB/s, MBU %.0f%%)\n",
           TOP_PRINTED, SWEEP_ROUNDS, baseline_blocks_val, baseline_threads_val, baseline_final_ms_val,
           gigabytes_per_second(bytes_val, baseline_final_ms_val), 100.0 * gigabytes_per_second(bytes_val, baseline_final_ms_val) / peak_val);
    std::vector<int> final_order_list(order_list.begin(), order_list.begin() + finalists_val);
    std::sort(final_order_list.begin(), final_order_list.end(),
              [&](int first_val, int second_val) { return settings_list[first_val].final_ms_val < settings_list[second_val].final_ms_val; });
    printf("         ms    GB/s   MBU   x current   blocks   setting\n");
    for (int rank_val = 0; rank_val < std::min(TOP_PRINTED, finalists_val); rank_val++)
    {
        const Setting &setting_val = settings_list[final_order_list[rank_val]];
        double speed_val = gigabytes_per_second(bytes_val, setting_val.final_ms_val);
        printf("    %7.4f  %6.0f  %3.0f%%     %5.2fx   %6d   %s\n", setting_val.final_ms_val, speed_val, 100.0 * speed_val / peak_val,
               baseline_final_ms_val / setting_val.final_ms_val, sms_val * setting_val.blocks_per_sm_val, describe(setting_val).c_str());
    }

    // 9. every setting into the .csv
    FILE *csv_ptr = fopen(csv_path.c_str(), "w");
    if (csv_ptr != nullptr)
    {
        // the settings that ran, then the skipped ones (no times; status: why, in quotes)
        fprintf(csv_ptr, "blocks,blocks_per_sm,resident_per_sm,threads,values,store,parts,share_bytes,tile_values,registers,spill_bytes,"
                         "shared_bytes,first_ms,first_runs,final_ms,gb_per_s,same,status\n");
        for (const std::vector<Setting> *list_ptr : {&settings_list, &skipped_list})
            for (const Setting &setting_val : *list_ptr)
            {
                const KernelInfo &info_val = infos_list[setting_val.values_index_val][setting_val.store_index_val][setting_val.warps_index_val];
                bool ran_val = setting_val.skipped_val.empty();
                fprintf(csv_ptr, "%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%zu,%zu,%.5f,%d,%.5f,%.0f,%d,\"%s\"\n", sms_val * setting_val.blocks_per_sm_val,
                        setting_val.blocks_per_sm_val, setting_val.resident_val, THREADS_LIST[setting_val.warps_index_val],
                        VALUES_LIST[setting_val.values_index_val], TMA_STORE_LIST[setting_val.store_index_val] ? "tma" : "threads",
                        setting_val.stages_val, setting_val.share_bytes_val, setting_val.tile_values_val, info_val.registers,
                        info_val.local_bytes, setting_val.shared_bytes_val, setting_val.first_ms_val, setting_val.first_runs_val,
                        setting_val.final_ms_val, ran_val ? gigabytes_per_second(bytes_val, setting_val.first_ms_val) : 0.0,
                        setting_val.same_val ? 1 : 0, ran_val ? "ran" : setting_val.skipped_val.c_str());
            }
        fclose(csv_ptr);
        printf("    every setting: %s\n", csv_path.c_str());
    }
    else
    {
        printf("    could not write %s\n", csv_path.c_str());
    }

    // 10. every setting that ran gave adamw_update_bf16's bits (the skipped ones are in the list above, not here);
    //     any that did not printed first, so the check's FAIL names them
    for (const Setting &setting_val : settings_list)
        if (!setting_val.same_val)
            printf("    NOT the same bits: %2d/SM  %s\n", setting_val.blocks_per_sm_val, describe(setting_val).c_str());
    long same_count_val = (long)std::count_if(settings_list.begin(), settings_list.end(), [](const Setting &s) { return s.same_val; });
    ASSERT_EQ(same_count_val, (long)settings_list.size());
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    // ./tune_adamw_tma_bf16: GPT-2 small's count with a float master (as training runs it); d24: nanochat d24's
    // count; nomaster: no master, the bfloat16 params read. Any order: ./tune_adamw_tma_bf16 d24 nomaster.
    AdamwSize size_val = GPT2_SIZE;
    bool has_master_val = true;
    for (int arg_index_val = 1; arg_index_val < argc; arg_index_val++)
    {
        std::string arg_val = argv[arg_index_val];
        if (arg_val == "d24")
            size_val = D24_SIZE;
        else if (arg_val == "nomaster")
            has_master_val = false;
    }
    std::string folder_val = "cuda/tests/results/" + gpu_folder_name();
    std::string log_path_val = start_run_log(folder_val.c_str(), __FILE__, "tune_", "txt");
    print_gpu();
    std::string csv_path_val = log_path_val.substr(0, log_path_val.size() - 3) + "csv";
    time_every_setting(size_val, has_master_val, csv_path_val);

    printf("\nall tunes done\n");
    return 0;
}
