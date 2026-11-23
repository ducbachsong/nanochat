// tune_encoder_tma_bf16.cu — the timings of cuda/kernels/bf16/encoder_tma_bf16.cuh: every setting, father to son, against encoder_bf16_forward_kernel.
#include <torch/types.h>              // torch::randn, torch::equal; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <random>                     // std::mt19937: random token ids
#include <algorithm>                  // std::sort
#include <chrono>                     // the sweep's progress
#include <map>                        // the skipped settings, counted by reason
#include <string>
#include <vector>
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gpu_folder_name, for_every_setting
#include "../../kernels/bf16/encoder_bf16.cuh"
#include "../../kernels/bf16/encoder_tma_bf16.cuh"

/// An encoder's sizes.
struct EncoderSize
{
    const char *name_ptr;
    size_t batch_size, seq_len, embed_dim;
    int vocab_size;
};
static const EncoderSize D24_SIZE = {"nanochat d24", 16, 2048, 1536, 32768};   // the trainer's: 16 rows of 2,048 tokens
static const EncoderSize GPT2_SIZE = {"GPT-2 small", 4, 1024, 768, 50304};     // tune_encoder_bf16's

// The sweep, every one with every other, father to son: blocks an SM, threads a block, values a thread, the tile (its
// rows, its columns, the swizzle TMA lays it in), the sets of tiles in shared memory
static constexpr int BLOCKS_PER_SM_LIST[] = {1, 2, 3, 4, 6, 8, 16, 32};
static constexpr int THREADS_LIST[] = {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024};   // threads a block: whole warps
#define WARPS_PACK 1, 2, 3, 4, 6, 8, 12, 16, 24, 32           // the kernels compiled, one a THREADS_LIST entry: its warps
static constexpr int WARPS_LIST[] = {WARPS_PACK};
static constexpr int VALUES_LIST[] = {8, 16, 32, 64, 128};
static constexpr int TILE_ROWS_LIST[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
static constexpr int TILE_COLUMNS_LIST[] = {64, 128, 256, 384, 768, 1536, 3072, 5120, 6144, 7680, 15360};
static constexpr int SWIZZLE_LIST[] = {128, 0};
static constexpr int SETS_LIST[] = {1, 2, 3, 4, 5, 6, 7, 8};
static_assert(sizeof(SETS_LIST) / sizeof(SETS_LIST[0]) == ENCODER_TMA_BF16_MAX_STAGES, "every set count the kernel takes");
enum SweepList { SWEEP_BLOCKS, SWEEP_THREADS, SWEEP_VALUES, SWEEP_ROWS, SWEEP_COLUMNS, SWEEP_SWIZZLE, SWEEP_SETS };   // each list's place in time_every_setting's tunes_list, father to son: the same order there
static constexpr int BASELINE_BLOCKS_PER_SM_LIST[] = {1, 2, 4, 8, 16, 32};   // encoder_bf16_forward_kernel's launches
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
#define PROGRESS_EVERY 5000           // settings between progress lines
#define MEGA 1e6

using EncoderTmaKernel = void (*)(const CUtensorMap, const CUtensorMap, const CUtensorMap, const int *, EncoderTmaBf16Tile);
static EncoderTmaKernel kernels_list[COUNT(VALUES_LIST)][COUNT(SWIZZLE_LIST)][COUNT(WARPS_LIST)];

/// One kernel's registers, and why its settings are skipped if they are.
struct KernelInfo
{
    int registers;
    size_t local_bytes;               // > 0: registers spilled to local memory, so its settings are skipped
    std::string error;                // not empty: CUDA would not raise its shared memory, so its settings are skipped
};
static KernelInfo infos_list[COUNT(VALUES_LIST)][COUNT(SWIZZLE_LIST)][COUNT(WARPS_LIST)];

/// One setting of the sweep, its tile and maps, and its times.
struct Setting
{
    alignas(64) CUtensorMap maps_list[3];   // wte, wpe, encoded
    int blocks_per_sm_val, warps_index_val, values_index_val, tile_rows_val, tile_columns_val, swizzle_index_val, sets_val;
    int resident_val = 0;                   // blocks of it on an SM at once: blocks_per_sm_val, or fewer if threads, registers or shared memory run out
    EncoderTmaBf16Tile tile_val = {};
    size_t shared_bytes_val = 0;
    std::string skipped_val;                // empty: it ran; else why it was skipped
    float first_ms_val = 0.0f;
    int first_runs_val = 0;
    bool same_val = false;
    float final_ms_val = 0.0f;
};

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

/// Makes an int tensor {batch_size_val, seq_len_val} of random token ids below vocab_size_val on the GPU.
///
///   batch_size_val, seq_len_val   its shape
///   vocab_size_val                the ids' bound
static Tensor random_token_ids(size_t batch_size_val, size_t seq_len_val, int vocab_size_val)
{
    std::mt19937 generator_val(0);
    std::uniform_int_distribution<int> pick_val(0, vocab_size_val - 1);
    std::vector<int> ids_list(batch_size_val * seq_len_val);
    for (int &id_val : ids_list)
        id_val = pick_val(generator_val);
    Tensor tensor = Tensor::empty({batch_size_val, seq_len_val}, DType::Int32);
    tensor.copy_from_cpu(ids_list.data());
    return tensor;
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

/// The encoder's tensors at one size, and encoder_bf16_forward's result on them.
struct EncoderData
{
    EncoderSize size_val;
    Tensor wte_tensor, wpe_tensor, token_ids_tensor, expected_tensor, encoded_tensor;

    /// Makes the tensors and the expected result.
    ///
    ///   size_val   the sizes
    explicit EncoderData(const EncoderSize &size_val) : size_val(size_val)
    {
        torch::manual_seed(0);
        wte_tensor = random_bf16_tensor({(size_t)size_val.vocab_size, size_val.embed_dim});
        wpe_tensor = random_bf16_tensor({size_val.seq_len, size_val.embed_dim});
        token_ids_tensor = random_token_ids(size_val.batch_size, size_val.seq_len, size_val.vocab_size);
        expected_tensor = Tensor::zeros({size_val.batch_size, size_val.seq_len, size_val.embed_dim}, DType::BFloat16);
        encoded_tensor = Tensor::zeros({size_val.batch_size, size_val.seq_len, size_val.embed_dim}, DType::BFloat16);
        encoder_bf16_forward(expected_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    }

    /// Gives the bytes the forward moves: 3 bytes a value read or written (wte's and wpe's rows, encoded), and the ids.
    double bytes(void) const
    {
        double tokens_val = (double)size_val.batch_size * size_val.seq_len;
        return 3.0 * tokens_val * size_val.embed_dim * sizeof(nv_bfloat16) + tokens_val * sizeof(int);
    }
};

/// Fills one row of kernels_list: one value count and swizzle, every warp count.
///
///   row_list   kernels_list[values][swizzle]
template <int VALUES, int SWIZZLE, int... WARPS>
static void fill_warps(EncoderTmaKernel (&row_list)[sizeof...(WARPS)])
{
    int index_val = 0;
    ((row_list[index_val++] = encoder_tma_bf16_kernel<VALUES, SWIZZLE, WARPS>), ...);
}

/// Fills kernels_list, every kernel the sweep can launch, and reads each one's registers into infos_list.
static void fill_kernels(void)
{
    static_assert(COUNT(VALUES_LIST) == 5 && COUNT(SWIZZLE_LIST) == 2, "one line a value count and swizzle below");
    fill_warps<8, 128, WARPS_PACK>(kernels_list[0][0]);
    fill_warps<8, 0, WARPS_PACK>(kernels_list[0][1]);
    fill_warps<16, 128, WARPS_PACK>(kernels_list[1][0]);
    fill_warps<16, 0, WARPS_PACK>(kernels_list[1][1]);
    fill_warps<32, 128, WARPS_PACK>(kernels_list[2][0]);
    fill_warps<32, 0, WARPS_PACK>(kernels_list[2][1]);
    fill_warps<64, 128, WARPS_PACK>(kernels_list[3][0]);
    fill_warps<64, 0, WARPS_PACK>(kernels_list[3][1]);
    fill_warps<128, 128, WARPS_PACK>(kernels_list[4][0]);
    fill_warps<128, 0, WARPS_PACK>(kernels_list[4][1]);
    for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                // a kernel CUDA refuses is not the end of the tune: its settings are skipped and the error reported
                EncoderTmaKernel kernel_ptr = kernels_list[values_index_val][swizzle_index_val][warps_index_val];
                KernelInfo &info_val = infos_list[values_index_val][swizzle_index_val][warps_index_val];
                cudaFuncAttributes attributes_val = {};
                cudaError_t error_val = cudaFuncGetAttributes(&attributes_val, kernel_ptr);
                if (error_val == cudaSuccess)
                    error_val = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, ENCODER_TMA_BF16_MAX_SHARED_BYTES);
                info_val = {attributes_val.numRegs, attributes_val.localSizeBytes, error_val == cudaSuccess ? "" : cudaGetErrorString(error_val)};
                cudaGetLastError();   // cleared: the next call does not see it
            }
}

/// Launches one setting, and gives the launch's error (cudaSuccess if it started).
///
///   setting_val   the setting
///   data_val      the tensors
///   sms_val       this GPU's SMs
static cudaError_t launch_setting(const Setting &setting_val, const EncoderData &data_val, int sms_val)
{
    EncoderTmaKernel kernel_ptr = kernels_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val];
    kernel_ptr<<<sms_val * setting_val.blocks_per_sm_val, THREADS_LIST[setting_val.warps_index_val], setting_val.shared_bytes_val>>>(
        setting_val.maps_list[0], setting_val.maps_list[1], setting_val.maps_list[2], data_val.token_ids_tensor.data_ptr<int>(),
        setting_val.tile_val);
    return cudaGetLastError();
}

/// Gives a setting as one line: its threads, values, tile, swizzle, sets, registers, shared memory, blocks on an SM.
///
///   setting_val   the setting
static std::string describe(const Setting &setting_val)
{
    const KernelInfo &info_val = infos_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val];
    char line_list[200];
    snprintf(line_list, sizeof(line_list),
             "%4d threads, %3d values, tile %3d x %5d, swizzle %3d, %d sets  (%3d registers%s, %6zu B shared, %2d of %2d blocks on an SM)",
             THREADS_LIST[setting_val.warps_index_val], VALUES_LIST[setting_val.values_index_val], setting_val.tile_rows_val,
             setting_val.tile_columns_val, SWIZZLE_LIST[setting_val.swizzle_index_val], setting_val.sets_val, info_val.registers,
             info_val.local_bytes > 0 ? " SPILLS" : "", setting_val.shared_bytes_val, setting_val.resident_val, setting_val.blocks_per_sm_val);
    return line_list;
}

/// Makes one point of the sweep a setting: its tile, shared memory and tensor maps, and why it is skipped if it is.
///
///   point_val   the point: a value of each list, read by SweepList (SWEEP_BLOCKS, ...)
///   data_val    the tensors: their sizes, and the addresses the maps describe
static Setting plan_setting(const TunePoint &point_val, const EncoderData &data_val)
{
    // Skipped (skipped_val says why): the kernel was refused or spills, the tile does not fit the sizes or shared
    // memory, no block of it fits on an SM, or the driver refuses a tensor map. Each reason names only what caused it
    // (the kernel, or the tile), so the log can count the settings under each. Else skipped_val stays empty.
    // 1. the values
    const EncoderSize &size_val = data_val.size_val;
    Setting setting_val;
    setting_val.blocks_per_sm_val = point_val.value(SWEEP_BLOCKS);
    setting_val.warps_index_val = point_val.index(SWEEP_THREADS);   // the same place in WARPS_LIST: its kernel
    setting_val.values_index_val = point_val.index(SWEEP_VALUES);
    setting_val.tile_rows_val = point_val.value(SWEEP_ROWS);
    setting_val.tile_columns_val = point_val.value(SWEEP_COLUMNS);
    setting_val.swizzle_index_val = point_val.index(SWEEP_SWIZZLE);
    setting_val.sets_val = point_val.value(SWEEP_SETS);
    int swizzle_val = point_val.value(SWEEP_SWIZZLE);
    int threads_val = point_val.value(SWEEP_THREADS);
    EncoderTmaKernel kernel_ptr = kernels_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val];
    const KernelInfo &info_val = infos_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val];
    char reason_list[200];
    // 2. the kernel: refused, or it spills
    if (!info_val.error.empty() || info_val.local_bytes > 0)
    {
        if (!info_val.error.empty())
            snprintf(reason_list, sizeof(reason_list), "kernel refused: %d threads, %d values, swizzle %d: %s", threads_val,
                     point_val.value(SWEEP_VALUES), swizzle_val, info_val.error.c_str());
        else
            snprintf(reason_list, sizeof(reason_list), "kernel spills: %d threads, %d values, swizzle %d: %zu bytes a thread", threads_val,
                     point_val.value(SWEEP_VALUES), swizzle_val, info_val.local_bytes);
        setting_val.skipped_val = reason_list;
        return setting_val;
    }
    // 3. the tile: rows divide seq_len (at most a box's 256), columns divide embed_dim in whole boxes, the sets fit a
    //    block's shared memory
    if (!encoder_tma_bf16_plan(&setting_val.tile_val, &setting_val.shared_bytes_val, swizzle_val, setting_val.tile_rows_val,
                               setting_val.tile_columns_val, setting_val.sets_val, (int)size_val.batch_size, (int)size_val.seq_len,
                               (int)size_val.embed_dim))
    {
        snprintf(reason_list, sizeof(reason_list), "does not fit: tile %d x %d, swizzle %d, %d sets (seq_len %zu, embed_dim %zu, %d B shared)",
                 setting_val.tile_rows_val, setting_val.tile_columns_val, swizzle_val, setting_val.sets_val, size_val.seq_len,
                 size_val.embed_dim, ENCODER_TMA_BF16_MAX_SHARED_BYTES);
        setting_val.skipped_val = reason_list;
        return setting_val;
    }
    // 4. a block of it fits on an SM; how many at once
    int blocks_on_sm_val = 0;
    cudaError_t error_val = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_on_sm_val, kernel_ptr, threads_val, setting_val.shared_bytes_val);
    cudaGetLastError();
    if (error_val != cudaSuccess || blocks_on_sm_val == 0)
    {
        snprintf(reason_list, sizeof(reason_list), "no room on an SM: %d threads, %d registers a thread, %zu B shared%s%s", threads_val,
                 info_val.registers, setting_val.shared_bytes_val, error_val != cudaSuccess ? ": " : "",
                 error_val != cudaSuccess ? cudaGetErrorString(error_val) : "");
        setting_val.skipped_val = reason_list;
        return setting_val;
    }
    setting_val.resident_val = std::min(blocks_on_sm_val, setting_val.blocks_per_sm_val);
    // 5. the maps: wte one row a box, wpe and encoded tile_rows rows a box
    int box_columns_val = setting_val.tile_val.box_columns;
    CUresult results_list[3] = {
        encoder_tma_bf16_try_tensor_map(&setting_val.maps_list[0], data_val.wte_tensor.data_ptr<nv_bfloat16>(), size_val.vocab_size,
                                        size_val.embed_dim, box_columns_val, 1, swizzle_val),
        encoder_tma_bf16_try_tensor_map(&setting_val.maps_list[1], data_val.wpe_tensor.data_ptr<nv_bfloat16>(), size_val.seq_len,
                                        size_val.embed_dim, box_columns_val, setting_val.tile_rows_val, swizzle_val),
        encoder_tma_bf16_try_tensor_map(&setting_val.maps_list[2], data_val.encoded_tensor.data_ptr<nv_bfloat16>(), setting_val.tile_val.tokens,
                                        size_val.embed_dim, box_columns_val, setting_val.tile_rows_val, swizzle_val)};
    for (int map_index_val = 0; map_index_val < 3; map_index_val++)
        if (results_list[map_index_val] != CUDA_SUCCESS)
        {
            const char *names_list[3] = {"wte", "wpe", "encoded"};
            snprintf(reason_list, sizeof(reason_list), "tensor map refused: %s, box %d x %d, swizzle %d: CUresult %d", names_list[map_index_val],
                     map_index_val == 0 ? 1 : setting_val.tile_rows_val, box_columns_val, swizzle_val, (int)results_list[map_index_val]);
            setting_val.skipped_val = reason_list;
            return setting_val;
        }
    return setting_val;
}

// ══ the tune ══════════════════════════════════════════════════════════════════

/// Launches encoder_bf16_forward_kernel, the current kernel, at one launch.
///
///   data_val                 the tensors
///   blocks_val, threads_val  the launch
static void launch_the_current_kernel(const EncoderData &data_val, int blocks_val, int threads_val)
{
    encoder_bf16_forward_kernel<<<blocks_val, threads_val>>>(
        data_val.encoded_tensor.data_ptr<nv_bfloat16>(), data_val.token_ids_tensor.data_ptr<int>(), data_val.wte_tensor.data_ptr<nv_bfloat16>(),
        data_val.wpe_tensor.data_ptr<nv_bfloat16>(), data_val.size_val.batch_size, data_val.size_val.seq_len, data_val.size_val.embed_dim);
    cudaCheck(cudaGetLastError());
}

/// Times encoder_bf16_forward_kernel at every launch of the baseline lists, and gives its fastest.
///
///   data_val                            the tensors
///   sms_val                             this GPU's SMs
///   best_blocks_ptr, best_threads_ptr   written: its fastest launch
static float time_the_current_kernel(const EncoderData &data_val, int sms_val, int *best_blocks_ptr, int *best_threads_ptr)
{
    // SWEEP_ROUNDS rounds, each timing every launch once.
    const int launches_val = COUNT(BASELINE_BLOCKS_PER_SM_LIST) * COUNT(BASELINE_THREADS_LIST);
    std::vector<float> total_ms_list(launches_val, 0.0f);
    auto blocks_of_val = [&](int launch_index_val) {
        return sms_val * BASELINE_BLOCKS_PER_SM_LIST[launch_index_val / COUNT(BASELINE_THREADS_LIST)];
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

/// Sweeps every setting, father to son, and prints the tree of the fastest, the tables, the finalists, and every setting in a .csv.
///
///   size_val   the encoder's sizes
///   csv_path   where every setting's row goes
static void time_every_setting(const EncoderSize &size_val, const std::string &csv_path)
{
    // Every one with every other, father to son: blocks an SM B (the launch is B x this GPU's SMs; each block walks
    // its tiles), threads a block, values a thread, then the tile: its rows (tokens), its columns (values of a row),
    // the swizzle TMA lays it in, and the sets of tiles in shared memory. 1 set: the block waits for TMA to bring a
    // tile, adds it, stores it, then asks for the next. n sets: the block adds one while TMA brings the others.
    //
    // A setting that cannot run is skipped, never the end of the tune: its kernel spills registers (or CUDA would not
    // raise its shared memory), its tile does not fit the sizes or shared memory, no block of it fits on an SM, the
    // driver refuses a tensor map, or its launch is refused. The log counts the settings under each reason; the .csv
    // has every setting with its own.
    //
    // First pass, in that order: each setting's result checked against encoder_bf16_forward's, to the last bit, then
    // timed FIRST_PASS_RUNS times (once if slower than SLOW_CUT x the current kernel's best). Second pass: the FINALISTS
    // fastest and the current kernel's best launch, timed over SWEEP_ROUNDS rounds taking turns.
    //
    // Expect: every setting gives the same bits.

    // 1. the tensors, the kernels, the current kernel's fastest launch
    EncoderData data_val(size_val);
    fill_kernels();
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    warm_up_gpu([&] { encoder_bf16_forward(data_val.encoded_tensor, data_val.token_ids_tensor, data_val.wte_tensor, data_val.wpe_tensor); });
    int baseline_blocks_val = 0, baseline_threads_val = 0;
    float baseline_ms_val = time_the_current_kernel(data_val, sms_val, &baseline_blocks_val, &baseline_threads_val);

    // 2. every setting, father to son (for_every_setting); the ones that cannot run skipped, with why
    const std::vector<TuneList> tunes_list = {
        {"blocks_per_sm", BLOCKS_PER_SM_LIST},   // SWEEP_BLOCKS
        {"threads", THREADS_LIST},               // SWEEP_THREADS: threads a block
        {"values", VALUES_LIST},                 // SWEEP_VALUES: values a thread adds a round
        {"tile_rows", TILE_ROWS_LIST},           // SWEEP_ROWS: tokens a tile
        {"tile_columns", TILE_COLUMNS_LIST},     // SWEEP_COLUMNS: values of a row a tile
        {"swizzle", SWIZZLE_LIST},               // SWEEP_SWIZZLE: TMA's 128-byte swizzle, or 0 for none
        {"sets", SETS_LIST},                     // SWEEP_SETS: sets of tiles in shared memory
    };
    std::vector<Setting> settings_list;
    std::vector<Setting> skipped_list;
    printf("    %zu settings in the sweep: %s", count_settings(tunes_list), tunes_list[0].name_ptr);
    for (size_t tune_index_val = 1; tune_index_val < tunes_list.size(); tune_index_val++)
        printf(" x %s", tunes_list[tune_index_val].name_ptr);
    printf("\n");
    for_every_setting(tunes_list, [&](const TunePoint &point_val) {
        Setting setting_val = plan_setting(point_val, data_val);
        (setting_val.skipped_val.empty() ? settings_list : skipped_list).push_back(setting_val);
    });
    long fewer_resident_val = (long)std::count_if(settings_list.begin(), settings_list.end(),
                                                  [](const Setting &s) { return s.resident_val < s.blocks_per_sm_val; });
    printf("    %zu settings to run (%ld with fewer than B blocks on an SM at once: they run in waves), %zu skipped (counted below)\n",
           settings_list.size(), fewer_resident_val, skipped_list.size());

    // 3. first pass: check, then time; a launch CUDA refuses (too many resources, a bad launch) skipped, with its error
    auto start_val = std::chrono::steady_clock::now();
    std::vector<Setting> ran_list;
    for (size_t setting_index_val = 0; setting_index_val < settings_list.size(); setting_index_val++)
    {
        Setting &setting_val = settings_list[setting_index_val];
        data_val.encoded_tensor.zero_();
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
            printf("    stopped: %2d/SM  %s failed while running: %s; the GPU takes no more work after this\n", setting_val.blocks_per_sm_val,
                   describe(setting_val).c_str(), cudaGetErrorString(error_val));
            exit(EXIT_FAILURE);
        }
        setting_val.same_val = same_bits(data_val.encoded_tensor, data_val.expected_tensor);
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
    printf("    %s: %zu rows of %zu tokens, embed_dim %zu, vocab %d: %.1f MB; this GPU's peak %.0f GB/s\n", size_val.name_ptr,
           size_val.batch_size, size_val.seq_len, size_val.embed_dim, size_val.vocab_size, bytes_val / MEGA, peak_val);
    printf("    registers a thread (S: spills to local memory, its settings skipped; X: CUDA refused it, its settings skipped)\n");
    printf("    threads  swizzle ");
    for (int values_val : VALUES_LIST)
        printf("  %3d values", values_val);
    printf("\n");
    for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
        {
            printf("    %7d  %7d ", THREADS_LIST[warps_index_val], SWIZZLE_LIST[swizzle_index_val]);
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                const KernelInfo &info_val = infos_list[values_index_val][swizzle_index_val][warps_index_val];
                printf("  %8d %c", info_val.registers, !info_val.error.empty() ? 'X' : info_val.local_bytes > 0 ? 'S' : ' ');
            }
            printf("\n");
        }

    // 5b. the settings skipped: how many for each kind of reason, then each reason with the settings under it
    printf("    %zu settings ran, %zu skipped\n", settings_list.size(), skipped_list.size());
    if (!skipped_list.empty())
    {
        const char *kinds_list[] = {"kernel spills", "kernel refused", "does not fit", "no room on an SM", "tensor map refused", "launch refused"};
        for (const char *kind_ptr : kinds_list)
        {
            long count_val = (long)std::count_if(skipped_list.begin(), skipped_list.end(),
                                                 [&](const Setting &s) { return s.skipped_val.rfind(kind_ptr, 0) == 0; });
            if (count_val > 0)
                printf("      %8ld  %s\n", count_val, kind_ptr);
        }
        // a reason names only what caused it (the kernel or the tile), so many settings share one: each reason once,
        // with how many settings it skipped; every setting with its reason is in the .csv
        std::map<std::string, long> reasons_list;
        for (const Setting &setting_val : skipped_list)
            reasons_list[setting_val.skipped_val]++;
        printf("    skipped, each reason (settings under it: why)\n");
        for (const auto &reason_val : reasons_list)
            printf("      %6ld  %s\n", reason_val.second, reason_val.first.c_str());
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

    // 7. two tables for each swizzle, the fastest setting in each cell as x the current kernel's best (above 1: faster):
    //    blocks an SM against sets (1 set: wait for each tile; n: add one while TMA brings the others), and the tile's
    //    rows against its columns
    for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
    {
        printf("    swizzle %d, the fastest at each blocks an SM and sets (first pass, x the current kernel's best)\n",
               SWIZZLE_LIST[swizzle_index_val]);
        printf("    blocks/SM");
        for (int sets_val : SETS_LIST)
            printf("   %d sets", sets_val);
        printf("\n");
        for (int blocks_per_sm_val : BLOCKS_PER_SM_LIST)
        {
            printf("    %9d", blocks_per_sm_val);
            for (int sets_val : SETS_LIST)
            {
                int best_val = fastest_val([&](const Setting &s) {
                    return s.blocks_per_sm_val == blocks_per_sm_val && s.swizzle_index_val == swizzle_index_val && s.sets_val == sets_val;
                });
                if (best_val < 0)
                    printf("  %7s", "-");
                else
                    printf("  %6.2fx", baseline_ms_val / settings_list[best_val].first_ms_val);
            }
            printf("\n");
        }
        printf("    swizzle %d, the fastest at each tile rows and columns (first pass, x the current kernel's best)\n",
               SWIZZLE_LIST[swizzle_index_val]);
        printf("    rows \\ columns");
        for (int tile_columns_val : TILE_COLUMNS_LIST)
            printf("  %6d", tile_columns_val);
        printf("\n");
        for (int tile_rows_val : TILE_ROWS_LIST)
        {
            printf("    %15d", tile_rows_val);
            for (int tile_columns_val : TILE_COLUMNS_LIST)
            {
                int best_val = fastest_val([&](const Setting &s) {
                    return s.tile_rows_val == tile_rows_val && s.tile_columns_val == tile_columns_val && s.swizzle_index_val == swizzle_index_val;
                });
                if (best_val < 0)
                    printf("  %6s", "-");
                else
                    printf("  %5.2fx", baseline_ms_val / settings_list[best_val].first_ms_val);
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
        fprintf(csv_ptr, "blocks,blocks_per_sm,resident_per_sm,threads,values,tile_rows,tile_columns,swizzle,sets,registers,spill_bytes,"
                         "shared_bytes,first_ms,first_runs,final_ms,gb_per_s,same,status\n");
        for (const std::vector<Setting> *list_ptr : {&settings_list, &skipped_list})
            for (const Setting &setting_val : *list_ptr)
            {
                const KernelInfo &info_val = infos_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val];
                bool ran_val = setting_val.skipped_val.empty();
                fprintf(csv_ptr, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%.5f,%d,%.5f,%.0f,%d,\"%s\"\n", sms_val * setting_val.blocks_per_sm_val,
                        setting_val.blocks_per_sm_val, setting_val.resident_val, THREADS_LIST[setting_val.warps_index_val],
                        VALUES_LIST[setting_val.values_index_val], setting_val.tile_rows_val, setting_val.tile_columns_val,
                        SWIZZLE_LIST[setting_val.swizzle_index_val], setting_val.sets_val, info_val.registers, info_val.local_bytes,
                        setting_val.shared_bytes_val, setting_val.first_ms_val, setting_val.first_runs_val, setting_val.final_ms_val,
                        ran_val ? gigabytes_per_second(bytes_val, setting_val.first_ms_val) : 0.0, setting_val.same_val ? 1 : 0,
                        ran_val ? "ran" : setting_val.skipped_val.c_str());
            }
        fclose(csv_ptr);
        printf("    every setting: %s\n", csv_path.c_str());
    }
    else
    {
        printf("    could not write %s\n", csv_path.c_str());
    }

    // 10. every setting that ran gave encoder_bf16_forward's bits (the skipped ones are counted above, not here);
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
    // ./tune_encoder_tma_bf16: nanochat d24's size; gpt2: GPT-2 small's; x<N>: d24's with embed_dim x N (x4, x10).
    std::string size_name_val = argc > 1 ? argv[1] : "";
    EncoderSize size_val = size_name_val == "gpt2" ? GPT2_SIZE : D24_SIZE;
    std::string name_val;
    if (size_name_val.size() > 1 && size_name_val[0] == 'x')
    {
        int times_val = std::stoi(size_name_val.substr(1));
        size_val.embed_dim *= times_val;
        name_val = "nanochat d24, embed_dim x " + std::to_string(times_val);
        size_val.name_ptr = name_val.c_str();
    }
    std::string folder_val = "cuda/tests/results/" + gpu_folder_name();
    std::string log_path_val = start_run_log(folder_val.c_str(), __FILE__, "tune_", "txt");
    print_gpu();
    std::string csv_path_val = log_path_val.substr(0, log_path_val.size() - 3) + "csv";
    time_every_setting(size_val, csv_path_val);

    printf("\nall tunes done\n");
    return 0;
}
