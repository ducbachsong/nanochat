// test_sgd_8bit.cu — tests for cuda/kernels/8bit/optimizer/sgd_8bit.cuh (bfloat16 params and grads, a float master,
// the momentum in 8 bits) against libtorch's torch::optim::SGD. Every check prints its test case, input, expected and
// actual value, and ok; the first FAIL stops the program.
//
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; libtorch before any CUDA header
#include <torch/optim/sgd.h>          // torch::optim::SGD: PyTorch's own SGD, the reference
#include <torch/types.h>              // torch::tensor
#include <torch/utils.h>              // torch::NoGradGuard
#include "../../common/test_helpers.h"                    // ASSERT_EQ, ASSERT_NEAR, ASSERT_FAR
#include "../../../kernels/8bit/optimizer/sgd_8bit.cuh"   // sgd_update_8bit

// ── helpers ───────────────────────────────────────────────────────────────────

/// One run's settings, the same for both sides.
struct SgdRun {
    size_t count_val = 1024;          // values: 8 scale groups
    size_t decayed_count_val = 512;   // the first half decayed, as the matrices
    float learning_rate_val = 0.1f;
    float momentum_val = 0.9f;
    bool nesterov_val = true;
    float weight_decay_val = 5e-4f;
    float max_norm_val = 0.0f;        // 0: no clipping
    int threads_per_block_val = SGD_8BIT_THREADS_PER_BLOCK;
};

/// Makes random values from Tensor::randn, times `scale_val`, as a CPU list.
///
///   count_val   how many
///   scale_val   multiplies them
static std::vector<float> random_values(size_t count_val, float scale_val) {
    std::vector<float> numbers_list = Tensor::randn({count_val}).tolist();
    for (float& number_val : numbers_list) number_val *= scale_val;
    return numbers_list;
}

/// Gives the values rounded to bfloat16, as floats: the gradients as both sides see them.
static std::vector<float> bf16_rounded(const std::vector<float>& values_list) {
    Tensor float_tensor = Tensor::empty({values_list.size()});
    float_tensor.copy_from_cpu(values_list.data());
    return float_tensor.to(DType::BFloat16).to(DType::Float32).tolist();
}

/// Gives each value's move from the start.
static std::vector<float> moves_of(const std::vector<float>& values_list, const std::vector<float>& start_list) {
    std::vector<float> moves_list = values_list;
    for (size_t value_index_val = 0; value_index_val < moves_list.size(); value_index_val++)
        moves_list[value_index_val] -= start_list[value_index_val];
    return moves_list;
}

/// Gives a PyTorch tensor's values as a CPU list of floats.
static std::vector<float> values_of(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.detach().to(torch::kFloat32).contiguous().cpu();
    return std::vector<float>(cpu_tensor.data_ptr<float>(), cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
}

/// Runs torch::optim::SGD on the same start and gradients: the decayed part and the rest as two param groups. Gives the params.
///
///   start_list    the params' start
///   grads_lists   a gradient a step, already bfloat16-rounded
///   run_val       the settings
static std::vector<float> pytorch_sgd(const std::vector<float>& start_list, const std::vector<std::vector<float>>& grads_lists,
                                      const SgdRun& run_val) {
    // Two tensors, so only the first gets weight decay; clip_grad_norm_ over both, as one norm, before each step.
    std::vector<float> decayed_start_list(start_list.begin(), start_list.begin() + run_val.decayed_count_val);
    std::vector<float> rest_start_list(start_list.begin() + run_val.decayed_count_val, start_list.end());
    torch::Tensor decayed_tensor = torch::tensor(decayed_start_list).to(torch::kCUDA).requires_grad_(true);
    torch::Tensor rest_tensor = torch::tensor(rest_start_list).to(torch::kCUDA).requires_grad_(true);
    torch::optim::SGDOptions decayed_options_val(run_val.learning_rate_val);
    decayed_options_val.momentum(run_val.momentum_val).nesterov(run_val.nesterov_val).weight_decay(run_val.weight_decay_val);
    torch::optim::SGDOptions rest_options_val(run_val.learning_rate_val);
    rest_options_val.momentum(run_val.momentum_val).nesterov(run_val.nesterov_val).weight_decay(0.0);
    std::vector<torch::optim::OptimizerParamGroup> groups_list;
    groups_list.emplace_back(std::vector<torch::Tensor>{decayed_tensor}, std::make_unique<torch::optim::SGDOptions>(decayed_options_val));
    groups_list.emplace_back(std::vector<torch::Tensor>{rest_tensor}, std::make_unique<torch::optim::SGDOptions>(rest_options_val));
    torch::optim::SGD optimizer_val(groups_list, rest_options_val);
    for (const std::vector<float>& grads_list : grads_lists) {
        std::vector<float> decayed_grads_list(grads_list.begin(), grads_list.begin() + run_val.decayed_count_val);
        std::vector<float> rest_grads_list(grads_list.begin() + run_val.decayed_count_val, grads_list.end());
        decayed_tensor.mutable_grad() = torch::tensor(decayed_grads_list).to(torch::kCUDA);
        rest_tensor.mutable_grad() = torch::tensor(rest_grads_list).to(torch::kCUDA);
        if (run_val.max_norm_val > 0.0f)
            torch::nn::utils::clip_grad_norm_(std::vector<torch::Tensor>{decayed_tensor, rest_tensor}, run_val.max_norm_val);
        optimizer_val.step();
    }
    std::vector<float> params_list = values_of(decayed_tensor);
    std::vector<float> rest_list = values_of(rest_tensor);
    params_list.insert(params_list.end(), rest_list.begin(), rest_list.end());
    return params_list;
}

/// Runs sgd_update_8bit on the same start and gradients. Gives the float master.
///
///   start_list ... run_val   as in pytorch_sgd
static std::vector<float> our_sgd(const std::vector<float>& start_list, const std::vector<std::vector<float>>& grads_lists,
                                  const SgdRun& run_val) {
    // The momentum bytes and scales all 0 at the start; the gradients' norm worked out on the CPU each step.
    Tensor master_tensor = Tensor::empty({run_val.count_val});
    master_tensor.copy_from_cpu(start_list.data());
    Tensor param_tensor = master_tensor.to(DType::BFloat16);
    Tensor m_tensor = Tensor::zeros({run_val.count_val}, DType::E4M3);
    Tensor m_scales_tensor = Tensor::zeros({CEIL_DIV(run_val.count_val, (size_t)ADAMW_VALUES_PER_SCALE)});
    Tensor grad_norm_tensor = Tensor::zeros({1});
    for (const std::vector<float>& grads_list : grads_lists) {
        double sum_squares_val = 0.0;
        for (float g_val : grads_list) sum_squares_val += (double)g_val * g_val;
        float grad_norm_val = (float)sqrt(sum_squares_val);
        grad_norm_tensor.copy_from_cpu(&grad_norm_val);
        Tensor grad_float_tensor = Tensor::empty({run_val.count_val});
        grad_float_tensor.copy_from_cpu(grads_list.data());
        Tensor grad_tensor = grad_float_tensor.to(DType::BFloat16);
        sgd_update_8bit(param_tensor.data_ptr<nv_bfloat16>(), grad_tensor.data_ptr<nv_bfloat16>(),
                        m_tensor.data_ptr<__nv_fp8_e4m3>(), m_scales_tensor.data(), run_val.count_val,
                        run_val.decayed_count_val, grad_norm_tensor.data(), run_val.max_norm_val,
                        run_val.learning_rate_val, run_val.momentum_val, run_val.nesterov_val,
                        run_val.weight_decay_val, master_tensor.data(), run_val.threads_per_block_val);
    }
    return master_tensor.tolist();
}

/// Makes `steps_val` random gradients, bfloat16-rounded.
///
///   steps_val   how many
///   count_val   values in each
static std::vector<std::vector<float>> random_grads(int steps_val, size_t count_val) {
    std::vector<std::vector<float>> grads_lists;
    for (int step_index_val = 0; step_index_val < steps_val; step_index_val++)
        grads_lists.push_back(bf16_rounded(random_values(count_val, 1.0f)));
    return grads_lists;
}

// ── tests ─────────────────────────────────────────────────────────────────────

static void one_step_matches_pytorch(void) {
    // Expect: 1024 params (std 0.02), the first 512 decayed, one random gradient, lr 0.1, momentum 0.9, with and
    // without Nesterov: step 1 works in float (m = g) before m is stored, so the moves equal torch::optim::SGD's
    // within 1e-5 of the largest move. PyTorch really trained: its largest move is over 0.1 (lr x |g| up to about 0.3).
    Tensor::manual_seed(0);
    for (bool nesterov_val : {true, false}) {
        SgdRun run_val;
        run_val.nesterov_val = nesterov_val;
        std::vector<float> start_list = random_values(run_val.count_val, 0.02f);
        std::vector<std::vector<float>> grads_lists = random_grads(1, run_val.count_val);
        std::vector<float> expected_moves_list = moves_of(pytorch_sgd(start_list, grads_lists, run_val), start_list);
        ASSERT_NEAR(moves_of(our_sgd(start_list, grads_lists, run_val), start_list), expected_moves_list, 1e-5);
        ASSERT_FAR(expected_moves_list, std::vector<float>(run_val.count_val, 0.0f), 0.1);
    }
}

static void many_steps_stay_near_pytorch(void) {
    // Expect: 4096 params, 20 random gradients, lr 0.1, momentum 0.9 with Nesterov, weight decay 5e-4: our momentum
    // in E4M3 (3 bits after the point, up to 6% off a value) against PyTorch's in float. The rounding errors have
    // random signs, so the moves stay close. Allowed: 5% of the largest move (a first guess: tighten it to what the
    // first GPU run shows).
    Tensor::manual_seed(1);
    SgdRun run_val;
    run_val.count_val = 4096;
    run_val.decayed_count_val = 2048;
    std::vector<float> start_list = random_values(run_val.count_val, 0.02f);
    std::vector<std::vector<float>> grads_lists = random_grads(20, run_val.count_val);
    std::vector<float> expected_moves_list = moves_of(pytorch_sgd(start_list, grads_lists, run_val), start_list);
    ASSERT_NEAR(moves_of(our_sgd(start_list, grads_lists, run_val), start_list), expected_moves_list, 0.05);
}

static void clipped_step_matches_pytorch(void) {
    // Expect: gradients x 100 (norm about 3200) clipped to max_norm 1: ours (clip = 1 / (norm + 1e-6)) equals
    // clip_grad_norm_ then SGD, within 1e-5 of the largest move, on step 1.
    Tensor::manual_seed(2);
    SgdRun run_val;
    run_val.max_norm_val = 1.0f;
    std::vector<float> start_list = random_values(run_val.count_val, 0.02f);
    std::vector<std::vector<float>> grads_lists = {bf16_rounded(random_values(run_val.count_val, 100.0f))};
    std::vector<float> expected_moves_list = moves_of(pytorch_sgd(start_list, grads_lists, run_val), start_list);
    ASSERT_NEAR(moves_of(our_sgd(start_list, grads_lists, run_val), start_list), expected_moves_list, 1e-5);
}

static void a_zero_gradient_only_decays(void) {
    // Expect: an all-0 gradient, no Nesterov, lr 0.1, weight decay 0.01: g = 0.01 p, so a decayed p becomes
    // p x (1 - 0.001) = p x 0.999; the rest keep p, their momentum groups all 0 (scale 0, no NaN).
    Tensor::manual_seed(3);
    SgdRun run_val;
    run_val.nesterov_val = false;
    run_val.weight_decay_val = 0.01f;
    std::vector<float> start_list = random_values(run_val.count_val, 0.02f);
    std::vector<std::vector<float>> grads_lists = {std::vector<float>(run_val.count_val, 0.0f)};
    std::vector<float> expected_list = start_list;
    for (size_t value_index_val = 0; value_index_val < run_val.decayed_count_val; value_index_val++)
        expected_list[value_index_val] *= 0.999f;
    ASSERT_NEAR(our_sgd(start_list, grads_lists, run_val), expected_list, 1e-6);
}

static void block_size_gives_the_same_bits(void) {
    // Expect: 5 steps on 4096 params with blocks of 32, 128 and 1024 threads: the same params to the bit, as a
    // scale group is a warp whatever the block.
    Tensor::manual_seed(4);
    SgdRun run_val;
    run_val.count_val = 4096;
    run_val.decayed_count_val = 2048;
    std::vector<float> start_list = random_values(run_val.count_val, 0.02f);
    std::vector<std::vector<float>> grads_lists = random_grads(5, run_val.count_val);
    std::vector<float> default_list = our_sgd(start_list, grads_lists, run_val);
    for (int threads_per_block_val : {32, 1024}) {
        run_val.threads_per_block_val = threads_per_block_val;
        ASSERT_NEAR(our_sgd(start_list, grads_lists, run_val), default_list, 0.0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    one_step_matches_pytorch();
    many_steps_stay_near_pytorch();
    clipped_step_matches_pytorch();
    a_zero_gradient_only_decays();
    block_size_gives_the_same_bits();
    printf("\nall tests passed\n");
    return 0;
}
