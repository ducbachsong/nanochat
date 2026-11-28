// test_muon_8bit.cu — tests for cuda/kernels/8bit/optimizer/muon_8bit.cuh (bfloat16 params and grads, a float master, the
// momentum in 8 bits, Newton–Schulz on WGMMA). Every check prints its test case, input, expected and actual value,
// and ok; the first FAIL stops the program. Needs an H100 and -arch=sm_90a (WGMMA and TMA).
//
#include <ATen/ops/linalg_svdvals.h>  // at::linalg_svdvals: the singular values of Newton–Schulz's output
#include <torch/types.h>              // torch::tensor, torch::matmul
#include <torch/utils.h>              // torch::NoGradGuard
#include "../../common/test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../../kernels/8bit/optimizer/muon_8bit.cuh"     // muon_update_8bit, muon_8bit_newton_schulz

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn, times `scale_val`, as a CPU list.
///
///   shape_list   the values' shape
///   scale_val    multiplies them; 1 if not given
static std::vector<float> random_values(const Shape& shape_list, float scale_val = 1.0f) {
    std::vector<float> numbers_list = Tensor::randn(shape_list).tolist();
    for (float& number_val : numbers_list) number_val *= scale_val;
    return numbers_list;
}

/// Makes a bfloat16 GPU tensor from a CPU list.
///
///   values_list   in row order
///   shape_list    its shape
static Tensor bf16_tensor_of(const std::vector<float>& values_list, const Shape& shape_list) {
    Tensor float_tensor = Tensor::empty(shape_list);
    float_tensor.copy_from_cpu(values_list.data());
    return float_tensor.to(DType::BFloat16);
}

/// Gives a bfloat16 tensor's values as floats.
static std::vector<float> values_of(const Tensor& tensor) { return tensor.to(DType::Float32).tolist(); }

/// Makes a PyTorch GPU tensor from a CPU list.
///
///   values_list   in row order
///   rows_val, columns_val   its shape
static torch::Tensor torch_tensor_of(const std::vector<float>& values_list, size_t rows_val, size_t columns_val) {
    return torch::tensor(values_list).reshape({(int64_t)rows_val, (int64_t)columns_val}).to(torch::kCUDA);
}

/// Gives a PyTorch tensor's values as a CPU list of floats.
static std::vector<float> values_of(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.to(torch::kFloat32).contiguous().cpu();
    return std::vector<float>(cpu_tensor.data_ptr<float>(), cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
}

/// Gives Keller Jordan's zeropower_via_newtonschulz5 in PyTorch, bfloat16: the reference.
///
///   g_tensor    the direction, any float type
///   steps_val   5
static torch::Tensor pytorch_newton_schulz(const torch::Tensor& g_tensor, int steps_val) {
    // X = G / (|G| + eps) in bfloat16, turned round if tall; each step A = X Xᵀ, B = b A + c A A, X = a X + B X.
    torch::Tensor x_tensor = g_tensor.to(torch::kBFloat16);
    bool tall_val = g_tensor.size(0) > g_tensor.size(1);
    if (tall_val) x_tensor = x_tensor.t();
    x_tensor = x_tensor / (x_tensor.norm() + MUON_NORM_EPS);
    for (int step_index_val = 0; step_index_val < steps_val; step_index_val++) {
        torch::Tensor a_tensor = torch::matmul(x_tensor, x_tensor.t());
        torch::Tensor b_tensor = MUON_NEWTON_SCHULZ_B * a_tensor + MUON_NEWTON_SCHULZ_C * torch::matmul(a_tensor, a_tensor);
        x_tensor = MUON_NEWTON_SCHULZ_A * x_tensor + torch::matmul(b_tensor, x_tensor);
    }
    return tall_val ? x_tensor.t() : x_tensor;
}

/// Runs Muon in PyTorch on the same params and gradients: float momentum, Newton–Schulz in bfloat16. Gives the params.
///
///   start_list           the params' start, {rows, columns}
///   grads_lists          a gradient a step
///   rows_val, columns_val   the shape
///   learning_rate_val    every step's
///   config_val           momentum, Nesterov, weight decay, steps, the shape's scale
static std::vector<float> pytorch_muon(const std::vector<float>& start_list, const std::vector<std::vector<float>>& grads_lists,
                                       size_t rows_val, size_t columns_val, float learning_rate_val, const MuonConfig& config_val) {
    // Keller Jordan's Muon: buf.lerp_(g, 1 - momentum); g = g.lerp_(buf, momentum) (Nesterov) or buf; the update is
    // Newton–Schulz of that; p x (1 - lr wd), then p -= lr x scale x update.
    torch::NoGradGuard no_grad_val;
    torch::Tensor p_tensor = torch_tensor_of(start_list, rows_val, columns_val);
    torch::Tensor m_tensor = torch::zeros_like(p_tensor);
    float scale_val = muon_8bit_shape_scale(rows_val, columns_val, config_val.match_adamw);
    for (const std::vector<float>& grads_list : grads_lists) {
        torch::Tensor g_tensor = torch_tensor_of(grads_list, rows_val, columns_val).to(torch::kBFloat16).to(torch::kFloat32);
        m_tensor = config_val.momentum * m_tensor + (1.0f - config_val.momentum) * g_tensor;
        torch::Tensor direction_tensor =
            config_val.nesterov ? (1.0f - config_val.momentum) * g_tensor + config_val.momentum * m_tensor : m_tensor;
        torch::Tensor update_tensor = pytorch_newton_schulz(direction_tensor, config_val.newton_schulz_steps).to(torch::kFloat32);
        p_tensor = p_tensor * (1.0f - learning_rate_val * config_val.weight_decay) - learning_rate_val * scale_val * update_tensor;
    }
    return values_of(p_tensor);
}

/// Runs muon_update_8bit on the same params and gradients. Gives the float master.
///
///   start_list ... config_val   as in pytorch_muon
static std::vector<float> our_muon(const std::vector<float>& start_list, const std::vector<std::vector<float>>& grads_lists,
                                   size_t rows_val, size_t columns_val, float learning_rate_val, const MuonConfig& config_val) {
    // One matrix, its own momentum bytes and scales (all 0 at the start), no clipping.
    size_t count_val = rows_val * columns_val;
    Tensor master_tensor = Tensor::empty({count_val});
    master_tensor.copy_from_cpu(start_list.data());
    Tensor param_tensor = master_tensor.to(DType::BFloat16).view({rows_val, columns_val});
    Tensor m_tensor = Tensor::zeros({count_val}, DType::E4M3);
    Tensor m_scales_tensor = Tensor::zeros({CEIL_DIV(count_val, (size_t)ADAMW_VALUES_PER_SCALE)});
    Tensor grad_norm_tensor = Tensor::zeros({1});
    Muon8bitScratch scratch_val(count_val, std::min(rows_val, columns_val));
    for (const std::vector<float>& grads_list : grads_lists)
        muon_update_8bit(param_tensor, bf16_tensor_of(grads_list, {rows_val, columns_val}), master_tensor.data(),
                         m_tensor.data_ptr<__nv_fp8_e4m3>(), m_scales_tensor.data(), scratch_val, grad_norm_tensor.data(),
                         0.0f, learning_rate_val, config_val);
    return master_tensor.tolist();
}

/// Gives each value's move from the start.
static std::vector<float> moves_of(const std::vector<float>& values_list, const std::vector<float>& start_list) {
    std::vector<float> moves_list = values_list;
    for (size_t value_index_val = 0; value_index_val < moves_list.size(); value_index_val++)
        moves_list[value_index_val] -= start_list[value_index_val];
    return moves_list;
}

// ── tests ─────────────────────────────────────────────────────────────────────

static void newton_schulz_matches_pytorch(void) {
    // Expect: a normalized random direction, wide {128, 384}, tall {384, 128} and square {256, 256}: our 5 steps
    // (WGMMA, X read turned for a tall one) near PyTorch's bfloat16 zeropower_via_newtonschulz5 (X turned round).
    // Both round to bfloat16 after each product, ours once per a X + B X where PyTorch rounds B X first, so a few %
    // apart. Allowed: 5% of the largest value (a first guess: tighten it to what the first H100 run shows).
    Tensor::manual_seed(0);
    std::vector<std::pair<size_t, size_t>> shapes_list = {{128, 384}, {384, 128}, {256, 256}};
    for (const std::pair<size_t, size_t>& shape_val : shapes_list) {
        size_t rows_val = shape_val.first, columns_val = shape_val.second;
        std::vector<float> g_list = random_values({rows_val, columns_val});
        double sum_squares_val = 0.0;
        for (float g_val : g_list) sum_squares_val += (double)g_val * g_val;
        for (float& g_val : g_list) g_val /= (float)sqrt(sum_squares_val);   // norm 1: both sides start the same
        Tensor direction = bf16_tensor_of(g_list, {rows_val, columns_val});
        Muon8bitScratch scratch_val(rows_val * columns_val, std::min(rows_val, columns_val));
        muon_8bit_newton_schulz(direction, scratch_val, 5);
        std::vector<float> expected_list = values_of(pytorch_newton_schulz(torch_tensor_of(g_list, rows_val, columns_val), 5));
        ASSERT_NEAR(values_of(direction), expected_list, 0.05);
    }
}

static void newton_schulz_singular_values_land_near_one(void) {
    // Expect: a normalized random {256, 512}: its singular values 0.018 .. 0.107 before; after 5 steps each in
    // about 0.68 .. 1.13 (Keller's quintic, bfloat16). Allowed: 0.5 .. 1.3.
    Tensor::manual_seed(1);
    size_t rows_val = 256, columns_val = 512;
    std::vector<float> g_list = random_values({rows_val, columns_val});
    double sum_squares_val = 0.0;
    for (float g_val : g_list) sum_squares_val += (double)g_val * g_val;
    for (float& g_val : g_list) g_val /= (float)sqrt(sum_squares_val);
    Tensor direction = bf16_tensor_of(g_list, {rows_val, columns_val});
    Muon8bitScratch scratch_val(rows_val * columns_val, rows_val);
    muon_8bit_newton_schulz(direction, scratch_val, 5);
    std::vector<float> singular_values_list = values_of(at::linalg_svdvals(torch_tensor_of(values_of(direction), rows_val, columns_val)));
    float smallest_val = *std::min_element(singular_values_list.begin(), singular_values_list.end());
    float largest_val = *std::max_element(singular_values_list.begin(), singular_values_list.end());
    ASSERT_EQ(smallest_val > 0.5f, true);
    ASSERT_EQ(largest_val < 1.3f, true);
}

static void one_step_matches_pytorch_muon(void) {
    // Expect: a {256, 768} param (std 0.02) and one random gradient, lr 0.02, momentum 0.95 with Nesterov: ours
    // (the first step's direction is in float, before the momentum is stored in 8 bits) near PyTorch's Muon, both
    // Newton–Schulz in bfloat16. Wide, tall and Moonshot's scale. Allowed: 5% of the largest move.
    Tensor::manual_seed(2);
    MuonConfig config_val;
    std::vector<std::pair<size_t, size_t>> shapes_list = {{256, 768}, {768, 256}};
    for (bool match_adamw_val : {false, true}) {
        config_val.match_adamw = match_adamw_val;
        for (const std::pair<size_t, size_t>& shape_val : shapes_list) {
            size_t rows_val = shape_val.first, columns_val = shape_val.second;
            std::vector<float> start_list = random_values({rows_val, columns_val}, 0.02f);
            std::vector<std::vector<float>> grads_lists = {random_values({rows_val, columns_val})};
            std::vector<float> ours_list = our_muon(start_list, grads_lists, rows_val, columns_val, 0.02f, config_val);
            std::vector<float> expected_list = pytorch_muon(start_list, grads_lists, rows_val, columns_val, 0.02f, config_val);
            ASSERT_NEAR(moves_of(ours_list, start_list), moves_of(expected_list, start_list), 0.05);
        }
    }
}

static void eight_bit_momentum_stays_near_float_momentum(void) {
    // Expect: 20 steps of random gradients on a {256, 512} param, weight decay 0.1: our momentum in E4M3 (3 bits
    // after the point, up to 6% off a value) against PyTorch's in float. Newton–Schulz spreads each value's error
    // over the matrix, so the moves stay close. Allowed: 10% of the largest move (a first guess).
    Tensor::manual_seed(3);
    MuonConfig config_val;
    config_val.weight_decay = 0.1f;
    size_t rows_val = 256, columns_val = 512;
    std::vector<float> start_list = random_values({rows_val, columns_val}, 0.02f);
    std::vector<std::vector<float>> grads_lists;
    for (int step_index_val = 0; step_index_val < 20; step_index_val++)
        grads_lists.push_back(random_values({rows_val, columns_val}));
    std::vector<float> ours_list = our_muon(start_list, grads_lists, rows_val, columns_val, 0.02f, config_val);
    std::vector<float> expected_list = pytorch_muon(start_list, grads_lists, rows_val, columns_val, 0.02f, config_val);
    ASSERT_NEAR(moves_of(ours_list, start_list), moves_of(expected_list, start_list), 0.1);
}

static void a_zero_gradient_only_decays(void) {
    // Expect: an all-0 gradient: the direction is 0, its norm 0 (MUON_NORM_EPS keeps the division finite), Newton–
    // Schulz keeps 0, so p = p x (1 - lr x weight_decay) = p x 0.998 at lr 0.02, weight decay 0.1. No NaN.
    Tensor::manual_seed(4);
    MuonConfig config_val;
    config_val.weight_decay = 0.1f;
    size_t rows_val = 128, columns_val = 256;
    std::vector<float> start_list = random_values({rows_val, columns_val}, 0.02f);
    std::vector<std::vector<float>> grads_lists = {std::vector<float>(rows_val * columns_val, 0.0f)};
    std::vector<float> ours_list = our_muon(start_list, grads_lists, rows_val, columns_val, 0.02f, config_val);
    std::vector<float> expected_list = start_list;
    for (float& value_val : expected_list) value_val *= 0.998f;
    ASSERT_NEAR(ours_list, expected_list, 1e-6);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    newton_schulz_matches_pytorch();
    newton_schulz_singular_values_land_near_one();
    one_step_matches_pytorch_muon();
    eight_bit_momentum_stays_near_float_momentum();
    a_zero_gradient_only_decays();
    printf("\nall tests passed\n");
    return 0;
}
