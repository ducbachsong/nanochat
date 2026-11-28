// model_bf16.cuh — the whole GPT-2 model in bfloat16.
//
//     tensor                              type in memory   shape
//     token_ids, targets                  int              {batch_size, seq_len}
//     activations                         bfloat16         {batch_size, seq_len, ...}
//     mean, rstd, logsumexp, losses       float            {batch_size, seq_len, ...}
//     query_rstd, key_rstd                float            {batch_size, num_heads, seq_len}
//     model_buffer: params, grads, m, v   bfloat16         flat
//     buffer: master params, grad_norm    float            flat
#ifndef MODEL_BF16_CUH
#define MODEL_BF16_CUH

#include <utility>
#include <vector>
#include "common/tensor.cuh"
#include "common/model_params.cuh"                  // GPT2Config, BlockParams, GPT2Params, the starting values, the mean loss
#include "kernels/bf16/attention_bf16.cuh"         // the steps
#include "kernels/bf16/attention_flash_bf16.cuh"   // flash attention
#include "kernels/bf16/encoder_bf16.cuh"
#include "kernels/bf16/layer_norm_bf16.cuh"
#include "kernels/bf16/linear_bf16.cuh"            // linear_bf16_bias_grad: the linears' bias gradients
#include "kernels/bf16/qk_norm_bf16.cuh"           // QK-norm before attention (GPT2Config::qk_norm)
#include "kernels/bf16/residual_bf16.cuh"
#include "kernels/bf16/softmax_loss_bf16.cuh"      // the softmax, the loss and its gradient from the bfloat16 logits
#include "kernels/bf16/optimizer/adamw_bf16.cuh"   // AdamWBf16's update
// GPT-2's 15 matrix products, each a kernel of its own with its own setting (its file's defines, from its
// cuda/tests/bf16/prod/tune_<product>_bf16.cu): a layer's forward, its backward's input_grad and weight_grad, and the head's
#include "kernels/bf16/prod/qkv_bf16.cuh"
#include "kernels/bf16/prod/qkv_input_grad_bf16.cuh"
#include "kernels/bf16/prod/qkv_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/attention_output_bf16.cuh"
#include "kernels/bf16/prod/attention_output_input_grad_bf16.cuh"
#include "kernels/bf16/prod/attention_output_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/mlp_up_bf16.cuh"                    // with GELU: gelu_output written beside mlp_up_output
#include "kernels/bf16/prod/mlp_up_input_grad_bf16.cuh"
#include "kernels/bf16/prod/mlp_up_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/mlp_down_bf16.cuh"
#include "kernels/bf16/prod/mlp_down_input_grad_bf16.cuh"       // with GELU's backward: times gelu'(mlp_up_output)
#include "kernels/bf16/prod/mlp_down_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/head_bf16.cuh"
#include "kernels/bf16/prod/head_input_grad_bf16.cuh"
#include "kernels/bf16/prod/head_weight_grad_bf16.cuh"

/// Everything the bfloat16 model works in, kept for the backward or used as scratch.
struct GPT2ActivationsBf16 {
    // Everything the model works in: mean, rstd, attention_logsumexp and the per-row tensors float, the rest
    // bfloat16. Kept for the backward: the model's tensors, and a block's as lists, one tensor per layer
    // (ln1_output[l] is layer l's); with flash attention, attention_logsumexp in place of attention_probs. With the
    // _scratch suffix: used only inside one forward or backward, one of each, shared by all the layers.
    bool flash_attention;                                   // flash attention (attention_bf16_flash_fits), or the steps
    std::vector<size_t> attention_windows;                  // each layer's sliding window (gpt2_attention_window); 0: none
    bool qk_norm = false;                                   // GPT2Config::qk_norm
    // ── kept for the backward: the model's ──
    Tensor encoded;                                         // {B, T, D}: the encoder's output, block 0's input
    std::vector<Tensor> block_outputs;                      // {B, T, D} each: block l's output, block l + 1's input
    Tensor lnf_output;                                      // {B, T, D}
    Tensor lnf_mean;                                        // {B, T}
    Tensor lnf_rstd;                                        // {B, T}
    Tensor logits;                                          // {B, T, padded_vocab_size}
    Tensor losses;                                          // {B, T}
    Tensor max_of_each_row;                                 // {B, T}: row_max · log2 e, as softmax_loss_bf16 keeps it
    Tensor sum_of_each_row;                                 // {B, T}
    // ── kept for the backward: one per layer ──
    std::vector<Tensor> ln1_output;                         // {B, T, D}
    std::vector<Tensor> ln1_mean;                           // {B, T}
    std::vector<Tensor> ln1_rstd;                           // {B, T}
    std::vector<Tensor> qkv;                                // {B, T, 3D}: attention reads query, key, value here (with
                                                            // QK-norm, query and key after the norm)
    std::vector<Tensor> query_rstd;                         // {B, NH, T}: QK-norm's 1 / RMS of each query head row
    std::vector<Tensor> key_rstd;                           // {B, NH, T}: ...of each key head row (none without QK-norm)
    std::vector<Tensor> attention_probs;                    // {B, NH, T, T}; empty with flash attention
    std::vector<Tensor> attention_logsumexp;                // {B, NH, T}: flash attention's; empty without
    std::vector<Tensor> attention_output;                   // {B, T, D}: the heads merged, before the projection
    std::vector<Tensor> residual2;                          // {B, T, D}: input + the attention branch
    std::vector<Tensor> ln2_output;                         // {B, T, D}
    std::vector<Tensor> ln2_mean;                           // {B, T}
    std::vector<Tensor> ln2_rstd;                           // {B, T}
    std::vector<Tensor> mlp_up_output;                      // {B, T, 4D}
    std::vector<Tensor> gelu_output;                        // {B, T, 4D}
    // ── only inside one forward or backward, shared by all the layers ──
    Tensor branch_output_scratch;                           // {B, T, D}: a branch's output, before its residual add
    Tensor residual2_grad_scratch;                          // {B, T, D}
    Tensor branch_output_grad_scratch;                      // {B, T, D}
    Tensor ln_output_grad_scratch;                          // {B, T, D}
    Tensor attention_output_grad_scratch;                   // {B, T, D}
    Tensor mlp_up_output_grad_scratch;                      // {B, T, 4D}
    Tensor qkv_grad_scratch;                                // {B, T, 3D}
    Tensor attention_probs_grad_scratch;                    // {B, NH, T, T}: the attention steps'; none with flash
    Tensor attention_row_dot_scratch;                       // {B, NH, T} float: flash attention's; none without
    Tensor logits_grad_scratch;                             // {B, T, padded_vocab_size}
    Tensor lnf_output_grad_scratch;                         // {B, T, D}
    Tensor residual_grads_scratch[2];                       // {B, T, D}: a block's output_grad and input_grad, in turns

    GPT2ActivationsBf16(size_t batch_size, size_t seq_len, const GPT2Config& config);
};

// ── one block ─────────────────────────────────────────────────────────────────
void block_bf16_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                        const GPT2ActivationsBf16& activations, size_t layer);
void block_bf16_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                         const Tensor& input, const BlockParams& params, const GPT2ActivationsBf16& activations,
                         size_t layer);

// ── the whole model ───────────────────────────────────────────────────────────
void gpt2_bf16_forward(const GPT2ActivationsBf16& activations, const Tensor& token_ids, const Tensor& targets,
                       const GPT2Params& params, const GPT2Config& config);
void gpt2_bf16_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                        const GPT2Params& params, const GPT2ActivationsBf16& activations, const GPT2Config& config,
                        size_t micro_batches = 1);
void gpt2_bf16_forward_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                                const GPT2Params& params, const GPT2ActivationsBf16& activations, const GPT2Config& config,
                                size_t micro_batches = 1);
float gpt2_bf16_train_step(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                           const GPT2Params& params, const GPT2ActivationsBf16& activations, const GPT2Config& config,
                           size_t micro_batches = 1);

// ── the params, grads and AdamW ──────────────────────────────────────────────
/// The bfloat16 model's params and grads, and AdamW's float master and bfloat16 m and v for this GPU's slice.
class AdamWBf16 : public Optimizer {
    // The model's params and grads (bfloat16, all of them on every GPU), and AdamW's float master and bfloat16 m and
    // v for this GPU's slice; the update: one launch of adamw_update_bf16.
public:
    // ── making it ───────────────────────────────────────────────────────────────
    AdamWBf16(const std::vector<Shape>& param_shapes, const AdamWConfig& config,  // the matrices first
              int rank = 0, int world_size = 1);
    void copy_params_to_master() override;                              // once, after filling param(k)

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& param(int k) const override;                          // parameter tensor k: bfloat16
    const Tensor& grad(int k) const override;                           // its gradient: bfloat16
    const Tensor& all_params() const override;                          // every param(k), one flat tensor
    const Tensor& all_grads() const override;                           // every grad(k), one flat tensor
    const Tensor& master_param(int k) const;                            // the float copy the update changes
    Tensor master_slice() const override;                               // this slice's master, flat
    const Tensor& m(int k) const;                                       // AdamW's m for parameter k: bfloat16
    const Tensor& v(int k) const;                                       // ...and v
    const Tensor& m_slice() const override;                             // m for this GPU's slice, flat
    const Tensor& v_slice() const override;                             // ...and v

protected:
    void update(int t) override;                                        // AdamW's step t: one launch

private:
    TensorBuffer model_buffer_;          // bfloat16: the params, grads, m and v
    int params_group_index_;             // model_buffer_'s group of the params
    int grads_group_index_;              // ...of the grads
    int m_group_index_;                  // ...of m, this slice only
    int v_group_index_;                  // ...of v
    int master_group_index_;             // buffer_'s group of the master params, this slice only
};

// ══ the code, in the order of the lists above ══════════════════════════════════

// ── the activations ───────────────────────────────────────────────────────────

/// Makes everything the model works in, all 0.
///
///   batch_size, seq_len   B and T; T at most config.max_seq_len
///   config                the sizes
inline GPT2ActivationsBf16::GPT2ActivationsBf16(size_t batch_size, size_t seq_len, const GPT2Config& config) {
    // The model's kept tensors, a block's for each layer, and the scratch. With flash attention,
    // attention_logsumexp and attention_row_dot_scratch in place of the attention steps' attention_probs and
    // attention_probs_grad_scratch.
    size_t B = batch_size;
    size_t T = seq_len;
    size_t D = config.embed_dim;
    size_t NH = config.num_heads;
    DType bf16 = DType::BFloat16;
    flash_attention = attention_bf16_flash_fits(T, D / NH);
    for (size_t l = 0; l < config.num_layers; l++) attention_windows.push_back(gpt2_attention_window(config, l));
    qk_norm = config.qk_norm;
    if (config.attention_window != 0 && !flash_attention) {
        fprintf(stderr, "error: a sliding window (GPT2Config::attention_window) needs flash attention: head_size %d, "
                        "seq_len a multiple of %d, an H100\n", FLASH_HEAD_SIZE, FLASH_TILE);
        exit(EXIT_FAILURE);
    }
    // 1. kept: the model's
    encoded = Tensor::zeros({B, T, D}, bf16);
    lnf_output = Tensor::zeros({B, T, D}, bf16);
    lnf_mean = Tensor::zeros({B, T}), lnf_rstd = Tensor::zeros({B, T});
    logits = Tensor::zeros({B, T, config.padded_vocab_size}, bf16);
    losses = Tensor::zeros({B, T}), max_of_each_row = Tensor::zeros({B, T}), sum_of_each_row = Tensor::zeros({B, T});
    // 2. kept: one per layer
    for (size_t l = 0; l < config.num_layers; l++) {
        block_outputs.push_back(Tensor::zeros({B, T, D}, bf16));
        ln1_output.push_back(Tensor::zeros({B, T, D}, bf16));
        ln1_mean.push_back(Tensor::zeros({B, T})), ln1_rstd.push_back(Tensor::zeros({B, T}));
        qkv.push_back(Tensor::zeros({B, T, 3 * D}, bf16));
        if (config.qk_norm) query_rstd.push_back(Tensor::zeros({B, NH, T})), key_rstd.push_back(Tensor::zeros({B, NH, T}));
        if (flash_attention) attention_logsumexp.push_back(Tensor::zeros({B, NH, T}));
        else attention_probs.push_back(Tensor::zeros({B, NH, T, T}, bf16));
        attention_output.push_back(Tensor::zeros({B, T, D}, bf16));
        residual2.push_back(Tensor::zeros({B, T, D}, bf16));
        ln2_output.push_back(Tensor::zeros({B, T, D}, bf16));
        ln2_mean.push_back(Tensor::zeros({B, T})), ln2_rstd.push_back(Tensor::zeros({B, T}));
        mlp_up_output.push_back(Tensor::zeros({B, T, 4 * D}, bf16));
        gelu_output.push_back(Tensor::zeros({B, T, 4 * D}, bf16));
    }
    // 3. scratch
    branch_output_scratch = Tensor::zeros({B, T, D}, bf16);
    residual2_grad_scratch = Tensor::zeros({B, T, D}, bf16), branch_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    ln_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    attention_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    mlp_up_output_grad_scratch = Tensor::zeros({B, T, 4 * D}, bf16);
    qkv_grad_scratch = Tensor::zeros({B, T, 3 * D}, bf16);
    if (flash_attention) attention_row_dot_scratch = Tensor::zeros({B, NH, T});
    else attention_probs_grad_scratch = Tensor::zeros({B, NH, T, T}, bf16);
    logits_grad_scratch = Tensor::zeros({B, T, config.padded_vocab_size}, bf16);
    lnf_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    residual_grads_scratch[0] = Tensor::zeros({B, T, D}, bf16), residual_grads_scratch[1] = Tensor::zeros({B, T, D}, bf16);
}

// ── one block ─────────────────────────────────────────────────────────────────

/// Writes output = block `layer` applied to input, keeping what the backward needs in that layer's activations.
///
///   output, input   {batch_size, seq_len, embed_dim} bfloat16; not the same tensor (the backward reads input)
///   params          the block's
///   activations     the layer's lists written, the scratch used
///   layer           which block: 0 .. num_layers - 1
inline void block_bf16_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                               const GPT2ActivationsBf16& activations, size_t layer) {
    // the attention branch
    layer_norm_bf16_forward(activations.ln1_output[layer], activations.ln1_mean[layer], activations.ln1_rstd[layer], input,
                            params.ln1_weight, params.ln1_bias);
    qkv_bf16(activations.qkv[layer], activations.ln1_output[layer], params.qkv_weight, params.qkv_bias);
    // QK-norm: query's and key's head rows to RMS 1.2 in place; qkv then keeps them after the norm, for attention's
    // backward and the norm's
    if (activations.qk_norm)
        qk_norm_bf16_forward(activations.qkv[layer], activations.query_rstd[layer], activations.key_rstd[layer]);
    if (activations.flash_attention)
        attention_bf16_flash_forward(activations.attention_output[layer], activations.attention_logsumexp[layer],
                                     activations.qkv[layer], activations.attention_windows[layer]);
    else
        attention_bf16_forward(activations.attention_output[layer], activations.attention_probs[layer], activations.qkv[layer]);
    attention_output_bf16(activations.branch_output_scratch, activations.attention_output[layer], params.attention_out_weight,
                          params.attention_out_bias);
    residual_bf16_forward(activations.residual2[layer], input, activations.branch_output_scratch);
    // the MLP branch; mlp_up and GELU in one product: GELU as each pair of mlp_up_output is stored
    layer_norm_bf16_forward(activations.ln2_output[layer], activations.ln2_mean[layer], activations.ln2_rstd[layer],
                            activations.residual2[layer], params.ln2_weight, params.ln2_bias);
    mlp_up_bf16(activations.mlp_up_output[layer], activations.gelu_output[layer], activations.ln2_output[layer],
                params.mlp_up_weight, params.mlp_up_bias);
    mlp_down_bf16(activations.branch_output_scratch, activations.gelu_output[layer], params.mlp_down_weight,
                  params.mlp_down_bias);
    residual_bf16_forward(output, activations.residual2[layer], activations.branch_output_scratch);
}

/// Writes input_grad and adds block `layer`'s parameter gradients onto param_grads.
///
///   input_grad                     written over
///   param_grads                    added onto: 0 before a step's first backward
///   output_grad                    the gradient from the next block
///   input, params, activations     the forward's; activations' scratch used
///   layer                          which block: 0 .. num_layers - 1
inline void block_bf16_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                                const Tensor& input, const BlockParams& params, const GPT2ActivationsBf16& activations,
                                size_t layer) {
    // The forward's steps the other way.
    // the MLP branch; LayerNorm 2 adds its part onto residual2_grad
    residual_bf16_backward(activations.residual2_grad_scratch, activations.branch_output_grad_scratch, output_grad);
    // mlp_down's backward and GELU's in one: its input_grad times gelu' as each pair is stored
    mlp_down_input_grad_bf16(activations.mlp_up_output_grad_scratch, activations.branch_output_grad_scratch,
                             params.mlp_down_weight, activations.mlp_up_output[layer]);
    mlp_down_weight_grad_bf16(param_grads.mlp_down_weight, activations.branch_output_grad_scratch, activations.gelu_output[layer]);
    if (param_grads.has_linear_bias()) linear_bf16_bias_grad(param_grads.mlp_down_bias, activations.branch_output_grad_scratch);
    mlp_up_input_grad_bf16(activations.ln_output_grad_scratch, activations.mlp_up_output_grad_scratch, params.mlp_up_weight);
    mlp_up_weight_grad_bf16(param_grads.mlp_up_weight, activations.mlp_up_output_grad_scratch, activations.ln2_output[layer]);
    if (param_grads.has_linear_bias()) linear_bf16_bias_grad(param_grads.mlp_up_bias, activations.mlp_up_output_grad_scratch);
    layer_norm_bf16_backward(activations.residual2_grad_scratch, param_grads.ln2_weight, param_grads.ln2_bias,
                             activations.ln_output_grad_scratch, activations.residual2[layer], params.ln2_weight,
                             activations.ln2_mean[layer], activations.ln2_rstd[layer]);
    // the attention branch; LayerNorm 1 adds its part onto input_grad
    residual_bf16_backward(input_grad, activations.branch_output_grad_scratch, activations.residual2_grad_scratch);
    attention_output_input_grad_bf16(activations.attention_output_grad_scratch, activations.branch_output_grad_scratch,
                                     params.attention_out_weight);
    attention_output_weight_grad_bf16(param_grads.attention_out_weight, activations.branch_output_grad_scratch,
                                      activations.attention_output[layer]);
    if (param_grads.has_linear_bias())
        linear_bf16_bias_grad(param_grads.attention_out_bias, activations.branch_output_grad_scratch);
    if (activations.flash_attention)
        attention_bf16_flash_backward(activations.qkv_grad_scratch, activations.attention_output_grad_scratch,
                                      activations.attention_output[layer], activations.qkv[layer],
                                      activations.attention_logsumexp[layer], activations.attention_row_dot_scratch,
                                      ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK, activations.attention_windows[layer]);
    else
        attention_bf16_backward(activations.qkv_grad_scratch, activations.attention_output_grad_scratch, activations.qkv[layer],
                                activations.attention_probs[layer], activations.attention_probs_grad_scratch);
    //    QK-norm's backward: query's and key's gradient after the norm turned into the one before it, in place, from
    //    query and key after the norm (qkv as the forward left it); before the qkv linear's products read qkv_grad
    if (activations.qk_norm)
        qk_norm_bf16_backward(activations.qkv_grad_scratch, activations.qkv[layer], activations.query_rstd[layer],
                              activations.key_rstd[layer]);
    qkv_input_grad_bf16(activations.ln_output_grad_scratch, activations.qkv_grad_scratch, params.qkv_weight);
    qkv_weight_grad_bf16(param_grads.qkv_weight, activations.qkv_grad_scratch, activations.ln1_output[layer]);
    if (param_grads.has_linear_bias()) linear_bf16_bias_grad(param_grads.qkv_bias, activations.qkv_grad_scratch);
    layer_norm_bf16_backward(input_grad, param_grads.ln1_weight, param_grads.ln1_bias, activations.ln_output_grad_scratch,
                             input, params.ln1_weight, activations.ln1_mean[layer], activations.ln1_rstd[layer]);
}

// ── the whole model ───────────────────────────────────────────────────────────

/// Works out the logits from the tokens: the encoder, the blocks, LayerNorm f and the head.
///
///   activations, token_ids, params   as in gpt2_bf16_forward
inline void gpt2_bf16_forward_to_logits(const GPT2ActivationsBf16& activations, const Tensor& token_ids,
                                        const GPT2Params& params) {
    // What the forward and a training micro-batch share.
    size_t num_layers = params.blocks.size();
    encoder_bf16_forward(activations.encoded, token_ids, params.wte, params.wpe);
    for (size_t l = 0; l < num_layers; l++) {
        const Tensor& input = l == 0 ? activations.encoded : activations.block_outputs[l - 1];
        block_bf16_forward(activations.block_outputs[l], input, params.blocks[l], activations, l);
    }
    layer_norm_bf16_forward(activations.lnf_output, activations.lnf_mean, activations.lnf_rstd,
                            activations.block_outputs[num_layers - 1], params.lnf_weight, params.lnf_bias);
    head_bf16(activations.logits, activations.lnf_output, params.wte);          // the head: no bias
}

/// Works out each row's loss into activations.losses, keeping what the backward needs.
///
///   activations          written
///   token_ids, targets   {batch_size, seq_len} ints, every id < vocab_size
///   params               the model's, bfloat16
///   config               the sizes
inline void gpt2_bf16_forward(const GPT2ActivationsBf16& activations, const Tensor& token_ids, const Tensor& targets,
                              const GPT2Params& params, const GPT2Config& config) {
    // With a softcap (GPT2Config::logit_softcap), each logit capped to c · tanh(logit / c) before the softmax.
    gpt2_bf16_forward_to_logits(activations, token_ids, params);
    softmax_loss_bf16_forward(activations.losses, activations.max_of_each_row, activations.sum_of_each_row, activations.logits,
                              targets, config.vocab_size, SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK, config.logit_softcap);
}

/// Adds every parameter's gradient onto param_grads from the logits' gradient on.
///
///   param_grads, token_ids, params, activations   as in gpt2_bf16_backward; activations.logits_grad_scratch read
inline void gpt2_bf16_backward_from_logits_grad(const GPT2Params& param_grads, const Tensor& token_ids,
                                                const GPT2Params& params, const GPT2ActivationsBf16& activations) {
    // The head's backward, LayerNorm f's, the blocks' the other way, the encoder's; what the backward and a
    // training micro-batch share.
    size_t num_layers = params.blocks.size();
    head_input_grad_bf16(activations.lnf_output_grad_scratch, activations.logits_grad_scratch, params.wte);
    head_weight_grad_bf16(param_grads.wte, activations.logits_grad_scratch, activations.lnf_output);
    Tensor output_grad = activations.residual_grads_scratch[0];
    Tensor input_grad = activations.residual_grads_scratch[1];
    output_grad.zero_();                                    // LayerNorm's backward adds onto it
    layer_norm_bf16_backward(output_grad, param_grads.lnf_weight, param_grads.lnf_bias, activations.lnf_output_grad_scratch,
                             activations.block_outputs[num_layers - 1], params.lnf_weight, activations.lnf_mean,
                             activations.lnf_rstd);
    for (size_t l = num_layers; l-- > 0;) {                 // l = num_layers - 1, ..., 0
        const Tensor& input = l == 0 ? activations.encoded : activations.block_outputs[l - 1];
        block_bf16_backward(input_grad, param_grads.blocks[l], output_grad, input, params.blocks[l], activations, l);
        std::swap(output_grad, input_grad);                 // block l - 1's output_grad
    }
    encoder_bf16_backward(param_grads.wte, param_grads.wpe, output_grad, token_ids);
}

/// Adds every parameter's gradient onto param_grads for the mean loss, after gpt2_bf16_forward.
///
///   param_grads                   added onto: 0 before a step's first backward
///   token_ids, targets            the forward's
///   params, activations           the forward's; activations' scratch used
///   config                        the sizes
///   micro_batches                 per step: each row's loss_grad is 1 / (micro_batches x B x T)
inline void gpt2_bf16_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                               const GPT2Params& params, const GPT2ActivationsBf16& activations,
                               const GPT2Config& config, size_t micro_batches) {
    // The logits' gradient from the forward's row_max and row_sum (back through the softcap, if any), then the rest
    // the other way.
    // loss = sum of the row losses / (micro_batches x B x T)
    float loss_grad_val = (float)(1.0 / ((double)micro_batches * activations.losses.numel()));
    softmax_loss_bf16_backward(activations.logits_grad_scratch, activations.logits, targets, activations.max_of_each_row,
                               activations.sum_of_each_row, config.vocab_size, loss_grad_val,
                               SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK, config.logit_softcap);
    gpt2_bf16_backward_from_logits_grad(param_grads, token_ids, params, activations);
}

/// Runs one training micro-batch on the GPU: forward, loss and logits' gradient in one pass, then the backward.
///
///   param_grads, token_ids, targets, params, activations, config, micro_batches   as in gpt2_bf16_backward
inline void gpt2_bf16_forward_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                                       const GPT2Params& params, const GPT2ActivationsBf16& activations,
                                       const GPT2Config& config, size_t micro_batches) {
    // Runs one training micro-batch on the GPU: the forward to the logits, the loss and the logits' gradient in one
    // pass over them (softmax_loss_bf16_forward_backward: the logits read from memory once, not twice), then the rest
    // of the backward. The same results as gpt2_bf16_forward then gpt2_bf16_backward, to the last bit.
    gpt2_bf16_forward_to_logits(activations, token_ids, params);
    float loss_grad_val = (float)(1.0 / ((double)micro_batches * activations.losses.numel()));
    softmax_loss_bf16_forward_backward(activations.losses, activations.max_of_each_row, activations.sum_of_each_row,
                                       activations.logits_grad_scratch, activations.logits, targets, config.vocab_size,
                                       loss_grad_val, SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK, config.logit_softcap);
    gpt2_bf16_backward_from_logits_grad(param_grads, token_ids, params, activations);
}

/// Runs one micro-batch of a training step (gpt2_bf16_forward_backward) and gives its mean loss.
///
///   param_grads, token_ids, targets, params, activations, config, micro_batches   as in gpt2_bf16_backward
inline float gpt2_bf16_train_step(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                                  const GPT2Params& params, const GPT2ActivationsBf16& activations, const GPT2Config& config,
                                  size_t micro_batches) {
    // So waits for the GPU.
    gpt2_bf16_forward_backward(param_grads, token_ids, targets, params, activations, config, micro_batches);
    return gpt2_mean_loss(activations);
}

// ── AdamWBf16: making it ─────────────────────────────────────────────────────

/// Makes a bfloat16 AdamW, all 0.
///
///   param_shapes        the matrices first
///   config              the settings
///   rank, world_size    this GPU of how many; the values must split into slices of a multiple of 4
inline AdamWBf16::AdamWBf16(const std::vector<Shape>& param_shapes, const AdamWConfig& config, int rank,
                            int world_size)
    : Optimizer(param_shapes, config, DType::BFloat16, rank, world_size) {
    // Makes a bfloat16 AdamW: the master for this slice and grad_norm in the float buffer; the params, grads, and
    // m and v for this slice in the bfloat16 one; all 0.
    master_group_index_ = buffer_.add_many(slice_shapes());
    allocate();
    params_group_index_ = model_buffer_.add_many(param_shapes);
    grads_group_index_ = model_buffer_.add_many(param_shapes);
    m_group_index_ = model_buffer_.add_many(slice_shapes());
    v_group_index_ = model_buffer_.add_many(slice_shapes());
    model_buffer_.allocate(DType::BFloat16);
}

/// Copies this slice of the model's params into the float master, once, before the first step.
inline void AdamWBf16::copy_params_to_master() {
    // Copies this slice of the model's params into the float master, once, after filling param(k) and before the
    // first step. Exact: every bfloat16 is a float.
    buffer_.group(master_group_index_).copy_(all_params().narrow(0, slice_start(), slice_size()));
}

// ── AdamWBf16: reading ───────────────────────────────────────────────────────

/// Gives parameter tensor k, bfloat16.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamWBf16::param(int k) const { return model_buffer_.tensor(params_group_index_, k); }

/// Gives the gradient of parameter k, bfloat16.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamWBf16::grad(int k) const { return model_buffer_.tensor(grads_group_index_, k); }

/// Gives every param(k) as one flat tensor.
inline const Tensor& AdamWBf16::all_params() const { return model_buffer_.group(params_group_index_); }

/// Gives every grad(k) as one flat tensor: one launch zeroes them all.
inline const Tensor& AdamWBf16::all_grads() const { return model_buffer_.group(grads_group_index_); }

/// Gives the float copy of parameter k the update changes. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamWBf16::master_param(int k) const {
    stop_unless_one_gpu("master_param(k)");
    return buffer_.tensor(master_group_index_, k);
}

/// Gives this slice's master params as one flat tensor: a checkpoint saves it.
inline Tensor AdamWBf16::master_slice() const { return buffer_.group(master_group_index_); }

/// Gives m for parameter k: the running average of its gradients. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamWBf16::m(int k) const {
    stop_unless_one_gpu("m(k)");
    return model_buffer_.tensor(m_group_index_, k);
}

/// Gives v for parameter k: the running average of its squared gradients. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamWBf16::v(int k) const {
    stop_unless_one_gpu("v(k)");
    return model_buffer_.tensor(v_group_index_, k);
}

/// Gives m for this slice as one flat tensor: a checkpoint saves it (as floats, exactly).
inline const Tensor& AdamWBf16::m_slice() const { return model_buffer_.group(m_group_index_); }

/// Gives v for this slice as one flat tensor: a checkpoint saves it (as floats, exactly).
inline const Tensor& AdamWBf16::v_slice() const { return model_buffer_.group(v_group_index_); }

// ── AdamWBf16: the update ────────────────────────────────────────────────────

/// Does AdamW's step t on this slice with one launch of adamw_update_bf16.
///
///   t   the step number: 1, 2, 3, ...
inline void AdamWBf16::update(int t) {
    // The master moves in float, then is rounded into this slice of the params. grad_norm is the norm of all the
    // gradients, so every GPU clips alike.
    size_t start = slice_start();
    adamw_update_bf16(all_params().data_ptr<nv_bfloat16>() + start, all_grads().data_ptr<nv_bfloat16>() + start,
                      m_slice().data_ptr<nv_bfloat16>(), v_slice().data_ptr<nv_bfloat16>(), slice_size(),
                      decayed_in_slice(), grad_norm().data(), config_.max_norm, config_.learning_rate, config_.beta1,
                      config_.beta2, config_.eps, config_.weight_decay, t, buffer_.group(master_group_index_).data());
}

#endif // MODEL_BF16_CUH
