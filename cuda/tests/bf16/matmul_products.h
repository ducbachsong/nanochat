// matmul_products.h — what tune_mma_matmul_bf16 and tune_mma_tma_matmul_bf16 share: a model's 15 linear products, their random operands, and PyTorch's matmul (cuBLAS) on them.
//
//     tensor          type in memory   shape
//     left            bfloat16         {rows, sum_length}, turned {sum_length, rows}
//     right           bfloat16         {columns, sum_length}, turned {sum_length, columns}
//     bias            bfloat16         {columns}, or none
//     output          bfloat16         {rows, columns}
//     cublas_output   bfloat16         {rows, columns}: PyTorch's
#ifndef TESTS_BF16_MATMUL_PRODUCTS_H
#define TESTS_BF16_MATMUL_PRODUCTS_H

#include <torch/types.h>              // torch::randn, torch::mm_out, torch::addmm_out; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <algorithm>                  // std::find
#include <string>
#include <vector>
#include "../../kernels/cuda_common.h"   // cudaCheck, CEIL_DIV
#include "../../common/tensor.cuh"

#define FLOPS_PER_MULTIPLY_ADD 2      // a product's flops: 2 x rows x columns x sum_length
#define TERA_IN_MILLISECONDS 1e9      // flops / ms / this: TFLOPS

/// A model's linear layers: their widths, the rows a micro-batch has, and whether the 4 block linears have a bias.
struct ModelSize
{
    const char *name_ptr;
    size_t rows, embed_dim, qkv_dim, mlp_dim, vocab_size;
    bool linear_bias;                 // the head never has one
};
static const ModelSize D24_SIZE = {"nanochat d24", 16 * 2048, 1536, 3 * 1536, 4 * 1536, 32768, false};   // the trainer's micro-batch; no linear bias, as nanochat
static const ModelSize GPT2_SIZE = {"GPT-2 small", 16 * 1024, 768, 3 * 768, 4 * 768, 50304, true};      // tune_wgmma_matmul_bf16's on main

/// One product of a linear layer: output = left · rightᵀ (left or right sitting turned round), as a linear layer runs it.
struct MatmulProduct
{
    std::string name;                 // "qkv forward", "qkv input_grad", "qkv weight_grad"
    size_t rows, columns, sum_length; // output {rows, columns}, each the sum of sum_length products
    Shape left_shape, right_shape;    // as they sit in memory
    bool left_turned, right_turned;   // left {sum_length, rows}, right {sum_length, columns}
    bool add_onto;                    // weight_grad: added onto, as a backward does
    bool has_bias;                    // the forward of a layer with a bias

    /// Gives its flops.
    double flops(void) const { return (double)FLOPS_PER_MULTIPLY_ADD * rows * columns * sum_length; }

    /// Gives its squares of output for a square of block_rows_val x block_columns_val.
    size_t squares(int block_rows_val, int block_columns_val) const
    {
        return CEIL_DIV(rows, (size_t)block_rows_val) * CEIL_DIV(columns, (size_t)block_columns_val);
    }
};

/// Gives a model's 15 products: each linear layer's forward, input_grad and weight_grad.
///
///   size_val      the model's widths and rows
///   layers_list   the layers to keep ("qkv", "attention_output", "mlp_up", "mlp_down", "head"); empty: every one
static std::vector<MatmulProduct> products_of(const ModelSize &size_val, const std::vector<std::string> &layers_list)
{
    // forward:     output {rows, out} = input {rows, in} · weight {out, in}ᵀ (+ bias)
    // input_grad:  input_grad {rows, in} = output_grad {rows, out} · weight, weight sitting turned
    // weight_grad: weight_grad {out, in} += output_gradᵀ · input, both sitting turned ({rows, out} and {rows, in})
    struct Layer
    {
        const char *name_ptr;
        size_t in_features, out_features;
        bool has_bias;
    };
    const Layer all_layers_list[] = {
        {"qkv", size_val.embed_dim, size_val.qkv_dim, size_val.linear_bias},
        {"attention_output", size_val.embed_dim, size_val.embed_dim, size_val.linear_bias},
        {"mlp_up", size_val.embed_dim, size_val.mlp_dim, size_val.linear_bias},
        {"mlp_down", size_val.mlp_dim, size_val.embed_dim, size_val.linear_bias},
        {"head", size_val.embed_dim, size_val.vocab_size, false},
    };
    std::vector<MatmulProduct> products_list;
    for (const Layer &layer_val : all_layers_list)
    {
        if (!layers_list.empty() && std::find(layers_list.begin(), layers_list.end(), layer_val.name_ptr) == layers_list.end())
            continue;
        size_t rows_val = size_val.rows, in_val = layer_val.in_features, out_val = layer_val.out_features;
        std::string name_val = layer_val.name_ptr;
        products_list.push_back({name_val + " forward", rows_val, out_val, in_val, {rows_val, in_val}, {out_val, in_val}, false, false,
                                 false, layer_val.has_bias});
        products_list.push_back({name_val + " input_grad", rows_val, in_val, out_val, {rows_val, out_val}, {out_val, in_val}, false, true,
                                 false, false});
        products_list.push_back({name_val + " weight_grad", out_val, in_val, rows_val, {rows_val, out_val}, {rows_val, in_val}, true, true,
                                 true, false});
    }
    return products_list;
}

/// Gives the products the command line names: d24's by default, gpt2's with "gpt2", only the layers named if any.
///
///   argc, argv   main's
///   size_ptr     written: the model's size
static std::vector<MatmulProduct> products_from_arguments(int argc, char **argv, ModelSize *size_ptr)
{
    // ./tune_...: nanochat d24's 15 products; gpt2: GPT-2 small's; a layer's name (qkv, attention_output, mlp_up,
    // mlp_down, head): only those layers' products. Any order: ./tune_... gpt2 head.
    *size_ptr = D24_SIZE;
    std::vector<std::string> layers_list;
    for (int arg_index_val = 1; arg_index_val < argc; arg_index_val++)
    {
        std::string arg_val = argv[arg_index_val];
        if (arg_val == "gpt2")
            *size_ptr = GPT2_SIZE;
        else
            layers_list.push_back(arg_val);
    }
    return products_of(*size_ptr, layers_list);
}

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

/// Gives a PyTorch view of my bfloat16 tensor: the same memory, no copy.
///
///   tensor     bfloat16, on the GPU
///   type_val   how PyTorch reads it: kBFloat16, or kInt16 to compare bits
static torch::Tensor torch_view_of(const Tensor &tensor, torch::ScalarType type_val = torch::kBFloat16)
{
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end());
    return torch::from_blob(tensor.data_ptr(), sizes_list, torch::TensorOptions().dtype(type_val).device(torch::kCUDA));
}

/// One product's random operands and output, and PyTorch's own run of it (cuBLAS) into an output of its own.
struct MatmulOperands
{
    MatmulProduct product;
    Tensor left_tensor, right_tensor, bias_tensor, output_tensor;
    torch::Tensor left_view, right_view, bias_view;   // PyTorch's views: left as {rows, sum_length}, right as {columns, sum_length}
    torch::Tensor cublas_output;

    /// Makes the operands (seed 0), output 0s.
    ///
    ///   product_val   the product
    explicit MatmulOperands(const MatmulProduct &product_val) : product(product_val)
    {
        torch::manual_seed(0);
        left_tensor = random_bf16_tensor(product.left_shape);
        right_tensor = random_bf16_tensor(product.right_shape);
        bias_tensor = product.has_bias ? random_bf16_tensor({product.columns}) : Tensor();
        output_tensor = Tensor::zeros({product.rows, product.columns}, DType::BFloat16);
        left_view = product.left_turned ? torch_view_of(left_tensor).t() : torch_view_of(left_tensor);
        right_view = product.right_turned ? torch_view_of(right_tensor).t() : torch_view_of(right_tensor);
        if (product.has_bias)
            bias_view = torch_view_of(bias_tensor);
        cublas_output = torch::zeros({(int64_t)product.rows, (int64_t)product.columns},
                                     torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    }

    /// Gives the bias to pass a kernel: nullptr with none.
    const nv_bfloat16 *bias_ptr(void) const { return product.has_bias ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr; }

    /// Runs PyTorch's bfloat16 product into cublas_output: cuBLAS, asked the same (added onto for weight_grad).
    void run_cublas(void)
    {
        if (product.add_onto)
            cublas_output.addmm_(left_view, right_view.t());
        else if (product.has_bias)
            torch::addmm_out(cublas_output, bias_view, left_view, right_view.t());
        else
            torch::mm_out(cublas_output, left_view, right_view.t());
    }
};

#endif // TESTS_BF16_MATMUL_PRODUCTS_H
