#pragma once

#include <cstdint>

#include <torch/extension.h>

torch::Tensor attention_naive_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value);

torch::Tensor attention_optimized_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t softmax_threads);
