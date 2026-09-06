#pragma once

#include <torch/extension.h>

torch::Tensor attention_naive_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value);
