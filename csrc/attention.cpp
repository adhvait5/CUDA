#include <torch/extension.h>

#include "attention.h"

namespace {

void check_attention_input(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(query.scalar_type() == torch::kFloat,
              "query must have dtype torch.float32");
  TORCH_CHECK(key.scalar_type() == torch::kFloat,
              "key must have dtype torch.float32");
  TORCH_CHECK(value.scalar_type() == torch::kFloat,
              "value must have dtype torch.float32");
  TORCH_CHECK(query.dim() == 4, "query must have shape [B, H, N, D]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [B, H, N, D]");
  TORCH_CHECK(value.dim() == 4, "value must have shape [B, H, N, D]");
  TORCH_CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes(),
              "query, key, and value must have identical shapes");
  TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
              "query, key, and value must be on the same CUDA device");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "query, key, and value must be contiguous");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 &&
                  query.size(3) > 0,
              "B, H, N, and D must all be positive");
}

torch::Tensor attention_naive(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
  check_attention_input(query, key, value);
  return attention_naive_cuda(query, key, value);
}

torch::Tensor attention_optimized(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t softmax_threads) {
  check_attention_input(query, key, value);
  TORCH_CHECK(
      softmax_threads == 128 || softmax_threads == 256,
      "softmax_threads must be 128 or 256 for the optimized CUDA kernel");
  return attention_optimized_cuda(query, key, value, softmax_threads);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_forward", &attention_naive,
        "Naive FP32 scaled dot-product self-attention (CUDA)");
  m.def(
      "optimized_forward",
      &attention_optimized,
      pybind11::arg("query"),
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("softmax_threads") = 128,
      "Shared-memory-tiled FP32 scaled dot-product self-attention (CUDA)");
}
