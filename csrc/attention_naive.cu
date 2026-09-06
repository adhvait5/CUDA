#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

#include "attention.h"

namespace {

constexpr int kThreadsPerBlock = 256;

// Stage 1: one thread produces exactly one score element S[bh, query_row, key_col].
//
// The grid is flattened over [B * H, N, N]. Consecutive threads write consecutive
// score columns, but every thread independently walks D. Q elements are repeatedly
// loaded by threads in a row, and K elements are accessed with a stride of D across
// neighboring threads. This is deliberately straightforward, not tiled or optimized.
__global__ void compute_scores_naive_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    float* __restrict__ scores,
    int64_t batch_heads,
    int64_t sequence,
    int64_t head_dim,
    float scale) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t num_scores = batch_heads * sequence * sequence;
  if (index >= num_scores) {
    return;
  }

  const int64_t key_col = index % sequence;
  const int64_t query_row = (index / sequence) % sequence;
  const int64_t bh = index / (sequence * sequence);
  const int64_t query_offset = (bh * sequence + query_row) * head_dim;
  const int64_t key_offset = (bh * sequence + key_col) * head_dim;

  float dot_product = 0.0f;
  for (int64_t dim = 0; dim < head_dim; ++dim) {
    dot_product += query[query_offset + dim] * key[key_offset + dim];
  }
  scores[index] = dot_product * scale;
}

// Stage 2: one CUDA block owns one [bh, query_row] softmax row, but only thread
// zero performs the row's reductions. This intentionally leaves most block threads
// idle, providing a clear baseline for the later parallel-reduction optimization.
//
// Scores and probabilities are the same buffer: after finding a stable row maximum,
// the kernel overwrites each score with exp(score - max), then normalizes in place.
__global__ void softmax_rows_naive_kernel(
    float* scores,
    int64_t batch_heads,
    int64_t sequence) {
  const int64_t row = blockIdx.x;
  if (row >= batch_heads * sequence || threadIdx.x != 0) {
    return;
  }

  float* row_scores = scores + row * sequence;
  float row_max = -INFINITY;
  for (int64_t col = 0; col < sequence; ++col) {
    row_max = fmaxf(row_max, row_scores[col]);
  }

  float row_sum = 0.0f;
  for (int64_t col = 0; col < sequence; ++col) {
    const float probability_unnormalized = expf(row_scores[col] - row_max);
    row_scores[col] = probability_unnormalized;
    row_sum += probability_unnormalized;
  }

  for (int64_t col = 0; col < sequence; ++col) {
    row_scores[col] /= row_sum;
  }
}

// Stage 3: one thread produces one output element O[bh, query_row, output_dim].
//
// This is flattened over [B * H, N, D]. A thread serially sums over N, repeatedly
// reading its probability row and the matching V vector from global memory. No shared
// memory reuse or cooperative loading is used in this baseline.
__global__ void compute_output_naive_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ value,
    float* __restrict__ output,
    int64_t batch_heads,
    int64_t sequence,
    int64_t head_dim) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t num_output_elements = batch_heads * sequence * head_dim;
  if (index >= num_output_elements) {
    return;
  }

  const int64_t output_dim = index % head_dim;
  const int64_t query_row = (index / head_dim) % sequence;
  const int64_t bh = index / (head_dim * sequence);
  const int64_t probability_offset = (bh * sequence + query_row) * sequence;

  float weighted_sum = 0.0f;
  for (int64_t key_col = 0; key_col < sequence; ++key_col) {
    const int64_t value_offset = (bh * sequence + key_col) * head_dim + output_dim;
    weighted_sum += probabilities[probability_offset + key_col] * value[value_offset];
  }
  output[index] = weighted_sum;
}

int blocks_for(int64_t elements) {
  return static_cast<int>((elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

}  // namespace

torch::Tensor attention_naive_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
  c10::cuda::CUDAGuard device_guard(query.device());

  const auto batch_heads = query.size(0) * query.size(1);
  const auto sequence = query.size(2);
  const auto head_dim = query.size(3);
  const auto options = query.options();
  auto scores = torch::empty({query.size(0), query.size(1), sequence, sequence}, options);
  auto output = torch::empty_like(query);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t score_elements = batch_heads * sequence * sequence;
  compute_scores_naive_kernel<<<blocks_for(score_elements), kThreadsPerBlock, 0, stream>>>(
      query.data_ptr<float>(),
      key.data_ptr<float>(),
      scores.data_ptr<float>(),
      batch_heads,
      sequence,
      head_dim,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  softmax_rows_naive_kernel<<<batch_heads * sequence, kThreadsPerBlock, 0, stream>>>(
      scores.data_ptr<float>(), batch_heads, sequence);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t output_elements = batch_heads * sequence * head_dim;
  compute_output_naive_kernel<<<blocks_for(output_elements), kThreadsPerBlock, 0, stream>>>(
      scores.data_ptr<float>(),
      value.data_ptr<float>(),
      output.data_ptr<float>(),
      batch_heads,
      sequence,
      head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
