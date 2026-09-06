#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

#include "attention.h"

namespace {

// A 16x16 block has 256 threads: enough to cover an output tile while fitting
// comfortably in Pascal's 48 KB shared-memory budget. Each tiled GEMM uses
// 2 * 16 * 16 * sizeof(float) = 2 KB of shared memory.
constexpr int kTile = 16;
constexpr int kMatrixThreads = kTile * kTile;

// Score stage: one 16x16 block produces scores for 16 query rows by 16 key
// columns in a single [batch, head] matrix. Q and K dimension tiles are loaded
// cooperatively into shared memory before use.
//
// Q loads: a warp's neighboring lanes load neighboring head dimensions of one
// query row. K loads: neighboring lanes load neighboring head dimensions of one
// key row. Both patterns are contiguous when D tiles are full. The later shared
// reads reuse each Q/K value across 16 dot products, eliminating the naive
// kernel's redundant global loads.
__global__ void compute_scores_tiled_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    float* __restrict__ scores,
    int64_t sequence,
    int64_t head_dim,
    float scale) {
  __shared__ float query_tile[kTile][kTile];
  __shared__ float key_tile[kTile][kTile];

  const int row_in_tile = threadIdx.y;
  const int col_in_tile = threadIdx.x;
  const int64_t bh = blockIdx.z;
  const int64_t query_row = static_cast<int64_t>(blockIdx.y) * kTile + row_in_tile;
  const int64_t key_col = static_cast<int64_t>(blockIdx.x) * kTile + col_in_tile;
  const int64_t tensor_base = bh * sequence * head_dim;

  float dot_product = 0.0f;
  for (int64_t dim_base = 0; dim_base < head_dim; dim_base += kTile) {
    const int64_t query_dim = dim_base + col_in_tile;
    const int64_t key_dim = dim_base + col_in_tile;

    // A full row of each shared tile is loaded by adjacent x threads, which
    // gives coalesced global loads for row-major [N, D] Q and K tensors.
    query_tile[row_in_tile][col_in_tile] =
        (query_row < sequence && query_dim < head_dim)
            ? query[tensor_base + query_row * head_dim + query_dim]
            : 0.0f;
    const int64_t key_row_for_load = static_cast<int64_t>(blockIdx.x) * kTile + row_in_tile;
    key_tile[row_in_tile][col_in_tile] =
        (key_row_for_load < sequence && key_dim < head_dim)
            ? key[tensor_base + key_row_for_load * head_dim + key_dim]
            : 0.0f;
    __syncthreads();

    if (query_row < sequence && key_col < sequence) {
      #pragma unroll
      for (int dim = 0; dim < kTile; ++dim) {
        dot_product += query_tile[row_in_tile][dim] * key_tile[col_in_tile][dim];
      }
    }
    __syncthreads();
  }

  if (query_row < sequence && key_col < sequence) {
    scores[(bh * sequence + query_row) * sequence + key_col] = dot_product * scale;
  }
}

__device__ __forceinline__ float warp_max(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// Softmax stage: one block owns a complete [bh, query_row] score row. Every
// thread processes strided columns, so adjacent threads load/store adjacent
// scores. Warp shuffles finish reductions without shared-memory traffic; only
// one float per warp is shared for the cross-warp reduction.
//
// Both 128 and 256 thread specializations are exposed for measurement on the
// GTX 1050 Ti. More threads lower per-thread row work for N=512, but can reduce
// occupancy or waste lanes for small N; no configuration is assumed optimal.
template <int kSoftmaxThreads>
__global__ void softmax_rows_parallel_kernel(float* scores, int64_t sequence) {
  __shared__ float warp_partials[kSoftmaxThreads / 32];

  const int thread = threadIdx.x;
  const int lane = thread % 32;
  const int warp = thread / 32;
  const int64_t row = blockIdx.x;
  float* row_scores = scores + row * sequence;

  float local_max = -INFINITY;
  for (int64_t col = thread; col < sequence; col += kSoftmaxThreads) {
    local_max = fmaxf(local_max, row_scores[col]);
  }

  local_max = warp_max(local_max);
  if (lane == 0) {
    warp_partials[warp] = local_max;
  }
  __syncthreads();

  float row_max = (thread < kSoftmaxThreads / 32) ? warp_partials[lane] : -INFINITY;
  if (warp == 0) {
    row_max = warp_max(row_max);
  }
  if (thread == 0) {
    warp_partials[0] = row_max;
  }
  __syncthreads();
  row_max = warp_partials[0];

  float local_sum = 0.0f;
  for (int64_t col = thread; col < sequence; col += kSoftmaxThreads) {
    const float exp_score = expf(row_scores[col] - row_max);
    row_scores[col] = exp_score;
    local_sum += exp_score;
  }

  local_sum = warp_sum(local_sum);
  if (lane == 0) {
    warp_partials[warp] = local_sum;
  }
  __syncthreads();

  float row_sum = (thread < kSoftmaxThreads / 32) ? warp_partials[lane] : 0.0f;
  if (warp == 0) {
    row_sum = warp_sum(row_sum);
  }
  if (thread == 0) {
    warp_partials[0] = row_sum;
  }
  __syncthreads();
  row_sum = warp_partials[0];

  for (int64_t col = thread; col < sequence; col += kSoftmaxThreads) {
    row_scores[col] /= row_sum;
  }
}

// Output stage: 16x16 blocks compute O = probabilities @ V. Probabilities are
// loaded as contiguous row segments; V is loaded as contiguous D segments.
// Shared-memory reuse lets each loaded probability/value participate in 16
// multiply-adds before the next global-memory tile is fetched.
__global__ void compute_output_tiled_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ value,
    float* __restrict__ output,
    int64_t sequence,
    int64_t head_dim) {
  __shared__ float probability_tile[kTile][kTile];
  __shared__ float value_tile[kTile][kTile];

  const int row_in_tile = threadIdx.y;
  const int col_in_tile = threadIdx.x;
  const int64_t bh = blockIdx.z;
  const int64_t query_row = static_cast<int64_t>(blockIdx.y) * kTile + row_in_tile;
  const int64_t output_dim = static_cast<int64_t>(blockIdx.x) * kTile + col_in_tile;
  const int64_t probability_base = bh * sequence * sequence;
  const int64_t value_base = bh * sequence * head_dim;

  float weighted_sum = 0.0f;
  for (int64_t key_base = 0; key_base < sequence; key_base += kTile) {
    const int64_t key_col = key_base + col_in_tile;
    probability_tile[row_in_tile][col_in_tile] =
        (query_row < sequence && key_col < sequence)
            ? probabilities[probability_base + query_row * sequence + key_col]
            : 0.0f;

    const int64_t value_row = key_base + row_in_tile;
    value_tile[row_in_tile][col_in_tile] =
        (value_row < sequence && output_dim < head_dim)
            ? value[value_base + value_row * head_dim + output_dim]
            : 0.0f;
    __syncthreads();

    if (query_row < sequence && output_dim < head_dim) {
      #pragma unroll
      for (int key_offset = 0; key_offset < kTile; ++key_offset) {
        weighted_sum += probability_tile[row_in_tile][key_offset] *
                        value_tile[key_offset][col_in_tile];
      }
    }
    __syncthreads();
  }

  if (query_row < sequence && output_dim < head_dim) {
    output[(bh * sequence + query_row) * head_dim + output_dim] = weighted_sum;
  }
}

dim3 matrix_grid(int64_t columns, int64_t rows, int64_t batch_heads) {
  return dim3(
      static_cast<unsigned int>((columns + kTile - 1) / kTile),
      static_cast<unsigned int>((rows + kTile - 1) / kTile),
      static_cast<unsigned int>(batch_heads));
}

}  // namespace

torch::Tensor attention_optimized_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t softmax_threads) {
  c10::cuda::CUDAGuard device_guard(query.device());

  const int64_t batch_heads = query.size(0) * query.size(1);
  const int64_t sequence = query.size(2);
  const int64_t head_dim = query.size(3);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto scores = torch::empty(
      {query.size(0), query.size(1), sequence, sequence}, query.options());
  auto output = torch::empty_like(query);

  compute_scores_tiled_kernel<<<
      matrix_grid(sequence, sequence, batch_heads),
      dim3(kTile, kTile),
      0,
      stream>>>(
      query.data_ptr<float>(),
      key.data_ptr<float>(),
      scores.data_ptr<float>(),
      sequence,
      head_dim,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (softmax_threads == 128) {
    softmax_rows_parallel_kernel<128><<<batch_heads * sequence, 128, 0, stream>>>(
        scores.data_ptr<float>(), sequence);
  } else {
    softmax_rows_parallel_kernel<256><<<batch_heads * sequence, 256, 0, stream>>>(
        scores.data_ptr<float>(), sequence);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  compute_output_tiled_kernel<<<
      matrix_grid(head_dim, sequence, batch_heads),
      dim3(kTile, kTile),
      0,
      stream>>>(
      scores.data_ptr<float>(),
      value.data_ptr<float>(),
      output.data_ptr<float>(),
      sequence,
      head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
