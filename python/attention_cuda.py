"""Python entry points for CUDA attention extension kernels."""

import torch

try:
    import cuda_attention
except ImportError:
    cuda_attention = None


def naive_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Run the deliberately unoptimized FP32 CUDA attention baseline.

    Build the extension with ``python setup.py build_ext --inplace`` before
    calling this function. Inputs must be contiguous CUDA tensors with layout
    ``[batch, heads, sequence, head_dim]``.
    """
    if cuda_attention is None:
        raise RuntimeError(
            "The cuda_attention extension is not built. Run "
            "`python setup.py build_ext --inplace` after installing CUDA Toolkit and MSVC."
        )
    return cuda_attention.naive_forward(query, key, value)


def optimized_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_threads: int = 128,
) -> torch.Tensor:
    """Run the shared-memory-tiled CUDA attention evolution.

    ``softmax_threads`` may be 128 or 256. Both configurations are kept
    available for benchmark evidence on the target GPU; neither is assumed to
    be universally faster.
    """
    if cuda_attention is None:
        raise RuntimeError(
            "The cuda_attention extension is not built. Run "
            "`python setup.py build_ext --inplace` after installing CUDA Toolkit and MSVC."
        )
    return cuda_attention.optimized_forward(query, key, value, softmax_threads)
