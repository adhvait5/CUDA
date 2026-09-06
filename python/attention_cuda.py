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
