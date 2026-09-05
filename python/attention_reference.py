"""Explicit PyTorch reference for scaled dot-product self-attention.

This module intentionally uses ordinary PyTorch matrix multiplication and
softmax operations. It is the correctness oracle for later CUDA kernels, not
a performance baseline based on PyTorch's fused SDPA implementation.
"""

import math

import torch


def _validate_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    """Validate the FP32 BHND self-attention tensor contract."""
    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [batch, heads, sequence, head_dim]; "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32; got {tensor.dtype}")

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "query, key, and value must have identical [batch, heads, sequence, head_dim] shapes"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if any(size == 0 for size in query.shape):
        raise ValueError("batch, heads, sequence, and head_dim must all be positive")


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Compute ``softmax(QK^T / sqrt(head_dim))V`` for FP32 self-attention.

    Args:
        query, key, value: Contiguous or strided FP32 tensors shaped
            ``[batch, heads, sequence, head_dim]`` on one device.

    Returns:
        An FP32 tensor shaped ``[batch, heads, sequence, head_dim]``.

    The score tensor has shape ``[batch, heads, sequence, sequence]``. Keeping
    these operations explicit lets later CUDA kernels be checked stage-by-stage
    and prevents PyTorch from selecting a fused attention implementation.
    """
    _validate_inputs(query, key, value)

    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value)
