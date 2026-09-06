import math
import os

import pytest

# Required by cuBLAS on CUDA 10.2+ when tests enable deterministic algorithms.
# It must be set before importing torch, which initializes CUDA dependencies.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from python.attention_reference import scaled_dot_product_attention
from python.attention_cuda import cuda_attention, naive_attention, optimized_attention


TARGET_SHAPES = [
    (sequence, head_dim)
    for sequence in (32, 64, 128, 256, 512)
    for head_dim in (32, 64, 128)
]
TILED_BOUNDARY_SHAPES = [(33, 47), (65, 31)]
NAIVE_CUDA_AVAILABLE = cuda_attention is not None and hasattr(cuda_attention, "naive_forward")
OPTIMIZED_CUDA_AVAILABLE = cuda_attention is not None and hasattr(cuda_attention, "optimized_forward")


def independent_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Independent, per-row formulation used to check the reference math."""
    outputs = torch.empty_like(query)
    scale = 1.0 / math.sqrt(query.shape[-1])
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            for row in range(query.shape[2]):
                scores = (key[batch, head] * query[batch, head, row]).sum(dim=-1) * scale
                probabilities = torch.softmax(scores, dim=0)
                outputs[batch, head, row] = (probabilities[:, None] * value[batch, head]).sum(dim=0)
    return outputs


@pytest.mark.parametrize(("sequence", "head_dim"), TARGET_SHAPES + TILED_BOUNDARY_SHAPES)
def test_cuda_reference_matches_independent_formula(sequence: int, head_dim: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is required for GTX 1050 Ti reference tests")

    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cuda").manual_seed(1000 + sequence * 10 + head_dim)
    shape = (1, 1, sequence, head_dim)
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) for _ in range(3)
    )

    actual = scaled_dot_product_attention(query, key, value)
    expected = independent_attention(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(actual).all()


def test_reference_is_deterministic_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is required for GTX 1050 Ti reference tests")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260905)
    query = torch.randn((1, 2, 32, 64), device="cuda", dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    first = scaled_dot_product_attention(query, key, value)
    second = scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_reference_rejects_wrong_dtype() -> None:
    tensor = torch.randn((1, 1, 4, 8), dtype=torch.float64)
    with pytest.raises(TypeError, match="torch.float32"):
        scaled_dot_product_attention(tensor, tensor, tensor)


def test_reference_rejects_mismatched_shapes() -> None:
    query = torch.randn((1, 1, 4, 8), dtype=torch.float32)
    key = torch.randn((1, 1, 5, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="identical"):
        scaled_dot_product_attention(query, key, query)


@pytest.mark.skipif(not NAIVE_CUDA_AVAILABLE, reason="Build cuda_attention before testing the CUDA baseline")
@pytest.mark.parametrize(("sequence", "head_dim"), TARGET_SHAPES)
def test_naive_cuda_matches_pytorch_reference(sequence: int, head_dim: int) -> None:
    """The future optimization oracle: each naive CUDA output must match V0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is required for CUDA extension tests")

    generator = torch.Generator(device="cuda").manual_seed(5000 + sequence * 10 + head_dim)
    shape = (1, 1, sequence, head_dim)
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) for _ in range(3)
    )
    expected = scaled_dot_product_attention(query, key, value)
    actual = naive_attention(query, key, value)

    # Different serial accumulation orders are expected, but FP32 error should
    # remain well inside this conservative CUDA baseline tolerance.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not NAIVE_CUDA_AVAILABLE, reason="Build cuda_attention before testing the CUDA baseline")
def test_naive_cuda_rejects_noncontiguous_input() -> None:
    tensor = torch.randn((1, 1, 8, 16), device="cuda", dtype=torch.float32).transpose(-1, -2)
    with pytest.raises(RuntimeError, match="contiguous"):
        naive_attention(tensor, tensor, tensor)


@pytest.mark.skipif(
    not OPTIMIZED_CUDA_AVAILABLE,
    reason="Rebuild cuda_attention before testing the optimized CUDA kernel",
)
@pytest.mark.parametrize(("sequence", "head_dim"), TARGET_SHAPES + TILED_BOUNDARY_SHAPES)
@pytest.mark.parametrize("softmax_threads", (128, 256))
def test_optimized_cuda_matches_pytorch_reference(
    sequence: int, head_dim: int, softmax_threads: int
) -> None:
    """Tiled score/output and parallel softmax must stay within FP32 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is required for CUDA extension tests")

    generator = torch.Generator(device="cuda").manual_seed(
        9000 + sequence * 100 + head_dim + softmax_threads
    )
    shape = (1, 1, sequence, head_dim)
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) for _ in range(3)
    )
    expected = scaled_dot_product_attention(query, key, value)
    actual = optimized_attention(query, key, value, softmax_threads)

    # Tiled dot products and parallel reductions change FP32 accumulation order.
    # This tolerance is intentionally explicit and shared across all test shapes.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(
    not OPTIMIZED_CUDA_AVAILABLE,
    reason="Rebuild cuda_attention before testing the optimized CUDA kernel",
)
def test_optimized_cuda_rejects_unsupported_softmax_block_size() -> None:
    tensor = torch.randn((1, 1, 8, 16), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="softmax_threads"):
        optimized_attention(tensor, tensor, tensor, softmax_threads=64)
