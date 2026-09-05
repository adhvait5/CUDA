"""Run deterministic CUDA correctness checks for the PyTorch reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from python.attention_reference import scaled_dot_product_attention


SHAPES = [(sequence, head_dim) for sequence in (32, 64, 128, 256, 512) for head_dim in (32, 64, 128)]


def make_inputs(sequence: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Make repeatable FP32 self-attention inputs."""
    generator = torch.Generator(device=device).manual_seed(20260905 + sequence * 1000 + head_dim)
    shape = (1, 1, sequence, head_dim)
    return tuple(torch.randn(shape, device=device, dtype=torch.float32, generator=generator) for _ in range(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Check only N=32,D=32 and N=128,D=64.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build and verify the NVIDIA driver.")

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    shapes = [(32, 32), (128, 64)] if args.quick else SHAPES

    for sequence, head_dim in shapes:
        query, key, value = make_inputs(sequence, head_dim, device)
        first = scaled_dot_product_attention(query, key, value)
        second = scaled_dot_product_attention(query, key, value)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        if not torch.isfinite(first).all():
            raise AssertionError(f"Non-finite output for N={sequence}, D={head_dim}")
        print(f"PASS N={sequence:3d}, D={head_dim:3d}")

    print(f"Validated {len(shapes)} deterministic FP32 CUDA reference cases on {torch.cuda.get_device_name()}.")


if __name__ == "__main__":
    main()
