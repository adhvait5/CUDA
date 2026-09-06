"""Measure the explicit PyTorch FP32 attention reference on CUDA.

The script writes only measurements collected on the current GPU. It never
ships or synthesizes benchmark rows.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from python.attention_reference import scaled_dot_product_attention
from python.attention_cuda import cuda_attention, naive_attention, optimized_attention


DEFAULT_SEQUENCES = (32, 64, 128, 256, 512)
DEFAULT_HEAD_DIMS = (32, 64, 128)
CSV_FIELDS = (
    "implementation",
    "batch_size",
    "heads",
    "sequence_length",
    "head_dim",
    "dtype",
    "device",
    "softmax_threads",
    "warmup_iterations",
    "timed_iterations",
    "median_latency_ms",
    "p25_latency_ms",
    "p75_latency_ms",
)


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list of positive integers.")
    return values


@torch.inference_mode()
def measure_cuda_ms(callable_to_time, warmup: int, iterations: int) -> list[float]:
    """Return event-based per-call latencies with synchronization at boundaries."""
    for _ in range(warmup):
        callable_to_time()

    # Warmup work must finish before the timed region begins.
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        callable_to_time()
        end.record()
        # Event elapsed_time requires that this stream's end event is complete.
        end.synchronize()
        samples.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return samples


def make_inputs(batch_size: int, heads: int, sequence: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260905 + sequence * 1000 + head_dim)
    shape = (batch_size, heads, sequence, head_dim)
    return tuple(torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) for _ in range(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=parse_int_list, default=DEFAULT_SEQUENCES)
    parser.add_argument("--head-dims", type=parse_int_list, default=DEFAULT_HEAD_DIMS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=("reference", "naive", "optimized"),
        default=("reference", "naive", "optimized"),
        help="Implementations to measure. The default compares all implemented versions.",
    )
    parser.add_argument(
        "--softmax-threads",
        nargs="+",
        type=int,
        choices=(128, 256),
        default=(128, 256),
        help="Parallel-softmax block sizes to measure for the optimized kernel.",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. This benchmark requires a CUDA-enabled PyTorch build.")
    if min(args.batch_size, args.heads, args.warmup) <= 0 or args.iterations < 4:
        raise SystemExit("Batch size, heads, and warmup must be positive; iterations must be at least 4.")
    if "naive" in args.implementations and (
        cuda_attention is None or not hasattr(cuda_attention, "naive_forward")
    ):
        raise SystemExit(
            "The cuda_attention extension is not built. Build it with "
            "`python setup.py build_ext --inplace`, or benchmark only `--implementations reference`."
        )
    if "optimized" in args.implementations and (
        cuda_attention is None or not hasattr(cuda_attention, "optimized_forward")
    ):
        raise SystemExit(
            "The optimized extension entry point is unavailable. Rebuild with "
            "`python setup.py build_ext --inplace`, or omit `optimized`."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name()
    rows = []
    for sequence in args.sequences:
        for head_dim in args.head_dims:
            query, key, value = make_inputs(args.batch_size, args.heads, sequence, head_dim)
            implementations = {
                "reference": ("pytorch_reference", scaled_dot_product_attention, (None,)),
                "naive": ("naive_cuda", naive_attention, (None,)),
                "optimized": ("optimized_cuda", optimized_attention, args.softmax_threads),
            }
            for implementation in args.implementations:
                label, attention, thread_options = implementations[implementation]
                for softmax_threads in thread_options:
                    if softmax_threads is None:
                        timed_call = lambda: attention(query, key, value)
                    else:
                        timed_call = lambda: attention(query, key, value, softmax_threads)
                    samples = measure_cuda_ms(timed_call, args.warmup, args.iterations)
                    row = {
                        "implementation": label,
                        "batch_size": args.batch_size,
                        "heads": args.heads,
                        "sequence_length": sequence,
                        "head_dim": head_dim,
                        "dtype": "float32",
                        "device": device_name,
                        "softmax_threads": softmax_threads or "",
                        "warmup_iterations": args.warmup,
                        "timed_iterations": args.iterations,
                        "median_latency_ms": f"{statistics.median(samples):.6f}",
                        "p25_latency_ms": f"{statistics.quantiles(samples, n=4, method='inclusive')[0]:.6f}",
                        "p75_latency_ms": f"{statistics.quantiles(samples, n=4, method='inclusive')[2]:.6f}",
                    }
                    rows.append(row)
                    config = "" if softmax_threads is None else f", softmax_threads={softmax_threads}"
                    print(
                        f"{label:17} N={sequence:3d}, D={head_dim:3d}{config}: "
                        f"{row['median_latency_ms']} ms median"
                    )

    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} measured rows to {args.output}")


if __name__ == "__main__":
    main()
