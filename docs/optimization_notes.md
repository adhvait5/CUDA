# Optimization Notes

Milestone 1 establishes the explicit FP32 PyTorch reference and measurement
workflow. Milestone 2 adds the intentionally direct CUDA baseline:

- one thread per score element, serial accumulation over `D`;
- one block per softmax row with only thread zero active;
- one thread per output element, serial accumulation over `N`.

Its score buffer is explicitly materialized as `[B, H, N, N]`. It uses no
shared-memory tiles, warp primitives, coalescing-specific transforms, or fused
attention algorithm. The redundant global reads and serial softmax are
intentional bottlenecks for later, isolated optimization experiments.

## Milestone 3: tiled CUDA evolution

`attention_optimized.cu` retains the same three-stage attention structure and
score buffer, changing only the implementation of those stages:

- Score/output threads previously reloaded operands from global memory. 16x16
  Q/K and probability/V shared-memory tiles reuse each loaded value across 16
  outputs, at the cost of synchronization and boundary guards. The tile is
  deliberately conservative for Pascal shared-memory and register limits.
- Neighboring score threads previously read K with a `D` stride. Cooperative
  tile loading assigns neighboring threads adjacent row-major Q/K/V elements;
  no extra transpose buffer is introduced.
- One thread previously calculated an entire softmax row. 128 or 256 threads
  now each handle a strided subset of that row. More threads can waste work on
  short rows, so neither configuration is presumed best.
- Warp shuffles reduce max/sum within each warp; shared memory carries only
  cross-warp partials. This avoids repeated shared-memory reduction barriers
  but requires warp-size-aware code and fixed 128/256-thread specializations.

Softmax is numerically stable: calculate `max(row)`, write
`exp(score - max)`, reduce their sum, then normalize. Parallel and tiled
accumulation change floating-point summation order, so correctness uses
`torch.testing.assert_close(rtol=1e-4, atol=1e-4)` against the explicit
PyTorch reference.

Benchmark both `--softmax-threads 128 256` on the GTX 1050 Ti for every target
shape before selecting a default. Record only actual latency and profiler
measurements in `benchmarks/results.csv`; this repository does not include
invented latency, bandwidth, occupancy, or speedup values.
