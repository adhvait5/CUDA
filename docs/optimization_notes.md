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

Future milestones will compare each CUDA version against this reference and
record measured data in `benchmarks/results.csv`; this repository does not
include invented latency, bandwidth, occupancy, or speedup values.
