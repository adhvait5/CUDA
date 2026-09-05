# Optimization Notes

Milestone 1 establishes the explicit FP32 PyTorch reference and measurement
workflow. CUDA kernels are intentionally not present yet.

Future milestones will compare each CUDA version against this reference and
record measured data in `benchmarks/results.csv`; this repository does not
include invented latency, bandwidth, occupancy, or speedup values.
