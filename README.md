# CUDA Attention Kernel Optimization

An educational CUDA performance-engineering project for scaled dot-product
self-attention on an NVIDIA GTX 1050 Ti. Milestone 2 adds an intentionally
straightforward FP32 CUDA baseline beside the PyTorch reference. It is designed
for clarity and correctness, not performance.

## Reference operation

Inputs use the layout `[batch, heads, sequence, head_dim]` and must be
`torch.float32` tensors on one device:

```text
scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)
probabilities = softmax(scores, dim=-1)
output = probabilities @ V
```

The reference in `python/attention_reference.py` deliberately uses explicit
`torch.matmul` and `torch.softmax` calls. It does not call PyTorch SDPA, so it
remains a readable correctness oracle for future CUDA kernels.

## Prerequisites

- Windows with a current NVIDIA driver for the GTX 1050 Ti
- Python 3.9 or newer
- A CUDA-enabled PyTorch installation that supports your GPU

Install the matching CUDA-enabled PyTorch wheel using the selector at
[pytorch.org](https://pytorch.org/get-started/locally/), then install test
dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check the runtime sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Validate correctness

Build the CUDA extension first. This requires the CUDA Toolkit (including
`nvcc`) and Microsoft C++ Build Tools; the toolkit must be compatible with the
CUDA-enabled PyTorch wheel:

```powershell
python setup.py build_ext --inplace
```

Then run the full suite:

```powershell
python -m pytest tests -v
```

For a lightweight PyTorch-reference CUDA smoke check:

```powershell
python python/validate.py --quick
```

The full validation sweep covers sequence lengths 32, 64, 128, 256, and 512
and head dimensions 32, 64, and 128.

## Benchmark the reference and baseline

Run the requested measurement grid:

```powershell
python python/benchmark.py
```

The script allocates inputs outside timed regions, warms up each case, uses
CUDA events, waits for recorded events, and synchronizes before returning
results. It overwrites `benchmarks/results.csv` with newly measured PyTorch
reference and naive CUDA rows from the current GPU only. The repository never
adds fabricated performance results.

Use a short smoke benchmark while setting up:

```powershell
python python/benchmark.py --sequences 32,128 --head-dims 32,64 --warmup 5 --iterations 20
```

To measure only the reference without building the extension:

```powershell
python python/benchmark.py --implementations reference
```

## Naive CUDA baseline

`csrc/attention_naive.cu` intentionally separates attention into three CUDA
kernels:

1. `compute_scores_naive_kernel`: one thread computes one `QK^T` score and
   serially loops over `head_dim`.
2. `softmax_rows_naive_kernel`: one block maps to one attention row, but only
   thread zero computes max, exponentials, sum, and normalization serially.
3. `compute_output_naive_kernel`: one thread computes one output element and
   serially loops over the key sequence.

The flattened tensor convention is `BH = batch * heads`. Scores are indexed as
`[BH, query_row, key_col]`; outputs are indexed as
`[BH, query_row, output_dim]`. The mapping is easy to inspect but has expected
bottlenecks: redundant global reads of Q/K/V, strided K reads across neighboring
score threads, a single-thread softmax row, and three kernel launches with a
materialized `N x N` score buffer. Shared-memory tiling and parallel reductions
are deliberately deferred to later milestones.

## Before the CUDA milestone

Verify that CUDA is available to PyTorch, every test passes on the GTX 1050
Ti, the extension was compiled with `TORCH_CUDA_ARCH_LIST=6.1`, CUDA outputs
match the PyTorch reference, and `benchmarks/results.csv` contains locally
measured rows. Record the PyTorch version, CUDA runtime version, driver
version, and GPU name alongside any results you share. Later CUDA versions will
be measured against these conditions rather than against pre-filled performance
claims.

## Layout

```text
csrc/                 # PyBind binding and naive C++/CUDA extension sources
python/
  attention_reference.py
  attention_cuda.py
  benchmark.py
  validate.py
tests/
  test_attention.py
benchmarks/
  results.csv          # Header only until you run the benchmark
  plots/               # Reserved for measured-data plots
docs/
  optimization_notes.md
```
