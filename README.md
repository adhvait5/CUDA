# CUDA Attention Kernel Optimization

An educational CUDA performance-engineering project for scaled dot-product
self-attention on an NVIDIA GTX 1050 Ti. This milestone implements the FP32
PyTorch reference, deterministic correctness tests, and a reproducible CUDA
timing scaffold. No CUDA attention kernel has been implemented yet.

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

Run the full deterministic FP32 test suite:

```powershell
python -m pytest tests -v
```

For a lightweight CUDA-only smoke check:

```powershell
python python/validate.py --quick
```

The full validation sweep covers sequence lengths 32, 64, 128, 256, and 512
and head dimensions 32, 64, and 128.

## Benchmark the reference

Run the requested measurement grid:

```powershell
python python/benchmark.py
```

The script allocates inputs outside timed regions, warms up each case, uses
CUDA events, waits for recorded events, and synchronizes before returning
results. It overwrites `benchmarks/results.csv` with measurements from the
current GPU only. The tracked CSV contains a header and no fabricated results.

Use a short smoke benchmark while setting up:

```powershell
python python/benchmark.py --sequences 32,128 --head-dims 32,64 --warmup 5 --iterations 20
```

## Before the CUDA milestone

Verify that CUDA is available to PyTorch, every test passes on the GTX 1050
Ti, reference outputs are finite, and `benchmarks/results.csv` contains
locally measured rows. Record the PyTorch version, CUDA runtime version,
driver version, and GPU name alongside any results you share. Later CUDA
versions will be measured against these conditions rather than against
pre-filled performance claims.

## Layout

```text
csrc/                 # Reserved for later C++/CUDA extension sources
python/
  attention_reference.py
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
