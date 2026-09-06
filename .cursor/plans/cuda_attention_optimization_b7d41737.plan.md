---
name: CUDA attention optimization
overview: Build a forward-only FP32 educational attention extension at the repository root, with an explicit PyTorch reference and five incremental CUDA versions that isolate coalescing, shared-memory tiling, parallel softmax, and warp-level reduction improvements. Every stage and both 128/256-thread softmax candidates will be validated and measured; the project deliberately materializes attention scores and avoids FlashAttention-style fusion.
todos:
  - id: scaffold-extension
    content: Create the root project scaffold, build configuration, bindings, and documented FP32 BHND API.
    status: pending
  - id: implement-kernels
    content: Implement and heavily document the five incremental CUDA attention versions.
    status: pending
  - id: validate-correctness
    content: Add deterministic validation and pytest coverage for target, boundary, and invalid-input cases.
    status: pending
  - id: benchmark-profile
    content: Add synchronized benchmark CSV generation, measured-data plotting, and profiler guidance.
    status: pending
  - id: verify-project
    content: Build and run available checks; clearly report any environment prerequisite that blocks GPU verification.
    status: pending
isProject: false
---

# CUDA Attention Kernel Optimization

## A. Proposed architecture

- Use tensor layout `[batch, heads, sequence, head_dim]`, contiguous CUDA `float32`; flatten `batch × heads` internally. Support arbitrary positive sequence/head dimensions with bounds checks, while validating and benchmarking the requested dimensions.
- Keep the implementation forward-only and non-causal: no masking, dropout, backward kernel, mixed precision, Tensor Cores, or fused/online softmax. Both CUDA paths explicitly materialize an FP32 `[B,H,N,N]` score/probability buffer.
- Build one PyTorch extension module, `cuda_attention`, exposing the five CUDA stages plus a convenient final `optimized` entry point. The Python reference will explicitly execute `Q @ K.transpose(-2,-1)`, scaling, `torch.softmax`, and `@ V` rather than calling PyTorch SDPA/FlashAttention.
- Treat the implementation as a controlled progression: V0 PyTorch; V1 naive CUDA; V2 coalesced accesses; V3 shared-memory tiled matrix products with serial softmax; V4 tiled matrix products plus block-parallel shared-memory softmax; V5 final tiled implementation with warp-shuffle softmax reductions. Reuse identical kernels between adjacent versions so each comparison changes one main factor.
- Put the requested layout directly at the currently empty repository root (avoiding an unnecessary nested project directory). The local machine currently lacks a detected CUDA toolkit, MSVC compiler, and PyTorch, so environment setup and version-compatibility checks will be documented before build instructions.

## B. File-by-file implementation plan

- [setup.py](setup.py): define `CUDAExtension`/`BuildExtension`, compile C++17 CUDA sources, and target Pascal `sm_61` with optimized release flags but no fast-math by default so numerical behavior remains understandable.
- [requirements.txt](requirements.txt): list Python test/benchmark/plot dependencies; document that a CUDA-enabled PyTorch wheel must be selected to match an installed toolkit rather than pretending one universal wheel works.
- [csrc/attention.cpp](csrc/attention.cpp): pybind bindings for `naive`, `coalesced`, `tiled`, `parallel_softmax`, and `optimized`, plus centralized tensor validation; allow 128/256 softmax threads only where relevant and dispatch on the current PyTorch CUDA stream.
- [csrc/attention_naive.cu](csrc/attention_naive.cu): host launcher plus readable baseline score, serial-row-softmax, and output kernels, with CUDA launch checks.
- [csrc/attention_optimized.cu](csrc/attention_optimized.cu): implement the V2–V5 launchers from shared kernel building blocks: a coalescing-only path, tiled score/output products, a conventional block-reduction softmax, and a warp-shuffle final softmax. Keep stage boundaries explicit and document memory access, synchronization, resource use, and edge handling.
- [python/validate.py](python/validate.py): deterministic CLI correctness sweep against the explicit PyTorch reference, reporting cases without hiding failures.
- [python/benchmark.py](python/benchmark.py): configurable warmup/repetition runner for all 15 requested `(N,D)` combinations and every V0–V5 stage; benchmark V4 and V5 with both 128 and 256 softmax threads, emit only measured rows to CSV, summarize the fastest candidate per shape, and optionally add NVTX ranges for profiler filtering.
- [python/plot_results.py](python/plot_results.py): validate the CSV and generate measured progression, speedup, and 128-vs-256 softmax-thread plots. For a single progression line/bar, select the lowest measured candidate per shape and label that post-measurement selection explicitly.
- [tests/test_attention.py](tests/test_attention.py): pytest parameterization for shapes, seeds, every CUDA version, both softmax thread counts, numerical checks, and invalid-input behavior; skip clearly when CUDA or the built extension is unavailable.
- [benchmarks/results.csv](benchmarks/results.csv): commit a schema/header only, with no synthetic rows; generated benchmark runs replace/populate it.
- [benchmarks/plots/](benchmarks/plots/): keep the generated-output directory without committing invented plots.
- [docs/optimization_notes.md](docs/optimization_notes.md): explain the V0–V5 experiment, the isolated bottleneck/transformation at each transition, 128-vs-256 tuning evidence, Pascal-specific benefits/tradeoffs, score-buffer memory cost, and Nsight Compute/Systems or legacy `nvprof` profiling workflows.
- [README.md](README.md): prerequisites, Windows build steps, API/layout, validation/benchmark commands, methodology, limitations, and an explicitly empty “results from your GPU” section populated only by the benchmark/plot workflow.

## C. CUDA kernel design

- V1 naive: one thread computes one `QKᵀ` element with a serial `D` loop, one thread computes a complete stable-softmax row, and one thread computes one output element with a serial `N` loop.
- V2 coalesced: isolate global-memory layout by explicitly transposing K into `[B,H,D,N]`, then reuse the serial arithmetic while score threads read adjacent K-transpose columns contiguously. Include transpose cost in end-to-end latency and document that this trades an extra buffer/kernel for improved access. Keep the already coalesced output mapping and serial softmax unchanged.
- V3 tiled: replace V2's transpose-plus-score and naive output with `16×16` shared-memory tiled `QKᵀ` and `P @ V` kernels; retain the exact V1 serial softmax so the measured delta primarily reflects shared-memory reuse and removal of the transpose buffer.
- V4 parallel softmax: retain V3 matrix kernels and use one block per `(batch, head, row)` with a straightforward shared-memory tree reduction for stable max and sum. Compile/launch both 128- and 256-thread forms so thread-count effects are directly measurable.
- V5 final: retain V3 matrix kernels but replace V4's reduction with warp shuffles plus a small shared cross-warp aggregate, reducing synchronization and shared-memory traffic. Expose both 128- and 256-thread forms; do not hard-code a claimed winner before GTX 1050 Ti measurements.
- Keep the score/softmax/output stages separate. This makes kernel timings and optimization effects attributable and ensures the design is not FlashAttention. Use the current CUDA stream, `C10_CUDA_KERNEL_LAUNCH_CHECK`, and PyTorch-managed temporary/output tensors.

## D. Benchmarking methodology

- Benchmark `N ∈ {32,64,128,256,512}` and `D ∈ {32,64,128}` with fixed seed and default `B=H=1` to fit and isolate kernel behavior on a 4 GB GTX 1050 Ti; expose batch/head/repetition controls via CLI.
- Allocate inputs before timing, run per-version warmups, randomize measurement order where practical, then measure repeated forward calls with CUDA events on the same stream and synchronize before reading elapsed time. Report median and useful dispersion (for example p25/p75), not a single unsynchronized wall-clock sample.
- Time V0 through V5 under identical shape/dtype/input conditions. Include each version's required temporary work (notably V2's K transpose) but exclude extension compilation, input allocation, CSV writing, and plotting.
- For V4 and V5, record separate 128- and 256-thread rows at `N=128,256,512` (and optionally all requested lengths), with a `softmax_threads` CSV column. Derive a per-shape recommendation only after measurement; retain both raw candidates so selection bias is visible. The final API accepts the selected count rather than presenting an unmeasured heuristic as optimal.
- Record environment metadata (GPU, compute capability, PyTorch/CUDA versions), measured latency, and only transparent quantities derived from measurements. If effective FLOP/s is shown, label it as model FLOPs (`~4·B·H·N²·D`) divided by measured end-to-end time—not a hardware-counter result.
- Use profiler-reported counters for occupancy and DRAM throughput/achieved bandwidth. Document `ncu --set full`/Nsight UI collection and Nsight Systems launch inspection, with version-dependent metric names and `nvprof` only as an older-toolkit fallback. Never prefill metrics or speedups.

## E. Correctness methodology

- Seed random FP32 inputs and compare every CUDA stage—and both 128/256-thread forms of V4/V5—independently against the explicit PyTorch reference using `torch.testing.assert_close()` across all requested `(N,D)` pairs and several small batch/head combinations.
- Start with `rtol=1e-4, atol=1e-4`; if Pascal accumulation order requires adjustment, use a documented, evidence-based tolerance rather than silently weakening checks.
- Exercise non-multiple tile sizes in tests in addition to the benchmark grid so partial-tile guards are verified, and include high-magnitude inputs to confirm stable max-subtracted softmax.
- Assert finite outputs and validate error paths for CPU, wrong dtype, mismatched shapes/devices, non-rank-4, and non-contiguous tensors. Run correctness before benchmarks and make the benchmark refuse or warn clearly if validation has not passed in that invocation.

## F. Expected GTX 1050 Ti challenges

- Pascal `sm_61` has no Tensor Cores or `cp.async`; optimization must come from reuse, coalescing, reductions, and sensible occupancy. A modern PyTorch/toolkit combination must still support compiling and running `sm_61`.
- Windows extension builds require a mutually compatible CUDA toolkit, MSVC Build Tools, Python, and CUDA-enabled PyTorch. The currently detected environment is missing the toolkit/compiler/PyTorch, so build verification cannot succeed until those prerequisites are installed.
- Only 48 KB shared memory per block and limited registers/SM constrain tile and block sizes; profiler results, not assumed occupancy, will determine whether 16×16/128–256-thread defaults need tuning.
- At small sequence lengths, three kernel launches and temporary allocation overhead can dominate; PyTorch may outperform educational kernels through mature libraries. At larger lengths, the materialized `N²` buffer increases memory traffic and VRAM use by design.
- Desktop GPU clock variability, Windows display/TDR activity, and profiler permission/tool availability can add noise. Warmups, repeated statistics, a quiet GPU, and explicit environment metadata are required for credible results.
- Numerical differences will arise from reduction/accumulation order; stable softmax and conservative FP32 compilation matter more than forcing bitwise equality.