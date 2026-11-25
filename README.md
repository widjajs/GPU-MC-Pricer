# GPU Monte Carlo Option Pricer

CUDA C++ Monte Carlo engine for pricing European, arithmetic/geometric Asian, and
correlated multi-asset basket options under geometric Brownian motion. Includes a
multithreaded C++ baseline, three generations of CUDA kernels benchmarked against
each other, FP32/FP64 paths, and variance reduction (antithetic variates and
closed-form control variates). Every backend is validated against closed-form prices.

## What's implemented

| Area | Details |
|---|---|
| Products | European call/put; discretely monitored arithmetic & geometric Asian; European on a basket of up to 16 correlated assets (arithmetic or geometric) |
| Ground truth | Black–Scholes; closed-form discrete geometric Asian; closed-form geometric basket (lognormal) |
| CPU baseline | `std::thread` work-stealing over fixed RNG blocks; bit-identical results for any thread count |
| CUDA kernels | **v0** one thread/path, payoffs to global memory, host reduction · **v1** in-kernel warp-shuffle + shared-memory block reduction · **v2** occupancy-sized grid-stride loop, register accumulation, one RNG init per thread · **basket** v2 structure with Cholesky factor in `__constant__` memory |
| RNG | cuRAND Philox4x32-10 (counter-based; independent subsequence per thread) on GPU, `mt19937_64` on CPU |
| Precision | FP64 or FP32 path arithmetic; payoff sums always accumulated in FP64 |
| Variance reduction | Antithetic variates; optimal-β control variates (geometric Asian for arithmetic Asian, geometric basket for arithmetic basket) |
| Linear algebra | Cholesky factorization of the correlation matrix, correlated normals via `L·z` |
| Tooling | Statistical test suite run against every backend × kernel × precision, CLI benchmark harness, CSV → markdown summaries, Nsight Compute/Systems profiling script, Docker-based CUDA compile check for macOS |

## Layout

```
include/pricer/types.hpp     options, configs, results
include/pricer/model.hpp     path + basket simulation, shared by CPU and GPU (__host__ __device__)
include/pricer/stats.hpp     moment accumulation, control-variate estimator, standard error
include/pricer/analytic.hpp  closed forms, Cholesky, validation
src/cpu_pricer.cpp           multithreaded CPU backend
src/gpu_pricer.cu            CUDA kernels v0/v1/v2 + basket, host launch code
src/main.cpp                 `mcprice` CLI / benchmark
tests/test_pricer.cpp        statistical correctness suite
scripts/                     gpu_setup.sh, bench.sh, profile.sh, summarize.py, check_cuda_compile.sh
```

## Build & run

```bash
cmake -S . -B build && cmake --build build -j
./build/pricer_tests
./build/mcprice --option asian --steps 252 --paths 4194304          # all backends
./build/mcprice --option asian --steps 252 --cv --antithetic         # variance reduction
./build/mcprice --basket 8 --rho 0.5 --paths 16777216 --cv           # correlated basket
./build/mcprice --backend gpu --kernel naive --float                 # one kernel/precision
./build/mcprice --help
```

Without a CUDA compiler (e.g. macOS) the CPU backend builds and GPU tests skip.
`scripts/check_cuda_compile.sh` compiles the CUDA code in NVIDIA's container and
prints per-kernel register/stack usage (`ARCH=80 scripts/check_cuda_compile.sh` for A100).

### On a GPU

| Where | Cost | Nsight Compute counters | Use for |
|---|---|---|---|
| Google Colab / Kaggle | free | maybe | correctness + timing |
| AWS EC2 `g4dn.xlarge` (T4), spot | ~$0.20–0.55/h | yes (full VM, `sudo`) | profiling |
| Lambda / GCP GPU VM | ~$0.35–1.00/h | yes | profiling |

```bash
git clone https://github.com/widjajs/GPU-MC-Pricer.git && cd GPU-MC-Pricer
bash scripts/gpu_setup.sh        # toolchain check, build, tests, ncu permission check
bash scripts/bench.sh t4         # results/t4.csv + markdown tables
NCU="sudo $(which ncu)" bash scripts/profile.sh   # Nsight reports in results/profiles/
```

Colab: Runtime → Change runtime type → T4 GPU, then in a cell:
`!git clone https://github.com/widjajs/GPU-MC-Pricer.git && cd GPU-MC-Pricer && bash scripts/gpu_setup.sh && bash scripts/bench.sh colab-t4`

## Design notes

**Why three kernel generations.** Each removes one bottleneck, so benchmarking them
side by side isolates its cost:

- **v0 → v1: memory traffic.** v0 writes two doubles per path to global memory and copies
  them back over PCIe (256 MB for 16M paths), then reduces on the CPU. v1 reduces each block
  in-kernel (`__shfl_down_sync` within warps, shared memory across warps) and writes one
  40-byte partial per block. Expect the biggest win on short-path products (European), where
  the kernel does little math per byte written.
- **v1 → v2: per-thread overhead.** v1 launches one thread per path, so every path pays for
  `curand_init` and a block reduction. v2 launches only as many blocks as can be resident
  (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SMs) and loops, so those costs are paid
  once per thread and partial sums stay in registers. (Measured: this wins for short paths
  but loses to v1 on 252-step paths; see Results.)
- **FP32 vs FP64.** Consumer and inference GPUs (T4, A10G, L4) run FP64 at 1/32–1/64 of FP32
  rate, so path-heavy products (252-step Asians) should be dramatically faster in FP32. Payoff
  sums stay in FP64 so precision loss is limited to the path itself; the tests check that FP32
  prices stay within 4 standard errors of the closed forms.
- **Basket.** The Cholesky factor lives in `__constant__` memory: all threads in a warp read the
  same `L[i][j]` at the same time, which the constant cache serves as a single broadcast.
  `ptxas` reports a small stack frame for the per-thread `z[16]` array (dynamic indexing);
  templating on the dimension would move it to registers.
- **Grid size (`--waves`).** v2 and basket grids default to one wave of resident blocks; `--waves W`
  launches W waves. Measured on the T4: 1 wave is best for short paths, and extra waves give
  ~10% on 252-step paths (see Results).

**Shared model code.** `model.hpp` is compiled for both host and device, so the CPU and GPU
backends simulate exactly the same model and differ only in RNG and parallelization. A
mismatch between backends therefore points at the parallel code, not the finance.

**Variance reduction.** Results are reported with their standard error, and the useful metric
is *time to reach a target error*, not raw paths/s. The geometric-Asian control variate tracks
the arithmetic payoff closely: measured on the CPU, 4M paths of a 252-step Asian
call go from SE 0.0039 to 0.00011 with antithetic + control variate (about 35× lower error,
equivalent to ~1200× more plain paths).

**Statistics.** Every backend reduces into the same five sums (Σy, Σy², Σx, Σx², Σxy) and calls
one `finalize()`, which applies the control variate with β = Cov(x,y)/Var(x) and computes the
standard error from the residual variance.

## Results

All prices are within statistical error of the closed forms, and every backend, kernel and
precision passes the test suite on the T4 (18/18 test groups).

**Setup.** GPU: NVIDIA Tesla T4 (Google Colab, CUDA 13.0), measured with
`scripts/bench.sh t4 gpu-all`: one untimed warm-up, then best of 5. *Total* is the wall time of
the whole call, including allocation and transfers; *kernel* is CUDA-event time. CPU: Apple M5 Pro,
18 threads, FP64, same harness. (Colab's own CPU is a 2-vCPU VM, so it is not used as a
baseline.)

### T4 vs 18-thread CPU

| Workload | M5 Pro, 18 threads | T4 best FP64 | T4 best FP32 | FP32 vs CPU |
|---|---|---|---|---|
| European, 16M paths | 12.4 ms | 9.6 ms (v2) | **1.22 ms** (v2) | **10×** |
| Arithmetic Asian, 252 steps, 4M paths | 603 ms | 549 ms (v1) | **13.7 ms** (v1) | **44×** |
| Same + antithetic + control variate | 388 ms | 359 ms (v1) | **7.9 ms** (v1) | **49×** |
| Arithmetic basket, d=8, ρ=0.5, 16M paths | 82.3 ms | 74.4 ms | **10.6 ms** | **7.7×** |

### Kernel progression (European call, 16M paths)

| Kernel | FP64 total / kernel ms | FP32 total / kernel ms |
|---|---|---|
| v0 naive | 247.8 / 17.5 | 227.3 / 1.35 |
| v1 block reduce | 25.3 / 24.2 | 6.57 / 5.46 |
| v2 grid stride | **9.63** / 9.46 | **1.22** / 1.08 |

### Findings

- **v0 → v2 is 26× end to end in FP64 and 186× in FP32.** v0's kernel is fast, but 93–99% of its
  time is copying 256 MB of payoffs to the host and reducing there.
- **v1's kernel is slower than v0's** (FP32: 5.46 vs 1.35 ms). With one path per thread, every
  path pays for a 5-value FP64 block reduction, which costs more than a 1-step path. v2 pays it
  once per thread instead of once per path.
- **FP64 throughput is the hard limit.** Nsight Compute shows every FP64 kernel at ~85% SM
  throughput and ~0% memory throughput, and all three FP64 Asian kernels take the same ~548 ms
  regardless of structure. FP32 path arithmetic is 7–41× faster depending on the product,
  consistent with the T4's 1:32 FP64:FP32 ratio, and FP32 prices stay within one standard error
  of FP64.
- **The best kernel depends on path length.** For 1-step paths, v2 wins by 5× (per-thread
  overhead dominates). For 252-step paths, v1 wins: v2 takes 16–20 ms across runs vs a steady
  13.2 ms for v1.
- **Investigating v2 on long paths.** Nsight Compute shows v2-FP32 at 75% achieved occupancy
  vs 99% for v0/v1, with 59 vs 27–46 registers per thread. Hypothesis: a one-wave grid leaves SMs
  idle when warps finish early. Sweeping the grid from 1 to 32 waves of resident blocks
  (`scripts/waves_sweep.sh`) recovered only ~10% (16.1 → 14.4 ms) and slowed the European option,
  so backfill explains only part of the gap. The next step is to compare warp-stall reasons and
  instruction mix between v1 and v2 in Nsight Compute.
- **Variance reduction compounds with GPU speed.** Antithetic + control variate reaches
  SE 0.00011 in 7.9 ms; plain FP32 reaches SE 0.0039 in 13.7 ms and would need ~1200× more paths
  to match.

### Nsight Compute summary (Asian, 252 steps, 4M paths)

Collected with `scripts/profile.sh`. Nsight locks clocks to base frequency, so durations are
longer than in the benchmark: compare them with each other, not with the tables above.

| Kernel | Duration | SM throughput | Memory throughput | Registers | Theoretical / achieved occupancy |
|---|---|---|---|---|---|
| v0 FP64 | 1.49 s | 85.2% | 0.02% | 46 | 100% / 99.4% |
| v1 FP64 | 1.49 s | 85.2% | 0.05% | 62 | 100% / 99.6% |
| v2 FP64 | 1.49 s | 85.1% | 0.00% | 78 | 75% / 60.3% |
| v0 FP32 | 32.7 ms | 80.8% | 0.74% | 27 | 100% / 98.7% |
| v1 FP32 | 34.3 ms | 77.5% | 2.14% | 46 | 100% / 98.7% |
| v2 FP32 | 42.5 ms | 70.1% | 0.02% | 59 | 100% / 74.9% |
| basket d=8 FP64 | 201 ms | 86.2% | 1.05% | 64 | 100% / 75.6% |

### CPU baseline (Apple M5 Pro, FP64)

| Workload | 1 thread | 18 threads |
|---|---|---|
| European, 16M paths | 159 ms | 12.4 ms |
| Arithmetic Asian, 252 steps, 4M paths | 8.0 s | 603 ms |
| Same + antithetic + control variate | 5.0 s | 388 ms |
| Arithmetic basket, d=8, ρ=0.5, 16M paths | 1.15 s | 82.3 ms |

## References

- NVIDIA, *CUDA C++ Programming Guide*; *CUDA C++ Best Practices Guide*; cuRAND device API docs; Nsight Compute docs
- Mark Harris, *Optimizing Parallel Reduction in CUDA*
- *GPU Gems 3*, ch. 37: Efficient Random Number Generation and Application Using CUDA
- Glasserman, *Monte Carlo Methods in Financial Engineering* (ch. 4, variance reduction)
- Kemna & Vorst (1990), *A pricing method for options based on average asset values*
