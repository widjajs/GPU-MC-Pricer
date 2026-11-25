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
  once per thread and partial sums stay in registers.
- **FP32 vs FP64.** Consumer and inference GPUs (T4, A10G, L4) run FP64 at 1/32–1/64 of FP32
  rate, so path-heavy products (252-step Asians) should be dramatically faster in FP32. Payoff
  sums stay in FP64 so precision loss is limited to the path itself; the tests check that FP32
  prices stay within 4 standard errors of the closed forms.
- **Basket.** The Cholesky factor lives in `__constant__` memory: all threads in a warp read the
  same `L[i][j]` at the same time, which the constant cache serves as a single broadcast.
  `ptxas` reports a small stack frame for the per-thread `z[16]` array (dynamic indexing);
  templating on the dimension would move it to registers.

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

Measured with `scripts/bench.sh`. Prices are within statistical error of the closed forms.

**CPU baseline (Apple M5 Pro, 18 threads, double precision)**

| Workload | 1 thread | 18 threads |
|---|---|---|
| European, 16M paths | 159 ms | 12 ms |
| Arithmetic Asian, 252 steps, 4M paths | 8.0 s | 603 ms |
| Arithmetic basket, d=8, ρ=0.5, 16M paths | 1.15 s | 82 ms |

**GPU** — _run `scripts/bench.sh <gpu>` and paste the tables from `summarize.py` here._

**Profiling** — _run `scripts/profile.sh`; record per-kernel duration, registers, achieved
occupancy, and the Speed-of-Light compute vs memory throughput._

## References

- NVIDIA, *CUDA C++ Programming Guide*; *CUDA C++ Best Practices Guide*; cuRAND device API docs; Nsight Compute docs
- Mark Harris, *Optimizing Parallel Reduction in CUDA*
- *GPU Gems 3*, ch. 37: Efficient Random Number Generation and Application Using CUDA
- Glasserman, *Monte Carlo Methods in Financial Engineering* (ch. 4, variance reduction)
- Kemna & Vorst (1990), *A pricing method for options based on average asset values*
