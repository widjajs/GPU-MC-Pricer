// CUDA backend. Three generations of the single-asset kernel are kept side by
// side so they can be benchmarked against each other:
//
//   v0 mc_naive         one thread per sample; every payoff is written to global
//                       memory and copied back; the host does the reduction.
//   v1 mc_block_reduce  one thread per sample; warp-shuffle + shared-memory
//                       reduction in the kernel, one partial result per block.
//   v2 mc_grid_stride   grid sized from the occupancy API (a few blocks per SM);
//                       each thread loops over many samples accumulating in
//                       registers, so RNG setup and reduction are amortized.
//
// mc_basket uses the v2 structure for correlated multi-asset baskets, with the
// Cholesky factor in __constant__ memory (every thread in a warp reads the same
// entry at the same time, which the constant cache broadcasts).
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pricer/analytic.hpp"
#include "pricer/gpu_pricer.hpp"
#include "pricer/model.hpp"
#include "pricer/stats.hpp"

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess)                                                              \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + \
                                     " at " __FILE__ ":" + std::to_string(__LINE__));         \
    } while (0)

namespace pricer {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
static_assert(kThreads % 32 == 0 && kThreads <= 1024, "block size must be a multiple of 32");

using Rng = curandStatePhilox4_32_10_t;  // counter-based: cheap init, independent subsequences

template <class Real>
struct DeviceNormal;
template <>
struct DeviceNormal<double> {
    Rng* state;
    __device__ double operator()() { return curand_normal_double(state); }
};
template <>
struct DeviceNormal<float> {
    Rng* state;
    __device__ float operator()() { return curand_normal(state); }
};

// Payoff sums are always accumulated in double, even when paths use float.
struct DevMoments {
    double sy, syy, sx, sxx, sxy;
};

__device__ inline void accumulate(DevMoments& m, const Sample& s) {
    m.sy += s.y;
    m.syy += s.y * s.y;
    m.sx += s.x;
    m.sxx += s.x * s.x;
    m.sxy += s.x * s.y;
}

__device__ inline double warp_sum(double v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Block-wide sum; the result is valid in thread 0. Every thread must call it.
__device__ DevMoments block_sum(const DevMoments& m) {
    __shared__ double partial[5][kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    double v[5] = {m.sy, m.syy, m.sx, m.sxx, m.sxy};
#pragma unroll
    for (int k = 0; k < 5; ++k) v[k] = warp_sum(v[k]);
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 5; ++k) partial[k][warp] = v[k];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < 5; ++k) v[k] = warp_sum(lane < kWarps ? partial[k][lane] : 0.0);
    }
    return {v[0], v[1], v[2], v[3], v[4]};
}

__device__ inline std::uint64_t global_thread_id() {
    return blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
}

// ---------------------------------------------------------------- v0: naive
template <class Real>
__global__ void mc_naive(PathParams<Real> p, bool antithetic, std::uint64_t n, std::uint64_t seed,
                         double* ys, double* xs) {
    const std::uint64_t i = global_thread_id();
    if (i >= n) return;
    Rng state;
    curand_init(seed, i, 0, &state);
    DeviceNormal<Real> normal{&state};
    const Sample s = simulate_path(p, normal, antithetic);
    ys[i] = s.y;
    xs[i] = s.x;
}

// --------------------------------------------------------- v1: block reduce
template <class Real>
__global__ void __launch_bounds__(kThreads)
    mc_block_reduce(PathParams<Real> p, bool antithetic, std::uint64_t n, std::uint64_t seed,
                    DevMoments* partials) {
    const std::uint64_t i = global_thread_id();
    DevMoments m{};
    if (i < n) {  // no early return: every thread must reach block_sum's barrier
        Rng state;
        curand_init(seed, i, 0, &state);
        DeviceNormal<Real> normal{&state};
        accumulate(m, simulate_path(p, normal, antithetic));
    }
    m = block_sum(m);
    if (threadIdx.x == 0) partials[blockIdx.x] = m;
}

// ---------------------------------------------------------- v2: grid stride
template <class Real>
__global__ void __launch_bounds__(kThreads)
    mc_grid_stride(PathParams<Real> p, bool antithetic, std::uint64_t n, std::uint64_t seed,
                   DevMoments* partials) {
    const std::uint64_t tid = global_thread_id();
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    Rng state;
    curand_init(seed, tid, 0, &state);
    DeviceNormal<Real> normal{&state};
    DevMoments m{};
    for (std::uint64_t i = tid; i < n; i += stride) accumulate(m, simulate_path(p, normal, antithetic));
    m = block_sum(m);
    if (threadIdx.x == 0) partials[blockIdx.x] = m;
}

// ------------------------------------------------------------------- basket
__constant__ BasketParams c_basket;

template <class Real>
__global__ void __launch_bounds__(kThreads)
    mc_basket(bool antithetic, std::uint64_t n, std::uint64_t seed, DevMoments* partials) {
    const std::uint64_t tid = global_thread_id();
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    Rng state;
    curand_init(seed, tid, 0, &state);
    DeviceNormal<Real> normal{&state};
    DevMoments m{};
    for (std::uint64_t i = tid; i < n; i += stride)
        accumulate(m, simulate_basket<Real>(c_basket, normal, antithetic));
    m = block_sum(m);
    if (threadIdx.x == 0) partials[blockIdx.x] = m;
}

// ------------------------------------------------------------- host helpers
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr_, std::max<size_t>(bytes, 1))); }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <class T>
    T* as() const {
        return static_cast<T*>(ptr_);
    }

   private:
    void* ptr_ = nullptr;
};

// Times GPU work between start() and stop_ms() using CUDA events.
class GpuTimer {
   public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&begin_));
        CUDA_CHECK(cudaEventCreate(&end_));
    }
    ~GpuTimer() {
        cudaEventDestroy(begin_);
        cudaEventDestroy(end_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(begin_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(end_));
        CUDA_CHECK(cudaEventSynchronize(end_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin_, end_));
        return ms;
    }

   private:
    cudaEvent_t begin_{}, end_{};
};

unsigned one_per_sample_blocks(std::uint64_t n) {
    const std::uint64_t blocks = (n + kThreads - 1) / kThreads;
    if (blocks > static_cast<std::uint64_t>(INT_MAX))
        throw std::invalid_argument("too many samples for a one-thread-per-sample kernel");
    return static_cast<unsigned>(blocks);
}

// Grid for grid-stride kernels: as many blocks as can be resident at once.
template <class Kernel>
unsigned resident_blocks(Kernel kernel, std::uint64_t n) {
    int device = 0, sms = 0, per_sm = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
    const std::uint64_t resident = static_cast<std::uint64_t>(std::max(per_sm, 1)) * sms;
    const std::uint64_t needed = (n + kThreads - 1) / kThreads;
    return static_cast<unsigned>(std::max<std::uint64_t>(1, std::min(resident, needed)));
}

Moments collect_partials(const DeviceBuffer& buf, unsigned blocks, std::uint64_t n) {
    std::vector<DevMoments> host(blocks);
    CUDA_CHECK(cudaMemcpy(host.data(), buf.as<DevMoments>(), blocks * sizeof(DevMoments),
                          cudaMemcpyDeviceToHost));
    Moments total;
    for (const DevMoments& d : host) {
        total.sy += d.sy;
        total.syy += d.syy;
        total.sx += d.sx;
        total.sxx += d.sxx;
        total.sxy += d.sxy;
    }
    total.n = n;
    return total;
}

template <class Real>
Moments run_single_asset(const PathParams<Real>& p, const SimConfig& cfg, float& kernel_ms) {
    const std::uint64_t n = sample_count(cfg);
    GpuTimer timer;

    switch (cfg.kernel) {
        case GpuKernel::Naive: {
            const unsigned blocks = one_per_sample_blocks(n);
            DeviceBuffer ys(n * sizeof(double)), xs(n * sizeof(double));
            timer.start();
            mc_naive<Real><<<blocks, kThreads>>>(p, cfg.antithetic, n, cfg.seed, ys.as<double>(),
                                                 xs.as<double>());
            CUDA_CHECK(cudaGetLastError());
            kernel_ms = timer.stop_ms();
            std::vector<double> hy(n), hx(n);
            CUDA_CHECK(cudaMemcpy(hy.data(), ys.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hx.data(), xs.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
            Moments total;
            for (std::uint64_t i = 0; i < n; ++i) total.add(hy[i], hx[i]);
            return total;
        }
        case GpuKernel::BlockReduce: {
            const unsigned blocks = one_per_sample_blocks(n);
            DeviceBuffer partials(blocks * sizeof(DevMoments));
            timer.start();
            mc_block_reduce<Real><<<blocks, kThreads>>>(p, cfg.antithetic, n, cfg.seed,
                                                        partials.as<DevMoments>());
            CUDA_CHECK(cudaGetLastError());
            kernel_ms = timer.stop_ms();
            return collect_partials(partials, blocks, n);
        }
        case GpuKernel::GridStride: {
            const unsigned blocks = resident_blocks(mc_grid_stride<Real>, n);
            DeviceBuffer partials(blocks * sizeof(DevMoments));
            timer.start();
            mc_grid_stride<Real><<<blocks, kThreads>>>(p, cfg.antithetic, n, cfg.seed,
                                                       partials.as<DevMoments>());
            CUDA_CHECK(cudaGetLastError());
            kernel_ms = timer.stop_ms();
            return collect_partials(partials, blocks, n);
        }
    }
    throw std::invalid_argument("unknown GPU kernel");
}

template <class Real>
Moments run_basket(const SimConfig& cfg, float& kernel_ms) {
    const std::uint64_t n = sample_count(cfg);
    const unsigned blocks = resident_blocks(mc_basket<Real>, n);
    DeviceBuffer partials(blocks * sizeof(DevMoments));
    GpuTimer timer;
    timer.start();
    mc_basket<Real><<<blocks, kThreads>>>(cfg.antithetic, n, cfg.seed, partials.as<DevMoments>());
    CUDA_CHECK(cudaGetLastError());
    kernel_ms = timer.stop_ms();
    return collect_partials(partials, blocks, n);
}

double elapsed_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string gpu_device_name() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "unknown";
    return prop.name;
}

PriceResult price_gpu(const Market& m, const Option& o, const SimConfig& cfg) {
    const auto start = std::chrono::steady_clock::now();
    validate(o, cfg);
    float kernel_ms = 0.f;
    const Moments s = cfg.precision == Precision::Single
                          ? run_single_asset(make_path_params<float>(m, o), cfg, kernel_ms)
                          : run_single_asset(make_path_params<double>(m, o), cfg, kernel_ms);
    PriceResult r = finalize(s, std::exp(-m.rate * o.maturity), control_mean(m, o, cfg),
                             paths_simulated(cfg));
    r.kernel_seconds = kernel_ms / 1e3;
    r.seconds = elapsed_since(start);
    return r;
}

PriceResult price_basket_gpu(const BasketOption& b, const SimConfig& cfg) {
    const auto start = std::chrono::steady_clock::now();
    const BasketParams p = make_basket_params(b, cfg);
    CUDA_CHECK(cudaMemcpyToSymbol(c_basket, &p, sizeof(p)));
    float kernel_ms = 0.f;
    const Moments s = cfg.precision == Precision::Single ? run_basket<float>(cfg, kernel_ms)
                                                         : run_basket<double>(cfg, kernel_ms);
    PriceResult r = finalize(s, std::exp(-b.rate * b.maturity), control_mean(b, cfg),
                             paths_simulated(cfg));
    r.kernel_seconds = kernel_ms / 1e3;
    r.seconds = elapsed_since(start);
    return r;
}

}  // namespace pricer
