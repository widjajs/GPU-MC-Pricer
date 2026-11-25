#include "pricer/cpu_pricer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

#include "pricer/analytic.hpp"
#include "pricer/model.hpp"
#include "pricer/stats.hpp"

namespace pricer {
namespace {

constexpr std::uint64_t kBlockSamples = 1ull << 14;  // samples per independent RNG stream

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

struct CpuNormal {
    std::mt19937_64 rng;
    std::normal_distribution<double> dist{0.0, 1.0};
    explicit CpuNormal(std::uint64_t seed) : rng(seed) {}
    double operator()() { return dist(rng); }
};

// Runs `simulate(normal)` for every sample across n_threads, returning merged moments.
template <class Simulate>
Moments run_parallel(std::uint64_t n_samples, std::uint64_t seed, unsigned n_threads,
                     const Simulate& simulate) {
    if (n_threads == 0) n_threads = std::max(1u, std::thread::hardware_concurrency());
    const std::uint64_t n_blocks = (n_samples + kBlockSamples - 1) / kBlockSamples;
    n_threads = static_cast<unsigned>(std::min<std::uint64_t>(n_threads, n_blocks));

    // Each block fills its own slot; slots are merged in order afterwards so the
    // floating-point summation order does not depend on the thread count.
    std::vector<Moments> per_block(n_blocks);
    std::atomic<std::uint64_t> next{0};
    auto worker = [&] {
        for (std::uint64_t b; (b = next.fetch_add(1)) < n_blocks;) {
            CpuNormal normal(splitmix64(seed ^ splitmix64(b)));
            const std::uint64_t count = std::min(kBlockSamples, n_samples - b * kBlockSamples);
            Moments m;
            for (std::uint64_t i = 0; i < count; ++i) {
                const Sample s = simulate(normal);
                m.add(s.y, s.x);
            }
            per_block[b] = m;
        }
    };

    std::vector<std::thread> threads;
    for (unsigned t = 1; t < n_threads; ++t) threads.emplace_back(worker);
    worker();
    for (auto& t : threads) t.join();

    Moments total;
    for (const Moments& m : per_block) total.merge(m);
    return total;
}

double elapsed_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

PriceResult price_cpu(const Market& m, const Option& o, const SimConfig& cfg, unsigned n_threads) {
    const auto start = std::chrono::steady_clock::now();
    validate(o, cfg);
    const PathParams<double> p = make_path_params<double>(m, o);
    const Moments s = run_parallel(sample_count(cfg), cfg.seed, n_threads, [&](CpuNormal& n) {
        return simulate_path(p, n, cfg.antithetic);
    });
    PriceResult r = finalize(s, std::exp(-m.rate * o.maturity), control_mean(m, o, cfg),
                             paths_simulated(cfg));
    r.seconds = elapsed_since(start);
    return r;
}

PriceResult price_basket_cpu(const BasketOption& b, const SimConfig& cfg, unsigned n_threads) {
    const auto start = std::chrono::steady_clock::now();
    const BasketParams p = make_basket_params(b, cfg);
    const Moments s = run_parallel(sample_count(cfg), cfg.seed, n_threads, [&](CpuNormal& n) {
        return simulate_basket<double>(p, n, cfg.antithetic);
    });
    PriceResult r = finalize(s, std::exp(-b.rate * b.maturity), control_mean(b, cfg),
                             paths_simulated(cfg));
    r.seconds = elapsed_since(start);
    return r;
}

}  // namespace pricer
