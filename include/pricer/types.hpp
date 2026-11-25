#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pricer {

// Black-Scholes world: one asset following geometric Brownian motion.
struct Market {
    double spot = 100.0;
    double rate = 0.05;  // continuously compounded risk-free rate
    double vol = 0.20;   // annualized volatility
};

enum class OptionKind {
    European,         // payoff on S(T)
    ArithmeticAsian,  // payoff on arithmetic mean of S(t_1..t_N); no closed form
    GeometricAsian,   // payoff on geometric mean of S(t_1..t_N); has a closed form
};

enum class CallPut { Call, Put };

struct Option {
    OptionKind kind = OptionKind::European;
    CallPut cp = CallPut::Call;
    double strike = 100.0;
    double maturity = 1.0;  // years
    int n_steps = 1;        // time steps == monitoring dates t_i = i*T/N
};

// European option on a basket of d correlated GBM assets, priced at maturity.
enum class BasketKind {
    Arithmetic,  // underlying = sum_i w_i S_i(T); no closed form
    Geometric,   // underlying = prod_i S_i(T)^{w_i}; lognormal, has a closed form
};

struct BasketOption {
    std::vector<double> spots, vols, weights;
    std::vector<double> corr;  // d x d correlation matrix, row-major
    double rate = 0.05;
    double strike = 100.0;
    double maturity = 1.0;
    CallPut cp = CallPut::Call;
    BasketKind kind = BasketKind::Arithmetic;

    int dim() const { return static_cast<int>(spots.size()); }

    // d identical assets with pairwise correlation rho and equal weights.
    static BasketOption equicorrelated(int d, double rho, const Market& m, double strike,
                                       double maturity, CallPut cp, BasketKind kind);
};

enum class Precision { Double, Single };

// GPU kernel generations, kept side by side so they can be benchmarked.
enum class GpuKernel {
    Naive,        // v0: one thread per path, payoffs to global memory, host reduction
    BlockReduce,  // v1: one thread per path, warp-shuffle + shared-memory block reduction
    GridStride,   // v2: occupancy-sized grid, many paths per thread, register accumulation
};

struct SimConfig {
    std::uint64_t n_paths = 1ull << 20;
    std::uint64_t seed = 42;
    bool antithetic = false;       // pair each path with its mirror (z -> -z)
    bool control_variate = false;  // arithmetic Asian/basket: geometric version as control
    Precision precision = Precision::Double;     // GPU only; CPU always uses double
    GpuKernel kernel = GpuKernel::GridStride;    // GPU only; basket always uses GridStride
    int grid_waves = 1;  // GridStride/basket: launch this many times the resident block count
};

struct PriceResult {
    double price = 0.0;
    double std_error = 0.0;      // Monte Carlo standard error of the price
    double seconds = 0.0;        // wall time of the whole pricing call
    double kernel_seconds = 0.0; // GPU kernel time only (0 for CPU)
    std::uint64_t n_paths = 0;   // paths actually simulated
};

std::string to_string(OptionKind k);
std::string to_string(GpuKernel k);

}  // namespace pricer
