#pragma once

#include <cmath>
#include <cstdint>
#include <optional>

#include "pricer/types.hpp"

namespace pricer {

// Running sums of the per-sample payoff y and control x. Every backend reduces
// into this, then calls finalize(), so the statistics are identical everywhere.
struct Moments {
    double sy = 0, syy = 0, sx = 0, sxx = 0, sxy = 0;
    std::uint64_t n = 0;

    void add(double y, double x) {
        sy += y;
        syy += y * y;
        sx += x;
        sxx += x * x;
        sxy += x * y;
        ++n;
    }
    void merge(const Moments& o) {
        sy += o.sy;
        syy += o.syy;
        sx += o.sx;
        sxx += o.sxx;
        sxy += o.sxy;
        n += o.n;
    }
};

// A "sample" is one path, or one antithetic pair (averaged) when antithetic is on.
inline std::uint64_t sample_count(const SimConfig& c) {
    return c.antithetic ? c.n_paths / 2 : c.n_paths;
}
inline std::uint64_t paths_simulated(const SimConfig& c) {
    return c.antithetic ? 2 * (c.n_paths / 2) : c.n_paths;
}

// Discounted price and standard error. With control_mean (= E[x], undiscounted),
// applies the optimal control variate: y - beta (x - E[x]), beta = Cov(x,y)/Var(x).
inline PriceResult finalize(const Moments& s, double discount, std::optional<double> control_mean,
                            std::uint64_t paths) {
    const double n = static_cast<double>(s.n);
    const double ybar = s.sy / n;
    const double syy = s.syy - n * ybar * ybar;  // centered sum of squares
    double estimate = ybar;
    double resid_ss = syy;
    if (control_mean) {
        const double xbar = s.sx / n;
        const double sxx = s.sxx - n * xbar * xbar;
        const double sxy = s.sxy - n * xbar * ybar;
        if (sxx > 0) {
            const double beta = sxy / sxx;
            estimate = ybar - beta * (xbar - *control_mean);
            resid_ss = syy - sxy * sxy / sxx;
        }
    }
    const double var = s.n > 1 ? resid_ss / (n - 1.0) : 0.0;

    PriceResult r;
    r.price = discount * estimate;
    r.std_error = discount * std::sqrt(var > 0 ? var / n : 0.0);
    r.n_paths = paths;
    return r;
}

}  // namespace pricer
