#pragma once
// Path simulation shared by the CPU and CUDA backends. Everything here compiles
// as host code and (under nvcc) as device code, so both backends simulate the
// exact same model; only the RNG and the parallelization differ.

#include <math.h>

#include "pricer/types.hpp"

#if defined(__CUDACC__)
#define PRICER_HD __host__ __device__
// For HD templates that call the RNG functor, which is device-only on the GPU
// and host-only on the CPU: silence nvcc's cross-space call check.
#define PRICER_NO_EXEC_CHECK _Pragma("nv_exec_check_disable")
#else
#define PRICER_HD
#define PRICER_NO_EXEC_CHECK
#endif

namespace pricer {

constexpr int kMaxAssets = 16;

PRICER_HD inline double m_exp(double x) { return ::exp(x); }
PRICER_HD inline float m_exp(float x) { return ::expf(x); }

template <class Real>
PRICER_HD inline Real positive_part(Real x) {
    return x > Real(0) ? x : Real(0);
}

template <class Real>
PRICER_HD inline Real payoff(CallPut cp, Real underlying, Real strike) {
    return cp == CallPut::Call ? positive_part(underlying - strike)
                               : positive_part(strike - underlying);
}

// y: the payoff being priced. x: control variate payoff (0 when unused).
struct Sample {
    double y;
    double x;
};

// ---------------------------------------------------------------------------
// Single asset: European / arithmetic Asian / geometric Asian.
// ---------------------------------------------------------------------------
template <class Real>
struct PathParams {
    Real log_s0;
    Real drift;      // (r - sigma^2/2) dt
    Real diffusion;  // sigma sqrt(dt)
    Real strike;
    Real inv_steps;
    int n_steps;
    OptionKind kind;
    CallPut cp;
};

template <class Real>
inline PathParams<Real> make_path_params(const Market& m, const Option& o) {
    const double dt = o.maturity / o.n_steps;
    PathParams<Real> p;
    p.log_s0 = static_cast<Real>(::log(m.spot));
    p.drift = static_cast<Real>((m.rate - 0.5 * m.vol * m.vol) * dt);
    p.diffusion = static_cast<Real>(m.vol * ::sqrt(dt));
    p.strike = static_cast<Real>(o.strike);
    p.inv_steps = static_cast<Real>(1.0 / o.n_steps);
    p.n_steps = o.n_steps;
    p.kind = o.kind;
    p.cp = o.cp;
    return p;
}

template <class Real>
PRICER_HD inline void path_payoffs(const PathParams<Real>& p, Real log_s_end, Real arith_sum,
                                   Real log_sum, Real& y, Real& x) {
    x = Real(0);
    y = Real(0);
    switch (p.kind) {
        case OptionKind::European:
            y = payoff(p.cp, m_exp(log_s_end), p.strike);
            break;
        case OptionKind::ArithmeticAsian:
            y = payoff(p.cp, arith_sum * p.inv_steps, p.strike);
            x = payoff(p.cp, m_exp(log_sum * p.inv_steps), p.strike);  // geometric control
            break;
        case OptionKind::GeometricAsian:
            y = payoff(p.cp, m_exp(log_sum * p.inv_steps), p.strike);
            break;
    }
}

// Simulates one sample: a path, or an antithetic pair sharing the same normals.
// `normal()` must return a standard normal of type Real.
PRICER_NO_EXEC_CHECK
template <class Real, class Normal>
PRICER_HD inline Sample simulate_path(const PathParams<Real>& p, Normal& normal, bool antithetic) {
    Real ls_a = p.log_s0, ls_b = p.log_s0;
    Real arith_a = 0, arith_b = 0, logsum_a = 0, logsum_b = 0;
    const bool asian = p.kind != OptionKind::European;

    for (int i = 0; i < p.n_steps; ++i) {
        const Real dz = p.diffusion * normal();
        ls_a += p.drift + dz;
        ls_b += p.drift - dz;
        if (asian) {
            arith_a += m_exp(ls_a);
            logsum_a += ls_a;
            if (antithetic) {
                arith_b += m_exp(ls_b);
                logsum_b += ls_b;
            }
        }
    }

    Real ya, xa;
    path_payoffs(p, ls_a, arith_a, logsum_a, ya, xa);
    if (!antithetic) return {static_cast<double>(ya), static_cast<double>(xa)};
    Real yb, xb;
    path_payoffs(p, ls_b, arith_b, logsum_b, yb, xb);
    return {0.5 * (static_cast<double>(ya) + yb), 0.5 * (static_cast<double>(xa) + xb)};
}

// ---------------------------------------------------------------------------
// Basket: d correlated assets, European payoff at maturity (exact, one step).
// Plain-old-data so it can live in CUDA __constant__ memory.
// ---------------------------------------------------------------------------
struct BasketParams {
    int dim;
    BasketKind kind;
    CallPut cp;
    double strike;
    double log_fwd[kMaxAssets];     // ln S0_i + (r - sigma_i^2/2) T
    double vol_sqrt_t[kMaxAssets];  // sigma_i sqrt(T)
    double weight[kMaxAssets];
    double chol[kMaxAssets * kMaxAssets];  // lower-triangular L, row-major, corr = L L^T
};

PRICER_NO_EXEC_CHECK
template <class Real, class Normal>
PRICER_HD inline Sample simulate_basket(const BasketParams& p, Normal& normal, bool antithetic) {
    Real z[kMaxAssets];
    for (int j = 0; j < p.dim; ++j) z[j] = normal();

    Real arith_a = 0, arith_b = 0, geo_a = 0, geo_b = 0;
    for (int i = 0; i < p.dim; ++i) {
        Real w = 0;  // correlated normal: (L z)_i
        for (int j = 0; j <= i; ++j) w += static_cast<Real>(p.chol[i * kMaxAssets + j]) * z[j];
        const Real base = static_cast<Real>(p.log_fwd[i]);
        const Real shock = static_cast<Real>(p.vol_sqrt_t[i]) * w;
        const Real wt = static_cast<Real>(p.weight[i]);
        arith_a += wt * m_exp(base + shock);
        geo_a += wt * (base + shock);
        if (antithetic) {
            arith_b += wt * m_exp(base - shock);
            geo_b += wt * (base - shock);
        }
    }

    const Real k = static_cast<Real>(p.strike);
    auto leg = [&](Real arith, Real geo, double& y, double& x) {
        const Real geo_pay = payoff(p.cp, m_exp(geo), k);
        if (p.kind == BasketKind::Arithmetic) {
            y = payoff(p.cp, arith, k);
            x = geo_pay;  // geometric basket is the control
        } else {
            y = geo_pay;
            x = 0;
        }
    };
    Sample a;
    leg(arith_a, geo_a, a.y, a.x);
    if (!antithetic) return a;
    Sample b;
    leg(arith_b, geo_b, b.y, b.x);
    return {0.5 * (a.y + b.y), 0.5 * (a.x + b.x)};
}

}  // namespace pricer
