#include "pricer/analytic.hpp"
#include "pricer/stats.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace pricer {

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

std::string to_string(OptionKind k) {
    switch (k) {
        case OptionKind::European: return "european";
        case OptionKind::ArithmeticAsian: return "asian";
        case OptionKind::GeometricAsian: return "geo-asian";
    }
    return "?";
}

std::string to_string(GpuKernel k) {
    switch (k) {
        case GpuKernel::Naive: return "v0-naive";
        case GpuKernel::BlockReduce: return "v1-reduce";
        case GpuKernel::GridStride: return "v2-grid";
    }
    return "?";
}

namespace {

// E[e^{-rT} (X - K)^+] for ln X ~ N(mu, s^2), and the put analogue.
double lognormal_option(double mu, double s, double strike, double discount, CallPut cp) {
    const double fwd = std::exp(mu + 0.5 * s * s);  // E[X]
    const double d1 = (mu - std::log(strike) + s * s) / s;
    const double d2 = d1 - s;
    if (cp == CallPut::Call) return discount * (fwd * norm_cdf(d1) - strike * norm_cdf(d2));
    return discount * (strike * norm_cdf(-d2) - fwd * norm_cdf(-d1));
}

void validate_basket(const BasketOption& b) {
    const int d = b.dim();
    if (d < 1 || d > kMaxAssets)
        throw std::invalid_argument("basket dimension must be 1.." + std::to_string(kMaxAssets));
    if (static_cast<int>(b.vols.size()) != d || static_cast<int>(b.weights.size()) != d ||
        static_cast<int>(b.corr.size()) != d * d)
        throw std::invalid_argument("basket: spots/vols/weights must have d entries, corr d*d");
    for (int i = 0; i < d; ++i) {
        if (b.spots[i] <= 0 || b.vols[i] <= 0)
            throw std::invalid_argument("basket: spots and vols must be positive");
        if (std::fabs(b.corr[i * d + i] - 1.0) > 1e-12)
            throw std::invalid_argument("basket: correlation diagonal must be 1");
    }
    if (b.strike <= 0 || b.maturity <= 0)
        throw std::invalid_argument("basket: strike and maturity must be positive");
}

}  // namespace

double black_scholes(const Market& m, const Option& o) {
    const double T = o.maturity;
    const double mu = std::log(m.spot) + (m.rate - 0.5 * m.vol * m.vol) * T;
    const double s = m.vol * std::sqrt(T);
    return lognormal_option(mu, s, o.strike, std::exp(-m.rate * T), o.cp);
}

double geometric_asian(const Market& m, const Option& o) {
    // ln G = ln S0 + nu * mean(t_i) + sigma * mean(W_{t_i})
    //   mean(t_i)          = T (N+1) / (2N)
    //   Var(mean(W_{t_i})) = T (N+1)(2N+1) / (6 N^2)
    const double T = o.maturity;
    const double N = static_cast<double>(o.n_steps);
    const double nu = m.rate - 0.5 * m.vol * m.vol;
    const double mu = std::log(m.spot) + nu * T * (N + 1.0) / (2.0 * N);
    const double s = m.vol * std::sqrt(T * (N + 1.0) * (2.0 * N + 1.0) / (6.0 * N * N));
    return lognormal_option(mu, s, o.strike, std::exp(-m.rate * T), o.cp);
}

double geometric_basket(const BasketOption& b) {
    // ln G = sum_i w_i ln S_i(T) is normal with
    //   mean = sum_i w_i (ln S0_i + (r - sigma_i^2/2) T)
    //   var  = T sum_ij w_i w_j sigma_i sigma_j rho_ij
    validate_basket(b);
    const int d = b.dim();
    const double T = b.maturity;
    double mu = 0, var = 0;
    for (int i = 0; i < d; ++i) {
        mu += b.weights[i] * (std::log(b.spots[i]) + (b.rate - 0.5 * b.vols[i] * b.vols[i]) * T);
        for (int j = 0; j < d; ++j)
            var += b.weights[i] * b.weights[j] * b.vols[i] * b.vols[j] * b.corr[i * d + j] * T;
    }
    return lognormal_option(mu, std::sqrt(var), b.strike, std::exp(-b.rate * T), b.cp);
}

std::optional<double> reference_price(const Market& m, const Option& o) {
    switch (o.kind) {
        case OptionKind::European: return black_scholes(m, o);
        case OptionKind::GeometricAsian: return geometric_asian(m, o);
        case OptionKind::ArithmeticAsian: return std::nullopt;
    }
    return std::nullopt;
}

std::optional<double> reference_price(const BasketOption& b) {
    if (b.kind == BasketKind::Geometric) return geometric_basket(b);
    if (b.dim() == 1) {  // a one-asset arithmetic basket is a plain European
        Market m{b.spots[0], b.rate, b.vols[0]};
        Option o;
        o.cp = b.cp;
        o.strike = b.strike / b.weights[0];
        o.maturity = b.maturity;
        return b.weights[0] * black_scholes(m, o);
    }
    return std::nullopt;
}

std::optional<double> control_mean(const Market& m, const Option& o, const SimConfig& cfg) {
    if (!cfg.control_variate || o.kind != OptionKind::ArithmeticAsian) return std::nullopt;
    Option geo = o;
    geo.kind = OptionKind::GeometricAsian;
    return geometric_asian(m, geo) * std::exp(m.rate * o.maturity);
}

std::optional<double> control_mean(const BasketOption& b, const SimConfig& cfg) {
    if (!cfg.control_variate || b.kind != BasketKind::Arithmetic) return std::nullopt;
    BasketOption geo = b;
    geo.kind = BasketKind::Geometric;
    return geometric_basket(geo) * std::exp(b.rate * b.maturity);
}

std::vector<double> cholesky(const std::vector<double>& a, int d) {
    if (static_cast<int>(a.size()) != d * d) throw std::invalid_argument("cholesky: size != d*d");
    std::vector<double> L(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (std::fabs(a[i * d + j] - a[j * d + i]) > 1e-12)
                throw std::invalid_argument("cholesky: matrix is not symmetric");
            double s = a[i * d + j];
            for (int k = 0; k < j; ++k) s -= L[i * d + k] * L[j * d + k];
            if (i == j) {
                if (s <= 1e-14)
                    throw std::invalid_argument("cholesky: matrix is not positive definite");
                L[i * d + i] = std::sqrt(s);
            } else {
                L[i * d + j] = s / L[j * d + j];
            }
        }
    }
    return L;
}

void validate(const Option& o, const SimConfig& cfg) {
    if (o.n_steps < 1) throw std::invalid_argument("n_steps must be >= 1");
    if (o.strike <= 0 || o.maturity <= 0)
        throw std::invalid_argument("strike and maturity must be positive");
    if (sample_count(cfg) < 2) throw std::invalid_argument("need at least 2 samples");
}

BasketParams make_basket_params(const BasketOption& b, const SimConfig& cfg) {
    validate_basket(b);
    if (sample_count(cfg) < 2) throw std::invalid_argument("need at least 2 samples");
    const int d = b.dim();
    const std::vector<double> L = cholesky(b.corr, d);

    BasketParams p{};
    p.dim = d;
    p.kind = b.kind;
    p.cp = b.cp;
    p.strike = b.strike;
    const double T = b.maturity;
    for (int i = 0; i < d; ++i) {
        p.log_fwd[i] = std::log(b.spots[i]) + (b.rate - 0.5 * b.vols[i] * b.vols[i]) * T;
        p.vol_sqrt_t[i] = b.vols[i] * std::sqrt(T);
        p.weight[i] = b.weights[i];
        for (int j = 0; j <= i; ++j) p.chol[i * kMaxAssets + j] = L[i * d + j];
    }
    return p;
}

BasketOption BasketOption::equicorrelated(int d, double rho, const Market& m, double strike,
                                          double maturity, CallPut cp, BasketKind kind) {
    BasketOption b;
    b.spots.assign(d, m.spot);
    b.vols.assign(d, m.vol);
    b.weights.assign(d, 1.0 / d);
    b.corr.assign(static_cast<size_t>(d) * d, rho);
    for (int i = 0; i < d; ++i) b.corr[i * d + i] = 1.0;
    b.rate = m.rate;
    b.strike = strike;
    b.maturity = maturity;
    b.cp = cp;
    b.kind = kind;
    return b;
}

}  // namespace pricer
