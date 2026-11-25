// Correctness tests. Every backend, kernel generation, and precision runs the same
// statistical suite against closed-form prices. GPU tests auto-skip without a GPU.
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pricer/analytic.hpp"
#include "pricer/cpu_pricer.hpp"
#include "pricer/gpu_pricer.hpp"

using namespace pricer;

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::printf("    FAIL %s:%d: %s\n      ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                             \
            std::printf("\n");                                                    \
            ++g_failures;                                                         \
        }                                                                         \
    } while (0)

// MC estimate should be within k standard errors of the exact price.
void check_within_se(const PriceResult& r, double exact, const std::string& what, double k = 4.0) {
    const double z = (r.price - exact) / r.std_error;
    CHECK(std::fabs(z) < k, "%s: mc=%.6f exact=%.6f se=%.6f z=%.2f", what.c_str(), r.price, exact,
          r.std_error, z);
}

// Two independent MC estimates of the same price should agree within k combined SEs.
void check_agree(const PriceResult& a, const PriceResult& b, const std::string& what, double k = 4.0) {
    const double se = std::hypot(a.std_error, b.std_error);
    CHECK(std::fabs(a.price - b.price) < k * se, "%s: %.6f vs %.6f (combined se %.6f)",
          what.c_str(), a.price, b.price, se);
}

Market market() { return Market{100.0, 0.05, 0.2}; }

Option make(OptionKind kind, CallPut cp, int steps, double strike = 100.0) {
    Option o;
    o.kind = kind;
    o.cp = cp;
    o.n_steps = steps;
    o.strike = strike;
    o.maturity = 1.0;
    return o;
}

SimConfig config(std::uint64_t paths, bool antithetic = false, bool cv = false) {
    SimConfig c;
    c.n_paths = paths;
    c.antithetic = antithetic;
    c.control_variate = cv;
    return c;
}

using Pricer = std::function<PriceResult(const Market&, const Option&, SimConfig)>;
using BasketPricer = std::function<PriceResult(const BasketOption&, SimConfig)>;

// Statistical checks shared by every single-asset backend.
void single_asset_suite(const Pricer& price) {
    const Market m = market();
    const SimConfig cfg = config(1ull << 20);

    for (CallPut cp : {CallPut::Call, CallPut::Put}) {
        for (double strike : {80.0, 100.0, 120.0}) {
            const std::string tag = std::string(cp == CallPut::Call ? "call" : "put") + " K=" +
                                    std::to_string(static_cast<int>(strike));
            Option eu = make(OptionKind::European, cp, 1, strike);
            check_within_se(price(m, eu, cfg), black_scholes(m, eu), "european " + tag);
            Option geo = make(OptionKind::GeometricAsian, cp, 12, strike);
            check_within_se(price(m, geo, cfg), geometric_asian(m, geo), "geo asian " + tag);
        }
    }

    // Multi-step European must still match Black-Scholes (tests the time stepping).
    Option eu50 = make(OptionKind::European, CallPut::Call, 50);
    check_within_se(price(m, eu50, cfg), black_scholes(m, eu50), "european 50 steps");

    // Arithmetic mean >= geometric mean, so the arithmetic Asian call is worth more.
    Option ar = make(OptionKind::ArithmeticAsian, CallPut::Call, 12);
    Option geo = make(OptionKind::GeometricAsian, CallPut::Call, 12);
    const PriceResult plain = price(m, ar, cfg);
    CHECK(plain.price > geometric_asian(m, geo), "arith asian %.6f should exceed geo %.6f",
          plain.price, geometric_asian(m, geo));

    // Path count that is not a multiple of any block/warp size.
    Option eu = make(OptionKind::European, CallPut::Call, 1);
    const PriceResult odd = price(m, eu, config(1000003));
    CHECK(odd.n_paths == 1000003, "n_paths=%llu", static_cast<unsigned long long>(odd.n_paths));
    check_within_se(odd, black_scholes(m, eu), "european odd path count");

    // Standard error should shrink like 1/sqrt(n): 4x paths -> half the error.
    const double ratio = price(m, eu, config(1ull << 20)).std_error /
                         price(m, eu, config(1ull << 18)).std_error;
    CHECK(ratio > 0.45 && ratio < 0.55, "std err ratio %.3f, expected ~0.5", ratio);

    // Antithetic: unbiased, and lower error for a monotone payoff at equal path count.
    const PriceResult anti = price(m, eu, config(1ull << 20, true));
    check_within_se(anti, black_scholes(m, eu), "european antithetic");
    const PriceResult plain_eu = price(m, eu, cfg);
    CHECK(anti.std_error < plain_eu.std_error, "antithetic se %.6f should be < plain %.6f",
          anti.std_error, plain_eu.std_error);

    // Control variate: agrees with plain MC and cuts the error by a large factor.
    const PriceResult cv = price(m, ar, config(1ull << 20, false, true));
    check_agree(cv, plain, "arith asian cv vs plain");
    CHECK(cv.std_error < 0.1 * plain.std_error, "cv se %.6f should be << plain se %.6f",
          cv.std_error, plain.std_error);
    const PriceResult both = price(m, ar, config(1ull << 20, true, true));
    check_agree(both, cv, "arith asian antithetic+cv vs cv");
}

// Statistical checks shared by every basket backend.
void basket_suite(const BasketPricer& price) {
    const Market m = market();
    const SimConfig cfg = config(1ull << 20);

    // d = 1 is a plain European option.
    for (CallPut cp : {CallPut::Call, CallPut::Put}) {
        BasketOption b1 = BasketOption::equicorrelated(1, 0.0, m, 100.0, 1.0, cp, BasketKind::Arithmetic);
        check_within_se(price(b1, cfg), *reference_price(b1), "basket d=1");
    }

    // Geometric basket has a closed form for any correlation structure.
    for (int d : {2, 4, 8, 16}) {
        for (double rho : {0.0, 0.5, 0.9}) {
            BasketOption g = BasketOption::equicorrelated(d, rho, m, 100.0, 1.0, CallPut::Call,
                                                          BasketKind::Geometric);
            check_within_se(price(g, cfg), geometric_basket(g),
                            "geo basket d=" + std::to_string(d) + " rho=" + std::to_string(rho));
        }
    }

    // Non-uniform weights / vols / correlations (exercises the full Cholesky).
    BasketOption g;
    g.spots = {90, 100, 110};
    g.vols = {0.15, 0.25, 0.35};
    g.weights = {0.5, 0.3, 0.2};
    g.corr = {1.0, 0.3, -0.2, 0.3, 1.0, 0.6, -0.2, 0.6, 1.0};
    g.kind = BasketKind::Geometric;
    g.cp = CallPut::Put;
    check_within_se(price(g, cfg), geometric_basket(g), "geo basket heterogeneous");

    // Arithmetic basket: geometric control variate agrees with plain MC, much lower error.
    BasketOption a = BasketOption::equicorrelated(8, 0.5, m, 100.0, 1.0, CallPut::Call,
                                                  BasketKind::Arithmetic);
    const PriceResult plain = price(a, cfg);
    const PriceResult cv = price(a, config(1ull << 20, false, true));
    check_agree(cv, plain, "arith basket cv vs plain");
    CHECK(cv.std_error < 0.2 * plain.std_error, "cv se %.6f should be << plain se %.6f",
          cv.std_error, plain.std_error);
    BasketOption ag = a;
    ag.kind = BasketKind::Geometric;
    CHECK(plain.price > geometric_basket(ag), "arith basket %.6f should exceed geo %.6f",
          plain.price, geometric_basket(ag));
}

struct Test {
    std::string name;
    std::function<void()> fn;
    bool needs_gpu = false;
};

Pricer gpu_pricer(GpuKernel k, Precision p) {
    return [k, p](const Market& m, const Option& o, SimConfig c) {
        c.kernel = k;
        c.precision = p;
        return price_gpu(m, o, c);
    };
}

BasketPricer gpu_basket_pricer(Precision p) {
    return [p](const BasketOption& b, SimConfig c) {
        c.precision = p;
        return price_basket_gpu(b, c);
    };
}

}  // namespace

int main() {
    std::vector<Test> tests = {
        {"black-scholes known value",
         [] {
             // Hull's textbook value: S=K=100, r=5%, vol=20%, T=1.
             const double c = black_scholes(market(), make(OptionKind::European, CallPut::Call, 1));
             CHECK(std::fabs(c - 10.450584) < 1e-5, "got %.6f", c);
         }},
        {"put-call parity",
         [] {
             const Market m = market();
             for (double k : {80.0, 100.0, 120.0}) {
                 const double c = black_scholes(m, make(OptionKind::European, CallPut::Call, 1, k));
                 const double p = black_scholes(m, make(OptionKind::European, CallPut::Put, 1, k));
                 const double parity = m.spot - k * std::exp(-m.rate);
                 CHECK(std::fabs(c - p - parity) < 1e-10, "K=%.0f: C-P=%.10f vs %.10f", k, c - p, parity);
             }
         }},
        {"geometric asian with 1 step == black-scholes",
         [] {
             const Market m = market();
             const double g = geometric_asian(m, make(OptionKind::GeometricAsian, CallPut::Call, 1));
             const double b = black_scholes(m, make(OptionKind::European, CallPut::Call, 1));
             CHECK(std::fabs(g - b) < 1e-12, "geo=%.12f bs=%.12f", g, b);
         }},
        {"geometric basket with d=1 == black-scholes",
         [] {
             const Market m = market();
             BasketOption b = BasketOption::equicorrelated(1, 0.0, m, 100.0, 1.0, CallPut::Call,
                                                           BasketKind::Geometric);
             const double bs = black_scholes(m, make(OptionKind::European, CallPut::Call, 1));
             CHECK(std::fabs(geometric_basket(b) - bs) < 1e-12, "%.12f vs %.12f", geometric_basket(b), bs);
         }},
        {"cholesky reconstructs the matrix and rejects non-SPD input",
         [] {
             const std::vector<double> a = {4, 2, 0.4, 2, 5, 1, 0.4, 1, 3};
             const std::vector<double> L = cholesky(a, 3);
             double max_err = 0;
             for (int i = 0; i < 3; ++i)
                 for (int j = 0; j < 3; ++j) {
                     double s = 0;
                     for (int k = 0; k < 3; ++k) s += L[i * 3 + k] * L[j * 3 + k];
                     max_err = std::fmax(max_err, std::fabs(s - a[i * 3 + j]));
                 }
             CHECK(max_err < 1e-12, "max |LL^T - A| = %g", max_err);
             bool threw = false;
             try {
                 cholesky({1, 2, 2, 1}, 2);  // eigenvalues 3, -1
             } catch (const std::invalid_argument&) {
                 threw = true;
             }
             CHECK(threw, "non-positive-definite matrix was accepted");
         }},
        {"cpu: deterministic across thread counts",
         [] {
             const Option o = make(OptionKind::ArithmeticAsian, CallPut::Call, 12);
             const double a = price_cpu(market(), o, config(200000), 1).price;
             const double b = price_cpu(market(), o, config(200000), 7).price;
             CHECK(a == b, "1 thread %.15f vs 7 threads %.15f", a, b);
         }},
        {"cpu: single-asset suite",
         [] { single_asset_suite([](auto& m, auto& o, SimConfig c) { return price_cpu(m, o, c); }); }},
        {"cpu: basket suite",
         [] { basket_suite([](auto& b, SimConfig c) { return price_basket_cpu(b, c); }); }},
    };

    for (GpuKernel k : {GpuKernel::Naive, GpuKernel::BlockReduce, GpuKernel::GridStride}) {
        for (Precision p : {Precision::Double, Precision::Single}) {
            tests.push_back({"gpu " + to_string(k) + (p == Precision::Single ? " f32" : " f64") +
                                 ": single-asset suite",
                             [k, p] { single_asset_suite(gpu_pricer(k, p)); }, true});
        }
    }
    for (Precision p : {Precision::Double, Precision::Single}) {
        tests.push_back({std::string("gpu basket ") + (p == Precision::Single ? "f32" : "f64") +
                             ": basket suite",
                         [p] { basket_suite(gpu_basket_pricer(p)); }, true});
    }
    tests.push_back({"gpu: agrees with cpu on arithmetic asian (64 steps)",
                     [] {
                         const Option o = make(OptionKind::ArithmeticAsian, CallPut::Call, 64);
                         check_agree(price_gpu(market(), o, config(1ull << 21)),
                                     price_cpu(market(), o, config(1ull << 21)), "gpu vs cpu");
                     },
                     true});

    const bool have_gpu = gpu_available();
    int skipped = 0;
    for (const Test& t : tests) {
        if (t.needs_gpu && !have_gpu) {
            std::printf("SKIP %s (no GPU)\n", t.name.c_str());
            ++skipped;
            continue;
        }
        const int before = g_failures;
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::printf("    FAIL threw: %s\n", e.what());
            ++g_failures;
        }
        std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", t.name.c_str());
    }
    std::printf("\n%d failure(s), %d skipped\n", g_failures, skipped);
    return g_failures == 0 ? 0 : 1;
}
