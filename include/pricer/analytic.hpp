#pragma once

#include <optional>
#include <vector>

#include "pricer/model.hpp"
#include "pricer/types.hpp"

namespace pricer {

double norm_cdf(double x);

// Closed-form Black-Scholes price of a European call/put.
double black_scholes(const Market& m, const Option& o);

// Closed-form price of a discretely monitored geometric-average Asian option
// with monitoring dates t_i = i*T/N, i = 1..N. With N = 1 it equals Black-Scholes.
double geometric_asian(const Market& m, const Option& o);

// Closed-form price of a European option on a geometric basket prod S_i^{w_i}.
double geometric_basket(const BasketOption& b);

// The exact price if one exists (used to validate Monte Carlo).
std::optional<double> reference_price(const Market& m, const Option& o);
std::optional<double> reference_price(const BasketOption& b);

// E[control payoff], undiscounted, when cfg.control_variate applies to this
// option (arithmetic Asian / arithmetic basket); nullopt otherwise.
std::optional<double> control_mean(const Market& m, const Option& o, const SimConfig& cfg);
std::optional<double> control_mean(const BasketOption& b, const SimConfig& cfg);

// Cholesky factor L (row-major d x d, lower-triangular) with a = L L^T.
// Throws std::invalid_argument if a is not symmetric positive definite.
std::vector<double> cholesky(const std::vector<double>& a, int d);

// Validates the inputs (throws std::invalid_argument) and builds the POD params.
void validate(const Option& o, const SimConfig& cfg);
BasketParams make_basket_params(const BasketOption& b, const SimConfig& cfg);

}  // namespace pricer
