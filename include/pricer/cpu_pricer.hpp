#pragma once

#include "pricer/types.hpp"

namespace pricer {

// Monte Carlo pricers on the CPU (always double precision). n_threads = 0 means
// "use all hardware threads". Samples are split into fixed-size RNG blocks seeded
// from (seed, block index), so results are bit-identical for any thread count.
PriceResult price_cpu(const Market& m, const Option& o, const SimConfig& cfg,
                      unsigned n_threads = 0);
PriceResult price_basket_cpu(const BasketOption& b, const SimConfig& cfg, unsigned n_threads = 0);

}  // namespace pricer
