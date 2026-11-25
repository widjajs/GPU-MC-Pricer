#pragma once

#include <string>

#include "pricer/types.hpp"

namespace pricer {

// False when built without CUDA or when no NVIDIA device is visible.
bool gpu_available();
std::string gpu_device_name();

// Monte Carlo pricers on the GPU. cfg.kernel picks the kernel generation and
// cfg.precision the arithmetic used inside the path loop (sums are always double).
PriceResult price_gpu(const Market& m, const Option& o, const SimConfig& cfg);
PriceResult price_basket_gpu(const BasketOption& b, const SimConfig& cfg);

}  // namespace pricer
