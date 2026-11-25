// Compiled instead of gpu_pricer.cu when no CUDA compiler is available (e.g. macOS).
#include <stdexcept>

#include "pricer/gpu_pricer.hpp"

namespace pricer {

bool gpu_available() { return false; }

std::string gpu_device_name() { return "none (built without CUDA)"; }

PriceResult price_gpu(const Market&, const Option&, const SimConfig&) {
    throw std::runtime_error("price_gpu: this build has no CUDA backend");
}

PriceResult price_basket_gpu(const BasketOption&, const SimConfig&) {
    throw std::runtime_error("price_basket_gpu: this build has no CUDA backend");
}

}  // namespace pricer
