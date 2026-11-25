// mcprice: run and benchmark the Monte Carlo pricers from the command line.
//
//   mcprice --option asian --steps 252 --paths 4194304            # every backend
//   mcprice --option asian --steps 252 --cv --antithetic          # variance reduction
//   mcprice --basket 8 --rho 0.5 --paths 16777216                 # correlated basket
//   mcprice --backend gpu --kernel naive --float --csv            # one configuration
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "pricer/analytic.hpp"
#include "pricer/cpu_pricer.hpp"
#include "pricer/gpu_pricer.hpp"

using namespace pricer;

namespace {

void usage() {
    std::puts(
        "usage: mcprice [options]\n"
        "single asset:\n"
        "  --option european|asian|geo-asian   (default european)\n"
        "  --steps N        time steps / monitoring dates (default 1; 252 for daily)\n"
        "basket (European on d correlated assets, equal weights, same spot/vol):\n"
        "  --basket D       number of assets (1..16)\n"
        "  --rho R          pairwise correlation (default 0.5)\n"
        "  --basket-kind arith|geo                (default arith)\n"
        "common:\n"
        "  --put            price a put instead of a call\n"
        "  --spot S --strike K --rate r --vol sigma --maturity T\n"
        "  --paths P        number of paths (default 2^20)\n"
        "  --seed X\n"
        "  --antithetic     antithetic variates\n"
        "  --cv             control variate (arithmetic Asian / arithmetic basket)\n"
        "backends:\n"
        "  --backend cpu1|cpu|gpu|all   (default all: CPU plus every GPU kernel/precision)\n"
        "  --threads T      CPU threads for 'cpu' (default: all)\n"
        "  --kernel naive|reduce|grid   GPU kernel for --backend gpu (default grid)\n"
        "  --float          FP32 path arithmetic for --backend gpu\n"
        "  --repeat R       run each backend R times, report the fastest (default 3)\n"
        "  --csv            machine-readable output, one line per backend");
}

struct Run {
    std::string label;
    std::function<PriceResult()> fn;
};

struct Row {
    std::string label;
    PriceResult r;
};

}  // namespace

int main(int argc, char** argv) {
    Market m;
    Option o;
    SimConfig cfg;
    unsigned threads = 0;
    std::string backend = "all";
    int repeat = 3;
    bool csv = false;
    int basket_dim = 0;
    double rho = 0.5;
    BasketKind basket_kind = BasketKind::Arithmetic;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", argv[i]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg("--option")) {
            std::string v = next();
            if (v == "european") o.kind = OptionKind::European;
            else if (v == "asian") o.kind = OptionKind::ArithmeticAsian;
            else if (v == "geo-asian") o.kind = OptionKind::GeometricAsian;
            else { std::fprintf(stderr, "unknown option kind %s\n", v.c_str()); return 2; }
        } else if (arg("--kernel")) {
            std::string v = next();
            if (v == "naive") cfg.kernel = GpuKernel::Naive;
            else if (v == "reduce") cfg.kernel = GpuKernel::BlockReduce;
            else if (v == "grid") cfg.kernel = GpuKernel::GridStride;
            else { std::fprintf(stderr, "unknown kernel %s\n", v.c_str()); return 2; }
        } else if (arg("--basket-kind")) {
            std::string v = next();
            if (v == "arith") basket_kind = BasketKind::Arithmetic;
            else if (v == "geo") basket_kind = BasketKind::Geometric;
            else { std::fprintf(stderr, "unknown basket kind %s\n", v.c_str()); return 2; }
        } else if (arg("--put")) o.cp = CallPut::Put;
        else if (arg("--spot")) m.spot = std::atof(next());
        else if (arg("--strike")) o.strike = std::atof(next());
        else if (arg("--rate")) m.rate = std::atof(next());
        else if (arg("--vol")) m.vol = std::atof(next());
        else if (arg("--maturity")) o.maturity = std::atof(next());
        else if (arg("--steps")) o.n_steps = std::atoi(next());
        else if (arg("--paths")) cfg.n_paths = std::strtoull(next(), nullptr, 10);
        else if (arg("--seed")) cfg.seed = std::strtoull(next(), nullptr, 10);
        else if (arg("--threads")) threads = static_cast<unsigned>(std::atoi(next()));
        else if (arg("--backend")) backend = next();
        else if (arg("--repeat")) repeat = std::atoi(next());
        else if (arg("--basket")) basket_dim = std::atoi(next());
        else if (arg("--rho")) rho = std::atof(next());
        else if (arg("--antithetic")) cfg.antithetic = true;
        else if (arg("--cv")) cfg.control_variate = true;
        else if (arg("--float")) cfg.precision = Precision::Single;
        else if (arg("--csv")) csv = true;
        else if (arg("--help") || arg("-h")) { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); usage(); return 2; }
    }
    if (repeat < 1) repeat = 1;
    if (backend != "cpu1" && backend != "cpu" && backend != "gpu" && backend != "all") {
        std::fprintf(stderr, "unknown backend %s\n", backend.c_str());
        return 2;
    }

    const bool is_basket = basket_dim > 0;
    std::optional<BasketOption> basket;
    if (is_basket)
        basket = BasketOption::equicorrelated(basket_dim, rho, m, o.strike, o.maturity, o.cp,
                                              basket_kind);

    // Build the list of (label, pricing call) to run.
    std::vector<Run> runs;
    auto cpu = [&](unsigned t) -> std::function<PriceResult()> {
        if (is_basket) return [&, t] { return price_basket_cpu(*basket, cfg, t); };
        return [&, t] { return price_cpu(m, o, cfg, t); };
    };
    auto gpu = [&](SimConfig c) -> std::function<PriceResult()> {
        if (is_basket) return [&, c] { return price_basket_gpu(*basket, c); };
        return [&, c] { return price_gpu(m, o, c); };
    };
    auto gpu_label = [&](const SimConfig& c) {
        std::string l = is_basket ? "gpu" : "gpu-" + to_string(c.kernel);
        return c.precision == Precision::Single ? l + "-f32" : l;
    };

    if (backend == "cpu1" || backend == "all") runs.push_back({"cpu1", cpu(1)});
    if (backend == "cpu" || backend == "all")
        runs.push_back({threads ? "cpu" + std::to_string(threads) : "cpu", cpu(threads)});
    if (backend == "gpu") runs.push_back({gpu_label(cfg), gpu(cfg)});
    if (backend == "all" && gpu_available()) {
        std::vector<SimConfig> variants;
        if (is_basket) {
            for (Precision p : {Precision::Double, Precision::Single}) {
                SimConfig c = cfg;
                c.precision = p;
                variants.push_back(c);
            }
        } else {
            for (GpuKernel k : {GpuKernel::Naive, GpuKernel::BlockReduce, GpuKernel::GridStride}) {
                SimConfig c = cfg;
                c.kernel = k;
                c.precision = Precision::Double;
                variants.push_back(c);
            }
            SimConfig c = cfg;
            c.kernel = GpuKernel::GridStride;
            c.precision = Precision::Single;
            variants.push_back(c);
        }
        for (const SimConfig& c : variants) runs.push_back({gpu_label(c), gpu(c)});
    }

    std::vector<Row> rows;
    try {
        for (const Run& run : runs) {
            PriceResult best = run.fn();
            for (int k = 1; k < repeat; ++k) {
                PriceResult r = run.fn();
                if (r.seconds < best.seconds) best = r;
            }
            rows.push_back({run.label, best});
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const std::optional<double> ref = is_basket ? reference_price(*basket) : reference_price(m, o);
    const std::string product = is_basket ? (basket_kind == BasketKind::Arithmetic ? "basket" : "geo-basket")
                                          : to_string(o.kind);
    const int steps = is_basket ? basket_dim : o.n_steps;  // CSV "steps" column holds d for baskets
    const double baseline = rows.front().r.seconds;

    if (csv) {
        // product,cp,steps_or_dim,paths,antithetic,cv,backend,price,std_error,reference,seconds,kernel_seconds,mpaths_per_s
        for (const Row& row : rows)
            std::printf("%s,%s,%d,%llu,%d,%d,%s,%.8f,%.8f,%s,%.6f,%.6f,%.2f\n", product.c_str(),
                        o.cp == CallPut::Call ? "call" : "put", steps,
                        static_cast<unsigned long long>(row.r.n_paths), cfg.antithetic,
                        cfg.control_variate, row.label.c_str(), row.r.price, row.r.std_error,
                        ref ? std::to_string(*ref).c_str() : "NA", row.r.seconds,
                        row.r.kernel_seconds, row.r.n_paths / row.r.seconds / 1e6);
        return 0;
    }

    if (is_basket)
        std::printf("%s %s  d=%d rho=%.2f S=%.2f K=%.2f r=%.3f vol=%.3f T=%.2f paths=%llu\n",
                    product.c_str(), o.cp == CallPut::Call ? "call" : "put", basket_dim, rho,
                    m.spot, o.strike, m.rate, m.vol, o.maturity,
                    static_cast<unsigned long long>(cfg.n_paths));
    else
        std::printf("%s %s  S=%.2f K=%.2f r=%.3f vol=%.3f T=%.2f steps=%d paths=%llu\n",
                    product.c_str(), o.cp == CallPut::Call ? "call" : "put", m.spot, o.strike,
                    m.rate, m.vol, o.maturity, o.n_steps,
                    static_cast<unsigned long long>(cfg.n_paths));
    if (cfg.antithetic || cfg.control_variate)
        std::printf("variance reduction:%s%s\n", cfg.antithetic ? " antithetic" : "",
                    cfg.control_variate ? " control-variate" : "");
    if (ref) std::printf("closed-form reference: %.6f\n", *ref);
    if (gpu_available()) std::printf("gpu: %s\n", gpu_device_name().c_str());

    std::printf("\n%-15s %11s %10s %7s %11s %11s %10s %8s\n", "", "price", "std err", "z",
                "total ms", "kernel ms", "Mpaths/s", "speedup");
    for (const Row& row : rows) {
        char z[16] = "      -";
        if (ref && row.r.std_error > 0)
            std::snprintf(z, sizeof z, "%7.2f", (row.r.price - *ref) / row.r.std_error);
        char kms[16] = "          -";
        if (row.r.kernel_seconds > 0) std::snprintf(kms, sizeof kms, "%11.3f", row.r.kernel_seconds * 1e3);
        std::printf("%-15s %11.6f %10.6f %s %11.3f %s %10.1f %7.1fx\n", row.label.c_str(),
                    row.r.price, row.r.std_error, z, row.r.seconds * 1e3, kms,
                    row.r.n_paths / row.r.seconds / 1e6, baseline / row.r.seconds);
    }
    if (ref) std::puts("\nz = (price - reference) / std err; |z| < 3 is expected for a correct pricer.");
    return 0;
}
