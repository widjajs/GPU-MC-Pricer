#!/usr/bin/env bash
# Benchmark every backend on a fixed set of workloads; writes results/<label>.csv
# and prints a markdown summary.
#   scripts/bench.sh t4            CPU + all GPU variants
#   scripts/bench.sh t4 gpu-all    GPU variants only (skip a slow CPU, e.g. on Colab)
set -euo pipefail
cd "$(dirname "$0")/.."
label="${1:?usage: scripts/bench.sh <label> [all|gpu-all]}"
backend="${2:-all}"
mkdir -p results
out="results/${label}.csv"
echo "product,cp,steps_or_dim,paths,antithetic,cv,backend,price,std_error,reference,seconds,kernel_seconds,mpaths_per_s" > "$out"

run() { ./build/mcprice --csv --repeat 5 --backend "$backend" "$@" >> "$out"; echo "  done: $*"; }

run --option european --paths 16777216
run --option asian --steps 252 --paths 4194304
run --option asian --steps 252 --paths 4194304 --antithetic --cv
run --basket 8 --rho 0.5 --paths 16777216
run --basket 8 --rho 0.5 --paths 16777216 --cv

python3 scripts/summarize.py "$out"
