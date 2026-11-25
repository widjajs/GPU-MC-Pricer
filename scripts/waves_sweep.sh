#!/usr/bin/env bash
# Profiling showed the one-wave grid-stride kernel at ~75% achieved occupancy and
# slower than the one-thread-per-path kernels in FP32. Sweep the grid size to find
# where the trade-off (amortized overhead vs. scheduler backfill) lands.
# GPU only; prints kernel ms (best of 5) per configuration.
set -euo pipefail
cd "$(dirname "$0")/.."

sweep() {
    local name="$1"; shift
    echo; echo "== $name =="
    printf "%-22s %10s %10s\n" config "kernel ms" "total ms"
    for k in naive reduce; do
        ./build/mcprice --csv --repeat 5 --backend gpu --float --kernel "$k" "$@" |
            awk -F, -v k="$k" '{printf "%-22s %10.3f %10.3f\n", k, $12*1e3, $11*1e3}'
    done
    for w in 1 2 4 8 16 32; do
        ./build/mcprice --csv --repeat 5 --backend gpu --float --kernel grid --waves "$w" "$@" |
            awk -F, -v w="$w" '{printf "%-22s %10.3f %10.3f\n", "grid waves=" w, $12*1e3, $11*1e3}'
    done
}

sweep "european, 16M paths, f32" --option european --paths 16777216
sweep "asian 252 steps, 4M paths, f32" --option asian --steps 252 --paths 4194304

echo; echo "== basket d=8, 16M paths, f32 =="
printf "%-22s %10s %10s\n" config "kernel ms" "total ms"
for w in 1 2 4 8 16 32; do
    ./build/mcprice --csv --repeat 5 --backend gpu --float --waves "$w" --basket 8 --paths 16777216 |
        awk -F, -v w="$w" '{printf "%-22s %10.3f %10.3f\n", "basket waves=" w, $12*1e3, $11*1e3}'
done
