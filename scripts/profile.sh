#!/usr/bin/env bash
# Profile each kernel generation with Nsight Compute (needs counter access: run on
# a full VM, possibly with sudo) and capture one Nsight Systems timeline.
# Open the .ncu-rep / .nsys-rep files in the Nsight GUIs on your laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/profiles
NCU="${NCU:-ncu}"   # e.g. NCU="sudo $(which ncu)" scripts/profile.sh

for k in naive reduce grid; do
    for prec in "" "--float"; do
        name="asian252_${k}${prec:+_f32}"
        echo "== $name =="
        $NCU --set full --kernel-name regex:mc_ --launch-count 1 -f -o "results/profiles/$name" \
            ./build/mcprice --backend gpu --kernel "$k" $prec --option asian --steps 252 \
            --paths 4194304 --repeat 1 >/dev/null
    done
done

$NCU --set full --kernel-name regex:mc_basket --launch-count 1 -f -o results/profiles/basket8 \
    ./build/mcprice --backend gpu --basket 8 --paths 16777216 --repeat 1 >/dev/null

# Quick text summary of the key metrics for each report.
for rep in results/profiles/*.ncu-rep; do
    echo; echo "== $rep =="
    ncu --import "$rep" --page details --section SpeedOfLight --section Occupancy \
        --section LaunchStats 2>/dev/null | grep -E "Duration|Throughput|Registers Per|Achieved Occupancy|Theoretical Occupancy|Block Limit" || true
done

if command -v nsys >/dev/null; then
    nsys profile -f true -o results/profiles/timeline_all --stats=true \
        ./build/mcprice --option asian --steps 252 --paths 4194304 --repeat 1 >/dev/null
fi
echo; echo "reports in results/profiles/"
