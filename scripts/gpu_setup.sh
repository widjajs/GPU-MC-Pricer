#!/usr/bin/env bash
# Run on a machine with an NVIDIA GPU (Colab, Kaggle, AWS EC2, RunPod).
# Checks the toolchain, builds, runs the tests, and checks whether Nsight Compute
# can read hardware counters (some cloud containers block this).
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== GPU =="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || {
    echo "no NVIDIA GPU visible"; exit 1; }

echo; echo "== Toolchain =="
nvcc --version | tail -n 2 || { echo "nvcc not on PATH (try: export PATH=\$PATH:/usr/local/cuda/bin)"; exit 1; }
cmake --version | head -n 1

echo; echo "== Build =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null &&
    cmake --build build -j || exit 1

echo; echo "== Tests =="
./build/pricer_tests

echo; echo "== Nsight Compute permission check =="
if ! command -v ncu >/dev/null; then
    echo "ncu not found. Try: export PATH=\$PATH:/usr/local/cuda/bin"
elif ncu --target-processes all -k regex:mc_ -c 1 ./build/mcprice --backend gpu --paths 65536 --repeat 1 \
        2>&1 | grep -q "ERR_NVGPUCTRPERM"; then
    echo "ncu is installed but this machine blocks GPU performance counters."
    echo "Timing still works; for profiling use a full VM (e.g. AWS g4dn) and run: sudo ncu ..."
else
    echo "ncu OK (or your kernel isn't launched yet -- re-run once Milestone 1 works)."
fi
