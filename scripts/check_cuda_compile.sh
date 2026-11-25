#!/usr/bin/env bash
# Compile the CUDA backend on a Mac (no GPU needed) inside NVIDIA's CUDA container,
# and print per-kernel register / shared / stack usage from ptxas.
# Requires Docker Desktop running.
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="${ARCH:-75}"  # 75 = T4, 80 = A100, 86 = A10G/RTX 30xx, 89 = L4/RTX 40xx, 90 = H100
docker run --rm -v "$PWD":/src:ro nvidia/cuda:12.6.3-devel-ubuntu24.04 bash -c "
cd /src && nvcc -std=c++17 -O3 -lineinfo -Xcompiler -Wall,-Wextra -Iinclude \
  -arch=sm_${ARCH} --resource-usage src/gpu_pricer.cu src/analytic.cpp src/cpu_pricer.cpp \
  tests/test_pricer.cpp -o /tmp/t -lcurand 2>&1 \
  | grep -E 'warning|error|Function properties|Used' \
  | sed -E 's/.*(mc_[a-z_]+I[df]).*/\1/; s/ptxas info +: //' \
  && echo 'compiled OK for sm_${ARCH}'
" 2>&1 | grep -v -E '^(==|\$|CUDA Version|Container image|This container|By pulling|https|A copy|WARNING|   Use the| +https)'
