#!/usr/bin/env bash
# Wrapper for cmake + the native build tool. Run from the repo root.
#
# Defaults target the assignment server (2x NVIDIA L4 = sm_89, CUDA 12.7).
# Override the architecture if you build on a different GPU:
#   CUDA_ARCH=80 ./scripts/build.sh
set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "[desrt] cmake configure  (arch=sm_${CUDA_ARCH}, type=${BUILD_TYPE}, dir=${BUILD_DIR})"
cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

echo "[desrt] cmake build      (jobs=${JOBS})"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

echo "[desrt] done — ${BUILD_DIR}/desrt"
"${BUILD_DIR}/desrt" --help || true
