#!/bin/bash
# Runs the native test suites on THIS machine (invoked on the GPU box by
# scripts/test-gpu.sh, or directly). Builds and runs the pure-C++ tests, the
# CUPTI GPU test when an NVIDIA toolchain is present, and the ROCm GPU test
# when a ROCm toolchain is present.
#
# CUPTI headers: CUDA runtime installs often ship libcupti without the dev
# headers, and a stale cupti.h may sit in /usr/include from an older package —
# so the toolkit's own header locations are tried first and otherwise the
# matching nvidia-cuda-cupti wheel is downloaded and extracted into a cache.

set -euo pipefail

cd "$(dirname "$0")/.."

export PATH="/usr/local/cuda/bin:/opt/rocm/bin:$PATH"

echo "--> pure C++ tests (probe, common)"
make -f Makefile.cupti test-probe test-common test-discovery

if command -v nvcc >/dev/null 2>&1; then
  cuda_release=$(nvcc --version | grep -o 'release [0-9]*\.[0-9]*' | awk '{print $2}')
  cuda_major=${cuda_release%%.*}
  echo "--> CUDA $cuda_release detected"

  # Locate CUPTI headers: toolkit locations first, then the wheel cache.
  cupti_include=""
  for dir in "/usr/local/cuda/extras/CUPTI/include" \
             /usr/local/cuda/targets/*/include; do
    if [[ -f "$dir/cupti.h" ]]; then
      cupti_include="$dir"
      break
    fi
  done
  if [[ -z "$cupti_include" ]]; then
    cache="/tmp/graphsignal-cupti-headers-$cuda_release"
    if [[ ! -f "$cache/nvidia/cu$cuda_major/include/cupti.h" ]]; then
      echo "--> downloading CUPTI $cuda_release headers (nvidia-cuda-cupti wheel)"
      if [[ "$cuda_major" -ge 13 ]]; then
        pkg="nvidia-cuda-cupti==$cuda_release.*"
      else
        pkg="nvidia-cuda-cupti-cu$cuda_major==$cuda_release.*"
      fi
      mkdir -p "$cache"
      python3 -m pip download "$pkg" --no-deps -d "$cache" -q
      unzip -oq "$cache"/*.whl -d "$cache"
    fi
    cupti_include=$(dirname "$(find "$cache" -name cupti.h | head -1)")
  fi
  echo "--> CUPTI headers: $cupti_include"

  # Locate the CUPTI runtime library matching the toolkit.
  cupti_lib=""
  for dir in /usr/local/cuda/targets/*/lib /usr/local/cuda/lib64 \
             "/usr/local/cuda/extras/CUPTI/lib64"; do
    if compgen -G "$dir/libcupti.so*" >/dev/null 2>&1; then
      cupti_lib="$dir"
      break
    fi
  done
  echo "--> CUPTI library dir: $cupti_lib"

  echo "--> CUPTI GPU test"
  make -f Makefile.cupti test-cupti-activity \
    CUDA_MAJOR="$cuda_major" \
    CUPTI_INCLUDE_DIR="$cupti_include" \
    CUPTI_LIB_DIR="$cupti_lib"
else
  echo "--> nvcc not found; skipping CUPTI GPU test"
fi

if command -v hipcc >/dev/null 2>&1; then
  rocm_major=$(hipcc --version | grep -o 'HIP version: [0-9]*' | awk '{print $3}')
  echo "--> ROCm GPU test (ROCm $rocm_major)"
  make -f Makefile.rocm test-rocm-activity ROCM_MAJOR="${rocm_major:-7}"
else
  echo "--> hipcc not found; skipping ROCm GPU test"
fi
