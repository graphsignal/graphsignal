#!/bin/bash
# Runs the Python suite on THIS machine (invoked on the GPU box by
# scripts/test-gpu.sh, or directly). Everything is installed into a venv that
# sees the system site-packages: the box's CUDA-enabled torch stays visible,
# and nothing is installed into the system Python the engines run from.
#
# Usage:
#   scripts/test-gpu-python.sh [--recreate] [pytest target args...]
#
#   --recreate   delete and recreate the venv first
#   venv path:   $GRAPHSIGNAL_GPU_VENV (default /tmp/gsrepo-venv)

set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${GRAPHSIGNAL_GPU_VENV:-/tmp/gsrepo-venv}"

if [[ "${1:-}" == "--recreate" ]]; then
  rm -rf "$VENV"
  shift
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "--> creating venv $VENV (with system site-packages)"
  python3 -m venv --system-site-packages "$VENV"
fi

# Pin everything to poetry.lock, the versions CI runs, so the venv neither
# picks up an untested release nor depends on what the system happens to have.
# A locked version the system lacks is installed into the venv, shadowing it.
constraints="$VENV/constraints.txt"
"$VENV/bin/python" - poetry.lock > "$constraints" <<'EOF'
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open(sys.argv[1], 'rb') as f:
    for package in tomllib.load(f)['package']:
        print(f"{package['name']}=={package['version']}")
EOF

echo "--> installing graphsignal + pytest (pinned to poetry.lock)"
"$VENV/bin/python" -m pip install -q -c "$constraints" -e . pytest

export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
python -m scripts.test_local "$@"
