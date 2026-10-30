#!/bin/bash
# Run the test suites on a remote GPU machine: rsync the repo there, then run
# the Python suite and the native GPU tests.
#
# Usage:
#   scripts/test-gpu.sh [options] [pytest target args...]
#
# Options:
#   --host HOST      remote host (default: $GRAPHSIGNAL_GPU_HOST or dev-sglang-spark-01)
#   --setup          recreate the remote venv from scratch
#
# The Python suite runs in a venv on the remote ($GRAPHSIGNAL_GPU_VENV, default
# /tmp/gsrepo-venv; see scripts/test-gpu-python.sh), never the system Python.
#   --python-only    run only the Python suite
#   --native-only    run only the native tests
#
# Examples:
#   scripts/test-gpu.sh
#   scripts/test-gpu.sh --setup
#   scripts/test-gpu.sh --python-only test/recorders/test_shm_recorder.py

set -euo pipefail

HOST="${GRAPHSIGNAL_GPU_HOST:-dev-sglang-spark-01}"
REMOTE_DIR="${GRAPHSIGNAL_GPU_DIR:-/tmp/gsrepo}"
REMOTE_VENV="${GRAPHSIGNAL_GPU_VENV:-/tmp/gsrepo-venv}"
SETUP=0
RUN_PYTHON=1
RUN_NATIVE=1
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --setup) SETUP=1; shift ;;
    --python-only) RUN_NATIVE=0; shift ;;
    --native-only) RUN_PYTHON=0; shift ;;
    *) PYTEST_ARGS+=("$1"); shift ;;
  esac
done

cd "$(dirname "$0")/.."

echo "==> rsync -> $HOST:$REMOTE_DIR"
rsync -a --delete \
  --exclude .git --exclude .venv --exclude venv --exclude build --exclude dist \
  --exclude __pycache__ --exclude .pytest_cache \
  ./ "$HOST:$REMOTE_DIR/"

if [[ $RUN_PYTHON -eq 1 ]]; then
  echo "==> python suite"
  RECREATE=""
  [[ $SETUP -eq 1 ]] && RECREATE="--recreate"
  ssh "$HOST" "cd $REMOTE_DIR && GRAPHSIGNAL_GPU_VENV=$REMOTE_VENV \
    bash scripts/test-gpu-python.sh $RECREATE ${PYTEST_ARGS[*]:-}"
fi

if [[ $RUN_NATIVE -eq 1 ]]; then
  echo "==> native tests"
  ssh "$HOST" "cd $REMOTE_DIR && bash scripts/test-gpu-native.sh"
fi

echo "==> all done"
