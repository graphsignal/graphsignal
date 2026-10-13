#!/bin/bash
# Start an SGLang ROCm container beside whatever else runs on the host and
# install this repo's profiler into it, editable, for the scenarios.
#
# Usage (on the AMD host):
#   bash scripts/scenarios/rocm_container_setup.sh [image] [name]
#
#   image  default lmsysorg/sglang:v0.5.21-rocm10-mi30x  (ROCm 10, rocprofiler-sdk 1.3.5)
#          lmsysorg/sglang:v0.5.21-rocm720-mi30x        (ROCm 7.2, rocprofiler-sdk 1.1.0)
#   name   container name (default rocm10)
#
# Environment:
#   REPO_DIR   the repo checkout on the host, bind-mounted at the same path
#              (default: this script's repo)
#   HF_DIR     Hugging Face cache on the host, bind-mounted (default $REPO_DIR/../hf)
#
# The container uses host networking (give each SGLang its own PORT) and runs
# `sleep infinity`; use `docker exec -w $REPO_DIR <name> bash scripts/scenarios/sglang_scenarios.sh ...`.
set -eu
IMAGE=${1:-lmsysorg/sglang:v0.5.21-rocm10-mi30x}
NAME=${2:-rocm10}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
HF_DIR=${HF_DIR:-$(dirname "$REPO_DIR")/hf}
mkdir -p "$HF_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16g \
  -v "$REPO_DIR:$REPO_DIR" -v "$HF_DIR:$HF_DIR" -e HF_HOME="$HF_DIR" \
  "$IMAGE" sleep infinity
docker exec "$NAME" bash -c "cat /opt/rocm/.info/version; ls /opt/rocm/lib/librocprofiler-sdk.so.*.*.*; python3 --version; pip list 2>/dev/null | grep -i -E '^(torch|sglang|amdsmi) '"
docker exec -w "$REPO_DIR" "$NAME" bash -c "pip install -q -e . --no-deps && pip install -q pytest pytest-forked && graphsignal-run --version"
echo "ready: docker exec -w $REPO_DIR $NAME bash scripts/scenarios/sglang_scenarios.sh baseline prof"
