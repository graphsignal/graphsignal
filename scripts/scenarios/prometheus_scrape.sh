#!/bin/bash
# Real Prometheus scrape: run a workload under graphsignal-run, point an
# unmodified Prometheus server at the watcher's GET /metrics, and check that
# the target is up, that queries return the profiler's series, and that
# `promtool check metrics` parses the exposition.
#
# Usage:
#   bash scripts/scenarios/prometheus_scrape.sh [--gpu] [workload command...]
#
#   (no command)  python3 -c 'import time; time.sleep(...)' — host and
#                 process metrics only
#   --gpu         an inline torch matmul loop on CUDA, so the kernel profile
#                 fills and `topk(rate(cuda_kernels_nanoseconds_total))` is
#                 checked too
#
# Prometheus runs in Docker (`prom/prometheus`, host networking on Linux,
# host.docker.internal on macOS) or, without Docker, from the release tarball
# downloaded into $OUT_DIR.
#
# Environment:
#   SIGNALS_PORT   the profiler's endpoint port (default 18259)
#   PROM_PORT      Prometheus web port (default 19090)
#   PROM_VERSION   Prometheus release (default 3.5.0)
#   PYTHON         interpreter for the default workloads (default python3)
#   GS_RUN         graphsignal-run command (default: graphsignal-run on PATH,
#                  else $PYTHON -m graphsignal.commands.graphsignal_run)
#   DURATION       seconds the default workloads run (default 180)
#   OUT_DIR        config, logs, scraped text (default /tmp/prometheus_scrape)
#
# promtool's lint flags two things the exposition does on purpose and that are
# allowed here: nanosecond units (the names match /signals and the probes) and
# colons in engine metric names (vLLM's `vllm:` families pass through as is).
set -u
SIGNALS_PORT=${SIGNALS_PORT:-18259}
PROM_PORT=${PROM_PORT:-19090}
PROM_VERSION=${PROM_VERSION:-3.5.0}
PYTHON=${PYTHON:-python3}
DURATION=${DURATION:-180}
OUT_DIR=${OUT_DIR:-/tmp/prometheus_scrape}
if [ -z "${GS_RUN:-}" ]; then
  if command -v graphsignal-run >/dev/null 2>&1; then GS_RUN=graphsignal-run
  else GS_RUN="$PYTHON -m graphsignal.commands.graphsignal_run"; fi
fi

GPU=0
if [ "${1:-}" = "--gpu" ]; then GPU=1; shift; fi
if [ $# -eq 0 ]; then
  if [ $GPU -eq 1 ]; then
    set -- "$PYTHON" -c "
import time, torch
a = torch.randn(2048, 2048, device='cuda')
b = torch.randn(2048, 2048, device='cuda')
end = time.time() + $DURATION
while time.time() < end:
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    time.sleep(0.05)
"
  else
    set -- "$PYTHON" -c "import time; time.sleep($DURATION)"
  fi
fi

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR/data"
CONTAINER="graphsignal-prom-$$"
WORKLOAD_PID=""
PROM_PID=""
USE_DOCKER=0
if docker info >/dev/null 2>&1; then USE_DOCKER=1; fi

case "$(uname -s)" in
  Darwin) LISTEN_HOST=0.0.0.0; TARGET_HOST=host.docker.internal ;;
  *)      LISTEN_HOST=127.0.0.1; TARGET_HOST=127.0.0.1 ;;
esac
[ $USE_DOCKER -eq 0 ] && { LISTEN_HOST=127.0.0.1; TARGET_HOST=127.0.0.1; }

cleanup() {
  [ -n "$WORKLOAD_PID" ] && kill "$WORKLOAD_PID" 2>/dev/null
  [ -n "$PROM_PID" ] && kill "$PROM_PID" 2>/dev/null
  [ $USE_DOCKER -eq 1 ] && docker rm -f "$CONTAINER" >/dev/null 2>&1
  [ -n "$WORKLOAD_PID" ] && wait "$WORKLOAD_PID" 2>/dev/null
}
trap cleanup EXIT

FAILURES=0
check() {
  local name=$1 ok=$2 detail=${3:-}
  if [ "$ok" = 1 ]; then echo "PASS  $name $detail"
  else echo "FAIL  $name $detail"; FAILURES=$((FAILURES + 1)); fi
}

# Number of series in a /api/v1/query result, and the series themselves.
query() {
  curl -s -m 5 -G "http://127.0.0.1:$PROM_PORT/api/v1/query" --data-urlencode "query=$1"
}
series_count() {
  "$PYTHON" -c 'import json,sys
try: print(len(json.load(sys.stdin)["data"]["result"]))
except Exception: print(0)'
}
print_series() {
  "$PYTHON" -c 'import json,sys
for r in json.load(sys.stdin)["data"]["result"][:5]:
    print("        ", r["metric"].get("__name__", ""), {k: v for k, v in r["metric"].items() if k not in ("__name__", "job", "instance")}, r["value"][1])'
}

echo "=== workload: $GS_RUN --listen-host $LISTEN_HOST --listen-port $SIGNALS_PORT $*"
$GS_RUN --listen-host "$LISTEN_HOST" --listen-port "$SIGNALS_PORT" "$@" > "$OUT_DIR/workload.log" 2>&1 &
WORKLOAD_PID=$!

up=0
for _ in $(seq 1 60); do
  if curl -s -m 2 -o "$OUT_DIR/metrics.txt" "http://127.0.0.1:$SIGNALS_PORT/metrics"; then up=1; break; fi
  sleep 1
done
check "endpoint serves /metrics" $up
[ $up -eq 1 ] || { tail -20 "$OUT_DIR/workload.log"; exit 1; }

cat > "$OUT_DIR/prometheus.yml" <<EOF
global:
  scrape_interval: 1s
scrape_configs:
  - job_name: graphsignal
    static_configs:
      - targets: ['$TARGET_HOST:$SIGNALS_PORT']
EOF

if [ $USE_DOCKER -eq 1 ]; then
  IMAGE="prom/prometheus:v$PROM_VERSION"
  if [ "$(uname -s)" = Darwin ]; then NET=(-p "$PROM_PORT:$PROM_PORT"); else NET=(--network host); fi
  echo "=== prometheus: $IMAGE (docker)"
  docker run -d --rm --name "$CONTAINER" "${NET[@]}" \
    -v "$OUT_DIR/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
    "$IMAGE" --config.file=/etc/prometheus/prometheus.yml \
    --web.listen-address=":$PROM_PORT" --storage.tsdb.path=/prometheus >/dev/null || exit 1
  PROMTOOL=(docker run --rm -i --entrypoint promtool "$IMAGE")
else
  case "$(uname -m)" in aarch64|arm64) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
  OS=$(uname -s | tr '[:upper:]' '[:lower:]')
  DIST="prometheus-$PROM_VERSION.$OS-$ARCH"
  if [ ! -x "$OUT_DIR/$DIST/prometheus" ]; then
    echo "=== downloading $DIST"
    curl -sfL "https://github.com/prometheus/prometheus/releases/download/v$PROM_VERSION/$DIST.tar.gz" \
      | tar -xz -C "$OUT_DIR" || { echo "download failed"; exit 1; }
  fi
  echo "=== prometheus: $OUT_DIR/$DIST (binary)"
  "$OUT_DIR/$DIST/prometheus" --config.file="$OUT_DIR/prometheus.yml" \
    --web.listen-address="127.0.0.1:$PROM_PORT" --storage.tsdb.path="$OUT_DIR/data" \
    > "$OUT_DIR/prometheus.log" 2>&1 &
  PROM_PID=$!
  PROMTOOL=("$OUT_DIR/$DIST/promtool")
fi

health=""
for _ in $(seq 1 60); do
  health=$(curl -s -m 2 "http://127.0.0.1:$PROM_PORT/api/v1/targets" | "$PYTHON" -c 'import json,sys
try:
    t = json.load(sys.stdin)["data"]["activeTargets"][0]
    print(t["health"], t.get("lastError", ""))
except Exception: print("")')
  case "$health" in up*) break ;; esac
  sleep 1
done
check "target health" "$([ "${health%% *}" = up ] && echo 1 || echo 0)" "($health)"

# A few scrapes in, so the series exist and rate() has two samples.
sleep 10
QUERIES=(graphsignal_profiler_info process_memory_usage_bytes host_memory_usage_bytes)
[ $GPU -eq 1 ] && QUERIES+=("topk(5, rate(cuda_kernels_nanoseconds_total[30s]))")
for q in "${QUERIES[@]}"; do
  n=0
  for _ in $(seq 1 30); do
    n=$(query "$q" | series_count)
    [ "$n" -gt 0 ] && break
    sleep 2
  done
  check "query $q" "$([ "$n" -gt 0 ] && echo 1 || echo 0)" "($n series)"
  query "$q" | print_series
done

curl -s -m 5 -o "$OUT_DIR/metrics.txt" "http://127.0.0.1:$SIGNALS_PORT/metrics"
"${PROMTOOL[@]}" check metrics < "$OUT_DIR/metrics.txt" > "$OUT_DIR/promtool.txt" 2>&1
promtool_rc=$?
unexpected=$(grep -v -E 'use base unit "seconds" instead of "nanoseconds"|metric names should not contain|^$' "$OUT_DIR/promtool.txt")
allowed=$(grep -c -E 'use base unit "seconds" instead of "nanoseconds"|metric names should not contain' "$OUT_DIR/promtool.txt")
check "promtool check metrics" "$([ -z "$unexpected" ] && echo 1 || echo 0)" \
  "(exit $promtool_rc, $allowed allowed lint notes)"
[ -n "$unexpected" ] && echo "$unexpected" | head -20

echo "=== $FAILURES failure(s); files in $OUT_DIR"
[ $FAILURES -eq 0 ]
