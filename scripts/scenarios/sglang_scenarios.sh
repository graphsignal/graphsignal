#!/bin/bash
# Engine-level scenarios: run SGLang under graphsignal-run (or bare, or under
# rocprofv3), drive it with sglang.bench_serving, and report whether the
# workload survived, what it served, and what /signals held afterwards.
#
# Usage (on the GPU machine, inside the engine's environment):
#   bash scripts/scenarios/sglang_scenarios.sh <scenario>...
#
# Scenarios:
#   baseline        SGLang alone — the number everything else is compared to
#   prof            graphsignal-run, default modes
#   prof_kernels    graphsignal-run --rocm-kernel-trace all (ROCm per-kernel tracing)
#   prof_nograph    graphsignal-run with --disable-cuda-graph (eager engine)
#   rocprofv3       AMD's rocprofv3 --kernel-trace instead of graphsignal-run
#
# Environment:
#   MODEL        model path (default Qwen/Qwen2.5-0.5B-Instruct)
#   PORT         SGLang port (default 30000)
#   HOST         SGLang bind host (default 127.0.0.1)
#   SIGNALS_PORT the profiler's /signals port (default 18259)
#   OUT_DIR      logs and payloads (default /tmp/sglang_scenarios)
#   NUM_PROMPTS, CONCURRENCY, INPUT_LEN, OUTPUT_LEN   benchmark shape
#   SGL_EXTRA    extra arguments appended to every sglang.launch_server
#
# Per scenario the summary line says: requests served, throughput, mean
# inter-token latency, crash markers found in the server log, whether the
# server was still alive after the benchmark, and the size of the /signals
# payload (fetched 70 s after the benchmark so the engine's Prometheus
# metrics have been scraped at least once).
#
# The launch commands are chosen in here by scenario name so that this
# script's own command line never matches the process patterns it kills.
set -u
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${PORT:-30000}
HOST=${HOST:-127.0.0.1}
SIGNALS_PORT=${SIGNALS_PORT:-18259}
OUT_DIR=${OUT_DIR:-/tmp/sglang_scenarios}
NUM_PROMPTS=${NUM_PROMPTS:-300}
CONCURRENCY=${CONCURRENCY:-32}
INPUT_LEN=${INPUT_LEN:-256}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SGL_EXTRA=${SGL_EXTRA:-}
SGL_ARGS="--model-path $MODEL --host $HOST --port $PORT --enable-metrics $SGL_EXTRA"
mkdir -p "$OUT_DIR"
# With host networking (a container started with --network host) gloo must be
# told which interface to use or SGLang fails with "Unable to find interface".
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}

kill_sglang() {
  # Anchored so the pattern never matches a shell whose command line merely
  # mentions sglang.launch_server (this script's own, for one).
  pkill -f "^[^ ]*python[^ ]* .*sglang\.launch_server"
  sleep 3
  pkill -f "^[^ ]*python[^ ]* -c from multiprocessing"
  sleep 6
}

run_scenario() {
  local name=$1; shift
  local log="$OUT_DIR/server_$name.log"
  echo "=== $name: $*"
  kill_sglang
  rm -rf /dev/shm/graphsignal_*
  (cd "$OUT_DIR" && GRAPHSIGNAL_DEBUG=1 "$@" > "$log" 2>&1) &
  local up=0 i
  for i in $(seq 1 60); do
    sleep 5
    if curl -s -m 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  done
  if [ $up -eq 0 ]; then
    echo "$name: server failed to start (see $log)"; tail -5 "$log"; kill_sglang; return
  fi
  echo "$name: server up after $((i*5))s"
  sleep 20
  timeout 600 python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" > "$OUT_DIR/bench_$name.log" 2>&1
  grep -E "Successful requests|Request throughput|Mean ITL" "$OUT_DIR/bench_$name.log"
  echo "$name: crash markers: $(grep -c -E 'Segfault encountered|crashed with exit code|terminated by signal 11' "$log")"
  if curl -s -m 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "$name: ALIVE after bench"
  else
    echo "$name: DEAD after bench"
  fi
  if ls /dev/shm/graphsignal_* >/dev/null 2>&1; then
    sleep 70
    curl -s -m 5 "http://127.0.0.1:$SIGNALS_PORT/signals" > "$OUT_DIR/signals_$name.json"
    echo "$name: signals bytes $(wc -c < "$OUT_DIR/signals_$name.json") -> $OUT_DIR/signals_$name.json"
  fi
  kill_sglang
}

for name in "$@"; do
  case "$name" in
    baseline)     run_scenario baseline python3 -m sglang.launch_server $SGL_ARGS ;;
    prof)         run_scenario prof graphsignal-run python3 -m sglang.launch_server $SGL_ARGS ;;
    prof_kernels) run_scenario prof_kernels graphsignal-run --rocm-kernel-trace all python3 -m sglang.launch_server $SGL_ARGS ;;
    prof_nograph) run_scenario prof_nograph graphsignal-run python3 -m sglang.launch_server $SGL_ARGS --disable-cuda-graph ;;
    rocprofv3)    mkdir -p "$OUT_DIR/rocprofv3"; run_scenario rocprofv3 rocprofv3 --kernel-trace --output-format csv -d "$OUT_DIR/rocprofv3" -- python3 -m sglang.launch_server $SGL_ARGS ;;
    *) echo "unknown scenario: $name (baseline|prof|prof_kernels|prof_nograph|rocprofv3)" ;;
  esac
done
echo "=== done: $*"
