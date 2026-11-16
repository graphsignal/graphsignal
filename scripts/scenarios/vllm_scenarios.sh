#!/bin/bash
# Engine-level scenarios: run vLLM under graphsignal-run (or bare), drive it
# with `vllm bench serve`, and report whether the workload survived, what it
# served, and what /signals held before and after the benchmark.
#
# Usage (on the GPU machine, inside the engine's environment, with `vllm`
# and `graphsignal-run` on PATH):
#   bash scripts/scenarios/vllm_scenarios.sh <scenario>...
#   TP=2 bash scripts/scenarios/vllm_scenarios.sh baseline prof
#
# Scenarios:
#   baseline      vLLM alone — the number everything else is compared to
#   prof          graphsignal-run, default modes
#   prof_debug    graphsignal-run with GRAPHSIGNAL_DEBUG=1
#   cupti_only    vLLM with only the CUPTI injection library loaded, as
#                 graphsignal-run would set it up — no watcher, no console tee
#   watch_only    bare vLLM with only the watcher attached (graphsignal-watch:
#                 NVML, /proc, Prometheus scrape, health probe) — no injection
#   prof_node     graphsignal-run --cuda-graph-trace node (kernels inside graphs)
#   prof_eager    graphsignal-run with --enforce-eager (no CUDA graphs)
#   engine_death  graphsignal-run; SIGKILL the EngineCore process and record
#                 what /health and the profiler's engine_* and errors report
#   graceful_stop graphsignal-run; SIGTERM the server (a normal shutdown) and
#                 record whether the health counters call it an outage
#   api_key       graphsignal-run with --api-key: /health and /metrics must
#                 stay reachable without the key, and uptime must accrue
#
# Environment:
#   MODEL        model (default Qwen/Qwen2.5-0.5B-Instruct)
#   TP           tensor-parallel size (default 1)
#   PORT         vLLM port (default 8000)
#   HOST         vLLM bind host (default 127.0.0.1)
#   SIGNALS_PORT the profiler's /signals port (default 18259)
#   OUT_DIR      logs and payloads (default /tmp/vllm_scenarios)
#   NUM_PROMPTS, CONCURRENCY, INPUT_LEN, OUTPUT_LEN   benchmark shape
#   VLLM_EXTRA   extra arguments appended to every `vllm serve`
#   GS_DEBUG     GRAPHSIGNAL_DEBUG for profiled runs (default 0; prof_debug sets 1)
#   GS_FLAGS     extra graphsignal-run flags for prof, e.g.
#                "--dcgm-exporter http://127.0.0.1:9400/metrics"
#   CUPTI_LIB    cupti_only: inject this libgscuptiprof.so instead of the one
#                graphsignal-run picks (A/B of library builds)
#
# For overhead comparisons use a benchmark that runs for tens of seconds
# (e.g. NUM_PROMPTS=3000); a few seconds of traffic is mostly ramp-up noise.
# cupti_only + watch_only isolate the two halves of `prof`.
#
# Per profiled scenario the summary covers, between a /signals read after
# warm-up and one 70 s after the benchmark (so the engine's Prometheus metrics
# have been scraped at least once): busy and kernel time per GPU, NVLink
# payload per GPU, the engine_* health counters and workload_errors by class.
#
# The launch commands are chosen in here by scenario name so that this
# script's own command line never matches the process patterns it kills.

set -u

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
TP=${TP:-1}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
SIGNALS_PORT=${SIGNALS_PORT:-18259}
OUT_DIR=${OUT_DIR:-/tmp/vllm_scenarios}
NUM_PROMPTS=${NUM_PROMPTS:-300}
CONCURRENCY=${CONCURRENCY:-32}
INPUT_LEN=${INPUT_LEN:-256}
OUTPUT_LEN=${OUTPUT_LEN:-128}
VLLM_EXTRA=${VLLM_EXTRA:-}
API_KEY=${API_KEY:-graphsignal-scenario-key}
GS_DEBUG=${GS_DEBUG:-0}
GS_FLAGS=${GS_FLAGS:-}

VLLM_ARGS="$MODEL --host $HOST --port $PORT --tensor-parallel-size $TP $VLLM_EXTRA"

mkdir -p "$OUT_DIR"

kill_vllm() {
  # Anchored so the pattern never matches a shell whose command line merely
  # mentions `vllm serve` (this script's own, for one). It matches the API
  # server whether launched bare or through graphsignal-run.
  pkill -f "^[^ ]*python[^ ]* [^ ]*(vllm|graphsignal-run) .*serve" 2>/dev/null
  sleep 3
  # vLLM names its engine and worker processes with setproctitle.
  pkill -9 -f "^VLLM::" 2>/dev/null
  sleep 6
}

health_code() {
  curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null
}

wait_up() {
  local i
  for i in $(seq 1 90); do
    sleep 5
    if [ "$(health_code)" = "200" ]; then echo $((i*5)); return 0; fi
  done
  return 1
}

save_signals() {
  curl -s -m 5 "http://127.0.0.1:$SIGNALS_PORT/signals" > "$1" 2>/dev/null
}

# Deltas between two /signals payloads: device time and NVLink per GPU, plus
# the health counters and error classes as they stand in the second.
summarize() {
  python3 - "$1" "$2" <<'EOF'
import json, sys

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"metrics": []}

def index(payload):
    out = {}
    for m in payload.get("metrics", []):
        s = m.get("stats", {})
        out[(m["name"], tuple(sorted(m.get("tags", {}).items())))] = s
    return out

pre, post = index(load(sys.argv[1])), index(load(sys.argv[2]))

def delta(key, field="total"):
    a = pre.get(key, {}).get(field) or 0
    b = post.get(key, {}).get(field) or 0
    return b - a

def dev(tags):
    t = dict(tags)
    return (t.get("device.uuid") or t.get("device.index") or "?")[-8:]

rows = {}
for (name, tags), stats in post.items():
    if name in ("cuda_device_busy_nanoseconds", "cuda_device_kernel_nanoseconds"):
        rows.setdefault(dev(tags), {})[name] = delta((name, tags)) / 1e9
    if name == "cuda_device_idle_gap_nanoseconds":
        rows.setdefault(dev(tags), {})["gap_p50_us"] = (stats.get("p50") or 0) / 1e3
        rows.setdefault(dev(tags), {})["gap_p95_us"] = (stats.get("p95") or 0) / 1e3
    if name == "gpu_nvlink_throughput_data_tx_kibibytes":
        rows.setdefault(dev(tags), {})["nvlink_tx_gib"] = delta((name, tags)) / 1024**2
for gpu, r in sorted(rows.items()):
    busy = r.get("cuda_device_busy_nanoseconds", 0)
    kern = r.get("cuda_device_kernel_nanoseconds", 0)
    ratio = f"{kern / busy:.3f}" if busy else "-"
    print(f"  gpu ..{gpu}: busy {busy:.2f}s kernel {kern:.2f}s overlap {ratio}"
          f" gap p50 {r.get('gap_p50_us', 0):.0f}us p95 {r.get('gap_p95_us', 0):.0f}us"
          f" nvlink tx {r.get('nvlink_tx_gib', 0):.2f} GiB")
health = {name: s.get("total") for (name, tags), s in post.items() if name.startswith("engine_")}
if health:
    print("  health: " + ", ".join(
        f"{k.replace('engine_', '')}={v / 1e9:.1f}s" if k.endswith("nanoseconds") else f"{k.replace('engine_', '')}={v}"
        for k, v in sorted(health.items())))
errors = {dict(tags).get("class"): s.get("total") for (name, tags), s in post.items() if name == "workload_errors"}
if errors:
    print("  workload_errors: " + ", ".join(f"{k}={v}" for k, v in sorted(errors.items())))
EOF
}

bench() {
  local name=$1
  timeout 900 vllm bench serve --backend vllm --model "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" > "$OUT_DIR/bench_$name.log" 2>&1
  grep -E "Successful requests|Request throughput|Output token throughput|Mean TTFT|Mean ITL" "$OUT_DIR/bench_$name.log"
}

start() {
  local name=$1; shift
  kill_vllm
  rm -rf /dev/shm/graphsignal_*
  (cd "$OUT_DIR" && GRAPHSIGNAL_DEBUG=${SCENARIO_DEBUG:-$GS_DEBUG} "$@" > "$OUT_DIR/server_$name.log" 2>&1) &
  local secs
  if ! secs=$(wait_up); then
    echo "$name: server failed to start (see $OUT_DIR/server_$name.log)"
    tail -5 "$OUT_DIR/server_$name.log"
    kill_vllm
    return 1
  fi
  echo "$name: server up after ${secs}s"
}

run_scenario() {
  local name=$1; shift
  echo "=== $name: $*"
  start "$name" "$@" || return
  sleep 20
  local profiled=0
  ls /dev/shm/graphsignal_* >/dev/null 2>&1 && profiled=1
  [ -n "${WATCH_PID:-}" ] && profiled=1
  [ $profiled -eq 1 ] && save_signals "$OUT_DIR/signals_${name}_pre.json"
  bench "$name"
  echo "$name: crash markers: $(grep -c -E 'Segmentation fault|Fatal Python error|terminated by signal 11|EngineDeadError' "$OUT_DIR/server_$name.log")"
  if [ "$(health_code)" = "200" ]; then echo "$name: ALIVE after bench"; else echo "$name: DEAD after bench"; fi
  if [ $profiled -eq 1 ]; then
    sleep 70
    save_signals "$OUT_DIR/signals_$name.json"
    echo "$name: signals bytes $(wc -c < "$OUT_DIR/signals_$name.json") -> $OUT_DIR/signals_$name.json"
    summarize "$OUT_DIR/signals_${name}_pre.json" "$OUT_DIR/signals_$name.json"
  fi
  kill_vllm
}

engine_death() {
  local name=engine_death
  echo "=== $name: graphsignal-run vllm serve $VLLM_ARGS; then SIGKILL EngineCore"
  start "$name" graphsignal-run vllm serve $VLLM_ARGS || return
  sleep 15  # three healthy probes
  save_signals "$OUT_DIR/signals_${name}_pre.json"
  local engine
  engine=$(pgrep -f "^VLLM::EngineCore" | head -1)
  if [ -z "$engine" ]; then echo "$name: no VLLM::EngineCore process found"; kill_vllm; return; fi
  echo "$name: killing EngineCore pid $engine"
  kill -9 "$engine"
  local i code last="" codes=""
  for i in $(seq 1 30); do
    sleep 1
    code=$(health_code)
    [ "$code" != "$last" ] && codes="$codes ${i}s:${code:-none}" && last=$code
    # Keep the newest payload the profiler served before the workload exited.
    save_signals "$OUT_DIR/signals_${name}.tmp" && [ -s "$OUT_DIR/signals_${name}.tmp" ] \
      && mv "$OUT_DIR/signals_${name}.tmp" "$OUT_DIR/signals_$name.json"
  done
  echo "$name: /health over 30 s after the kill (changes only):$codes"
  if pgrep -f "^[^ ]*python[^ ]* [^ ]*graphsignal-run .*serve" >/dev/null; then
    echo "$name: API server still running"
  else
    echo "$name: API server exited"
  fi
  summarize "$OUT_DIR/signals_${name}_pre.json" "$OUT_DIR/signals_$name.json"
  python3 - "$OUT_DIR/signals_$name.json" <<'EOF'
import json, sys
try:
    errors = json.load(open(sys.argv[1])).get("errors", [])
except Exception:
    errors = []
for e in errors[-5:]:
    print("  error:", (e.get("message") or "")[:160])
EOF
  kill_vllm
}

# Polls /signals until the profiler's endpoint goes away with the workload,
# keeping the newest payload in $2.
keep_last_signals() {
  local secs=$1 out=$2 i
  for i in $(seq 1 $((secs * 2))); do
    sleep 0.5
    save_signals "$out.tmp" && [ -s "$out.tmp" ] && mv "$out.tmp" "$out" || break
  done
  rm -f "$out.tmp"
}

graceful_stop() {
  local name=graceful_stop
  echo "=== $name: graphsignal-run vllm serve $VLLM_ARGS; then SIGTERM"
  start "$name" graphsignal-run vllm serve $VLLM_ARGS || return
  sleep 15  # three healthy probes
  save_signals "$OUT_DIR/signals_${name}_pre.json"
  local sup
  sup=$(pgrep -f "^[^ ]*python[^ ]* [^ ]*graphsignal-run .*serve" | head -1)
  if [ -z "$sup" ]; then echo "$name: graphsignal-run process not found"; kill_vllm; return; fi
  local t0=$SECONDS
  kill -TERM "$sup"
  keep_last_signals 120 "$OUT_DIR/signals_$name.json"
  echo "$name: profiler endpoint gone $((SECONDS - t0))s after SIGTERM"
  summarize "$OUT_DIR/signals_${name}_pre.json" "$OUT_DIR/signals_$name.json"
  kill_vllm
}

# The injection environment graphsignal-run would give the workload
# (CUDA_INJECTION64_PATH and the libcupti directory), read off a throwaway
# `graphsignal-run env`.
injection_env() {
  GRAPHSIGNAL_DEBUG=0 graphsignal-run --listen-port $((SIGNALS_PORT + 1)) env 2>/dev/null \
    | grep -E '^(CUDA_INJECTION64_PATH|LD_LIBRARY_PATH|GRAPHSIGNAL_CUDA_GRAPH_TRACE)=' | tr '\n' ' '
}

cupti_only() {
  local inj
  inj=$(injection_env)
  if [ -n "${CUPTI_LIB:-}" ]; then
    inj="$(echo "$inj" | tr ' ' '\n' | grep -v '^CUDA_INJECTION64_PATH=' | tr '\n' ' ')CUDA_INJECTION64_PATH=$CUPTI_LIB"
  fi
  if [ -z "$inj" ]; then echo "cupti_only: graphsignal-run set up no CUPTI injection"; return; fi
  echo "cupti_only: $inj"
  run_scenario cupti_only env $inj vllm serve $VLLM_ARGS
  rm -rf /dev/shm/graphsignal_*
}

watch_only() {
  local name=watch_only
  echo "=== $name: vllm serve $VLLM_ARGS + graphsignal-watch"
  start "$name" vllm serve $VLLM_ARGS || return
  local api
  api=$(pgrep -f "^[^ ]*python[^ ]* [^ ]*vllm serve" | head -1)
  python3 -m graphsignal.commands.graphsignal_watch --pid "$api" --metrics-port "$PORT" \
    --health-port "$PORT" --health-path /health --health-host 127.0.0.1 \
    --listen-port "$SIGNALS_PORT" > "$OUT_DIR/watch_$name.log" 2>&1 &
  WATCH_PID=$!
  sleep 20
  save_signals "$OUT_DIR/signals_${name}_pre.json"
  bench "$name"
  sleep 70
  save_signals "$OUT_DIR/signals_$name.json"
  summarize "$OUT_DIR/signals_${name}_pre.json" "$OUT_DIR/signals_$name.json"
  kill "$WATCH_PID" 2>/dev/null
  unset WATCH_PID
  kill_vllm
}

api_key() {
  local name=api_key
  echo "=== $name: graphsignal-run vllm serve $VLLM_ARGS --api-key ***"
  start "$name" graphsignal-run vllm serve $VLLM_ARGS --api-key "$API_KEY" || return
  echo "$name: /health without key -> $(health_code)"
  echo "$name: /metrics without key -> $(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/metrics")"
  echo "$name: /v1/models without key -> $(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models")"
  sleep 30
  save_signals "$OUT_DIR/signals_$name.json"
  python3 - "$OUT_DIR/signals_$name.json" <<'EOF'
import json, sys
metrics = json.load(open(sys.argv[1])).get("metrics", [])
up = [m["stats"].get("total") for m in metrics if m["name"] == "engine_uptime_nanoseconds"]
scraped = sum(1 for m in metrics if m["name"].startswith("vllm:"))
print(f"  engine_uptime_nanoseconds={up[0] / 1e9 if up else None}s, vllm:* metrics scraped: {scraped}")
EOF
  kill_vllm
}

for name in "$@"; do
  case "$name" in
    baseline)     run_scenario baseline vllm serve $VLLM_ARGS ;;
    prof)         run_scenario prof graphsignal-run $GS_FLAGS vllm serve $VLLM_ARGS ;;
    prof_debug)   SCENARIO_DEBUG=1 run_scenario prof_debug graphsignal-run vllm serve $VLLM_ARGS ;;
    cupti_only)   cupti_only ;;
    watch_only)   watch_only ;;
    prof_node)    run_scenario prof_node graphsignal-run --cuda-graph-trace node vllm serve $VLLM_ARGS ;;
    prof_eager)   run_scenario prof_eager graphsignal-run vllm serve $VLLM_ARGS --enforce-eager ;;
    engine_death) engine_death ;;
    graceful_stop) graceful_stop ;;
    api_key)      api_key ;;
    *) echo "unknown scenario: $name (baseline|prof|prof_debug|cupti_only|watch_only|prof_node|prof_eager|engine_death|graceful_stop|api_key)" ;;
  esac
done
echo "=== done: $*"
