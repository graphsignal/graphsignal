# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running tests

```bash
# Python suite (dev task — run as a module, not a published console script)
poetry run python -m scripts.test_local

# Single file / single test
poetry run python -m scripts.test_local test/recorders/test_process_recorder.py
poetry run python -m scripts.test_local test/signals/test_metrics.py::MetricStoreTest::test_add_gauge

# Tests use --forked (each test in a fresh process) and -vv with DEBUG logging by default
```

**CUDA tests:** Any test that touches real CUDA must be marked with `@pytest.mark.cuda`. The default run invokes `pytest-forked`, and a `fork()` after CUDA init corrupts the child's CUDA context; `test/conftest.py` intercepts `cuda`-marked tests and reruns them unforked in a fresh subprocess.

```bash
# Native pure-C++ tests (no GPU; run on macOS or Linux)
make -f Makefile.cupti test-probe test-common

# Native GPU tests (Linux + GPU)
make -f Makefile.cupti test-cupti-activity
make -f Makefile.rocm test-rocm-activity ROCM_MAJOR=7
```

```bash
# Everything on the GPU box in one shot: rsync the repo there, run the Python
# suite and the native tests (CUPTI headers auto-downloaded from the matching
# nvidia-cuda-cupti wheel when the toolkit doesn't ship them)
scripts/test-gpu.sh                 # host: $GRAPHSIGNAL_GPU_HOST or dev-sglang-spark-01
scripts/test-gpu.sh --setup         # first run: also installs deps via scripts/init-novenv.sh
scripts/test-gpu.sh --python-only test/recorders/test_shm_recorder.py
scripts/test-gpu.sh --native-only
```

GPU box: `ssh dev-sglang-spark-01` (DGX Spark, arm64, CUDA 13.x). `scripts/test-gpu-native.sh` is the remote half (runs on the box; also usable directly on any GPU machine).

### Build

```bash
poetry install            # deps (use `poetry config virtualenvs.in-project true` for ./.venv)
poetry build              # wheel/sdist
bash scripts/build-protobuf.sh               # regenerate graphsignal/proto/signals_pb2.py

# Native libraries (multi-arch, via docker buildx; copies .so's into graphsignal/_native/)
make -f Makefile.cupti cupti-buildx cupti-buildx-install
make -f Makefile.rocm rocm-buildx rocm-buildx-install
```

## Architecture

The product is a **universal, local profiler for AI inference workloads**. It observes a workload from a sidecar process and exposes everything through a local HTTP endpoint (`GET /signals`, default port 18259) designed for AI-agent consumers. Uploading is an optional addon (see Collector below).

### Two-process model

`graphsignal-run <cmd>` (commands/graphsignal_run.py) consumes its leading flags (`--metrics-port`, `--listen-host`, `--listen-port`, `--cuda-graph-trace`), picks a launcher (vllm/sglang/trtllm/fallback — first `match()` wins), sets up the native injection env vars (`CUDA_INJECTION64_PATH` via profilers/cupti_profiler.py, `ROCP_TOOL_LIBRARIES` via profilers/rocm_profiler.py; `--cuda-graph-trace` becomes `GRAPHSIGNAL_CUDA_GRAPH_TRACE` there too, and the flag beats an inherited value), and hands off to `launchers/supervisor.py::launch_supervised`: the workload runs as a supervised child (tini-style signal forwarding, exact exit-status propagation, console tee to `/dev/shm/graphsignal_log_<pid>/` for LogRecorder), and a **watcher** subprocess (`python -m graphsignal.commands.graphsignal_watch --pid <workload_pid>`) observes it externally. The watcher never shares a process with CUDA.

Launcher argv policy: the workload command passes through byte-for-byte, with exactly two metric-related adjustments — the SGLang launcher appends `--enable-metrics` (its Prometheus endpoint is off by default) and the vLLM launcher strips `--disable-log-stats`.

### Watcher (graphsignal/watcher/)

`graphsignal.watcher.configure()` creates the `Watcher` singleton (access via `graphsignal.watcher.watcher()`; `is_configured()` is the non-raising check). `Watcher.setup()`:

1. Creates the stores: `MetricStore`, `LogStore`, `ResourceStore` (graphsignal/signals/).
2. Sets the `instance.id` tag; global tags are served once as the `/signals` `context` object — stores keep only caller-supplied tags (`process.pid`, `kernel`, `device.uuid`, …) and never touch the singleton.
3. Creates a `Collector` **only if an API key is configured** (api key is optional; env `GRAPHSIGNAL_API_KEY`).
4. Starts the `SignalsEndpoint` (signals/routes.py) on `<listen_host>:<listen_port>` (default `127.0.0.1:18259`) — port conflicts log an error and disable the endpoint, never crash the watcher.
5. Starts a `PidMonitor` (watcher/pid_monitor.py) polling the target and its descendants every 2s, and a 1s tick loop.

On each tick: every recorder's `on_tick()` runs, then the collector's `on_tick()` (when present). On target termination: recorders `finalize()` first (drains a crash's last console lines into the stores), then a final blocking tick, then the watcher process exits.

PidMonitor callbacks → recorder wiring: `on_target_known` → `LogRecorder` (works even if the target dies before being seen alive); `on_target_created` → `HostRecorder` (first — sets host tags), `NVMLRecorder`, `PrometheusRecorder` (only with `--metrics-port` resolved), `ProcessRecorder`, `ShmRecorder`; `on_child_created` → per-child `ProcessRecorder` + `ShmRecorder`.

### Signal stores (graphsignal/signals/)

Plain Python, thread-safe, non-destructive reads:

- `metrics.py` — four types: gauge, counter, histogram, profile. Every metric holds a **single datapoint — the latest snapshot, overwritten by `set_gauge`/`set_counter`/`set_histogram`/`set_profile`** (histograms carry bins/counts and/or the exact cumulative `count`/`sum` plus lifetime `min`/`max` — **either half alone is a valid histogram**, so a Prometheus summary is a histogram with no bins and neither half means the call is rejected; profiles carry a frame-name→cumulative-value dict, capped at `MAX_PROFILE_FRAMES = 250`). The store key includes the type, so one name can exist as both counter and histogram. Cleanup runs on `set_*` at most once per minute: metrics not updated for 10 min expire. `MAX_METRICS = 5000` series cap. `export()` returns a snapshot and never resets the store.
- `logs.py` — ring of the last 200 entries; `last_entries(min_level, limit)` feeds the endpoint's `errors` section.
- `resources.py` — keyed `(kind, tags)`; keeps earliest `first_seen_ts`, latest `last_seen_ts`. Kinds: `host`, `process`, `device`.
- `routes.py` — the watcher's local HTTP routes: a stdlib `ThreadingHTTPServer` (`SignalsEndpoint`) serving `build_payload()` at `GET /signals` (+ `/health`, named to match the platform's). The payload: gauge `value`; counter `total`; histogram `count/sum/min/max` (exact, when the source supplied them; else `null`) plus `mean` (sum / count when exact, else bin-weighted) and `p50/p95` nearest-rank quantiles from the bins (`null` for a histogram with no bins); profile `frames` as a list of `{name, value}` sorted by value descending; plus `errors`, `resources`, `context`. Convention: `null` = not measured, `0` = measured zero.

### Native ↔ watcher interchange (the shm metrics contract)

Writers inside the workload process publish to **`/dev/shm/graphsignal_<pid>/<lib>.json`** (`cupti.json`, `rocm.json`), rewritten atomically (tmp+rename) every second with full cumulative state — no deltas, no timestamped files; every read is an immutable snapshot:

```json
{"version":1, "pid":123, "start_ts":<ns>, "write_ts":<ns>,
 "context": {"rank":"0", "slurm_job_id":"..."},
 "metrics":[
  {"name":"cuda_kernels_nanoseconds","type":"profile","tags":{},
   "frames":{"<kernel symbol>":[<cumulative ns>, <samples>], ...}},
  {"name":"user.hist","type":"histogram","tags":{},
   "bins":[<bin values>],"counts":[...],
   "count":N,"sum":S,"min":m,"max":M},
  {"name":"user.hist_never_recorded","type":"histogram","tags":{}},
  {"name":"cuda_memcpy_bytes","type":"counter","tags":{"kind":"host_to_device"},"value":N},
  {"name":"x.gauge","type":"gauge","value":X}],
 "log":[{"ts":<ns>,"msg":"..."}]}
```

The universal importer is `recorders/shm_recorder.py` (`ShmRecorder`, one per observed pid): it reads every `*.json` in the pid's dir each tick, requires `version == 1`, maps gauge→`set_gauge`, counter→`set_counter` (latest total), histogram→`set_histogram` (bins/counts pass through, with the writer's exact `count/sum/min/max` when present; **each half is optional and an entry with neither is skipped** — that is how an unrecorded instrument stays out of `/signals`; routes.py computes quantiles), profile→`set_profile` (frames pass through as string names, split into value and sample maps). Context entries become tags (`rank`→`process.rank`, `slurm_job_id`→`slurm.job_id`, …); `log[]` re-emits at debug level, deduped by ts. Stale dirs of dead pids are swept at setup (root recorder only). Changing any field of this schema requires changing both the native writers and the importer.

### Native libraries (src/, include/)

Two injection libraries share one internal layer:

- `include/graphsignal/probe.h` (+ `probe_cuda.h`, `probe_rocm.h`) — the **public, vendorable probe API** (Apache-2.0, header-only, C++17). Developers/agents register gauge/counter/histogram/profile instruments and record from host code or kernels; the record path is lock-free (relaxed 64-bit atomics; histograms use fixed 256-slot log-linear bins: exact 0–3, then 4 sub-bins per power of two; profiles map up to 250 named frames to a cumulative counter plus a sample count via `graphsignal_profile_frame_get`/`graphsignal_profile_add`). The process-global registry is exported as `__graphsignal_probe_registry_v1` (C ABI, discoverable via `dlsym(RTLD_DEFAULT)`), with a 4096-instrument cap. The header also carries the reader API (`graphsignal_probe_reader_attach/count/snapshot`) used by the injection libs.
- `src/common/metrics_writer.h` — the **private writer layer** (`graphsignal::MetricsWriter`): the libs' own instruments (mutex per instrument, count/sum/min/max + bins; both halves omitted from the JSON when nothing was recorded), the 1s writer thread with tmp+rename serialization, rank/SLURM context capture, a retained 256-entry log ring (never drained; readers dedup by ts), the flush hook (cupti: `cuptiActivityFlushAll` — **never called on the final shutdown write**, so teardown makes no CUDA calls), and the probe-registry merge: host probes are snapshotted directly; device-storage probes are all handed to `set_device_probe_reader` in one call per write (the CUPTI lib: per device, `cuMemcpyDtoHAsync` of every block back to back into a pinned staging buffer on a private non-blocking stream, then one `cuStreamSynchronize` — ~4 ms per write for 1,440 probes on GB10 — using the probe's own device's primary context — only when the workload already made it active, never creating one — in relaxed capture mode, so the reads neither serialize against the workload's legacy-stream kernels nor invalidate a global-mode CUDA graph capture; nothing on this path is ever freed; the ROCm lib skips device probes). `graphsignal_probe_register_cuda` is capture-safe the same way: a repeat registration is a registry lookup with no CUDA calls, a new one initializes its block on a private non-blocking stream in relaxed mode, and blocks are never freed.
- `src/cupti/cupti_activity.cpp` — CUPTI injection lib (`CUDA_INJECTION64_PATH` → `InitializeInjection`). Enables CONCURRENT_KERNEL, MEMCPY, MEMSET, SYNCHRONIZATION, GRAPH_TRACE; single-pass `bufferCompleted` records cumulative time into profile instruments — `cuda_kernels_nanoseconds` (frames = raw kernel symbols), `cuda_graphs_nanoseconds` (frames = 16-hex FNV-1a64 of the structural graph signature), `cuda_memcpy_nanoseconds`/`cuda_memset_nanoseconds`/`cuda_sync_nanoseconds` (frames = kind/type) — plus `cuda_memcpy_bytes{kind}`/`cuda_memset_bytes{kind}` counters. Graph identity comes from RESOURCE-domain callbacks at graph instantiation. Sync records are cast to the base `CUpti_ActivitySynchronization` struct — never the versioned one. **Graph tracing granularity** is chosen once at init from `GRAPHSIGNAL_CUDA_GRAPH_TRACE=graph|node` (`cupti_activity_graph_trace_mode_from_env` → `cupti_activity_start`'s `graph_trace_mode`): `graph` (default) enables GRAPH_TRACE, so a replay is one record and its kernels are not instrumented; `node` skips only that one `cuptiActivityEnable`, so the graph's kernels arrive through the already-enabled CONCURRENT_KERNEL kind and land in `cuda_kernels_nanoseconds` while `cuda_graphs_nanoseconds` stays empty. The resource callback runs in both modes. An unrecognized value falls back to `graph` with a debug note. The mode is published as the `cuda_graph_trace_mode` gauge (0 graph, 1 node) — the only reason an empty `cuda_graphs_nanoseconds` is not ambiguous.
- `src/rocm/rocm_activity.cpp` — rocprofiler-sdk analog (`ROCP_TOOL_LIBRARIES` → `rocprofiler_configure`); LOSSLESS double-buffering; kernel dispatch + memory copy + five HIP sync APIs → `rocm_kernels_nanoseconds`/`rocm_memcpy_nanoseconds`/`rocm_sync_nanoseconds` profiles + `rocm_memcpy_bytes{kind}` counters; `tool_fini` never calls `rocprofiler_stop_context`.

**Hard constraints for the native libs:** they run inside the inference engine's process — they must never crash, hang, or measurably slow the workload; exceptions must never cross a C callback boundary; no CUDA/ROCm calls on the teardown path; `-Wl,-z,nodelete`; all memory bounded. Env vars read: `GRAPHSIGNAL_DEBUG`, `GRAPHSIGNAL_CUDA_GRAPH_TRACE` (CUPTI lib only; the ROCm lib has no graph/node distinction) and the rank/SLURM context family — nothing else.

Build floor: the CUPTI `.so` is built on UBI8 (glibc 2.28) via `Dockerfile.cupti` so it loads in older inference containers; the prebuilt per-arch artifacts are committed under `graphsignal/_native/<arch>-cu<major>/` and ship in the wheel.

### Collector — production feedback loop (graphsignal/collector/, optional addon)

Active **only when `GRAPHSIGNAL_API_KEY` is set**. `Collector.on_tick()` snapshots the stores via their non-destructive `export()` and keeps all delta state itself: gauges upload only when the snapshot ts advanced; counters upload the increase since last upload; histograms upload per-bin count deltas and count/sum deltas independently (whichever halves the source reports) with `min`/`max` sent **undeltaed** — they are lifetime extremes and the difference of two of them is not the interval's extreme; profiles upload per-frame value and sample-count deltas (all skipped when unchanged). Profile frames carry string names everywhere up to this point — the collector is where they are encoded to integer frame ids (`xxhash.xxh64(name)`) as `ProfileFrame{frame_id, name}` + `dp.profile.frame_ids/values`. Snapshots convert to protobuf (compiled from `proto/signals.proto` into `graphsignal/proto/signals_pb2.py` via `scripts/build-protobuf.sh`, which PINS protoc — the gencode it stamps is the `protobuf` floor users must satisfy, and `test/test_proto_version.py` fails if the two drift) with the watcher's global tags merged per signal, and `SignalUploader` gzips an `UploadRequest` to `{GRAPHSIGNAL_API_BASE:-https://api.graphsignal.com}/api/v1/ingest` with the `X-API-Key` header. Without a key, the collector is never constructed — zero upload code on the core path.

**The one exception, and it is keyless.** `graphsignal/watcher/version_check.py` GETs `{GRAPHSIGNAL_API_BASE:-https://api.graphsignal.com}/api/v1/version_check?version=<__version__>` once per run, fired from `Watcher.on_target_created` on a daemon thread of its own (that callback runs on the pid monitor's poll loop, which must not block). It compares `(major, minor)` only, so a patch release says nothing, and `GRAPHSIGNAL_DISABLE_VERSION_CHECK=1` turns it off. The notice is `logger.warning`, which puts it in the `errors` array of `GET /signals` **on purpose** — an agent reading the payload can act on a stale profiler. Every failure path is caught and logged at `debug` only: a version check that did not work must never make a working run look broken. `test/test_utils.py` sets the disable flag so no test phones home.

### Recorders (graphsignal/recorders/)

All inherit `BaseRecorder(root_pid, pid, args)`. Metric emission uses `set_gauge`/`set_counter`/`set_histogram` — every call overwrites the metric's snapshot; counters always carry **cumulative totals** (recorders keep running totals where the source is event-based, e.g. `gpu_xid_critical_errors`). `PrometheusRecorder` scrapes one explicitly configured endpoint (never port-scans — probing unknown sockets can corrupt engines' IPC) and passes scraped values through as-is: raw cumulative counter totals, and histogram AND summary families as one histogram each — the exact `_count`/`_sum`, plus the `le` buckets converted to non-cumulative bins where the family has them (a summary has none, and its quantiles are not kept). `LogRecorder` tails the supervisor's console segments and extracts error-level lines (vLLM/SGLang/uvicorn/TRT-LLM formats, multi-line tracebacks, and the supervisor's terminal exit-status record). Recorders that emit per-process metrics must tag them with `process.pid`.
