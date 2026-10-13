#include "rocm_activity.h"

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/hip/api_id.h>

#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "metrics_writer.h"

// Log a rocprofiler_status_t failure without aborting — mirrors CUPTI_CALL.
#ifndef ROCP_CALL
#define ROCP_CALL(call)                                                        \
  do {                                                                         \
    rocprofiler_status_t _status = (call);                                     \
    if (_status != ROCPROFILER_STATUS_SUCCESS && g_writer) {                  \
      g_writer->error("rocprofiler error %d (%s) at %s:%d", _status,          \
                      rocprofiler_get_status_string(_status), __FILE__,        \
                      __LINE__);                                               \
    }                                                                          \
  } while (0)
#endif

namespace {

// rocprofiler buffer sizing. LOSSLESS applies backpressure (an emplace waits on
// a flush) ONLY when the active internal buffer fills while the other is still
// draining. We avoid that entirely by: (a) keeping buffer_callback I/O-free and
// cheap (it only records into Writer instruments — shm serialization happens on
// the Writer's own thread), (b) draining on a dedicated callback thread, and
// (c) flushing early. A 4 MB buffer (~22k kernel-dispatch records) with a
// 1 MB watermark means we swap+drain at ~25% full, so a burst can't catch both
// double-buffers busy. The Writer's ~1s flush hook bounds staleness independently.
constexpr size_t kBufferSize = 4 * 1024 * 1024;
constexpr size_t kBufferWatermark = 1 * 1024 * 1024;

std::atomic<bool> g_running{false};
std::atomic<bool> g_debug_enabled{false};
graphsignal::MetricsWriter* g_writer = nullptr;

// Per-kernel tracing, chosen once at tool_init from
// GRAPHSIGNAL_ROCM_KERNEL_TRACE=none|all (default none). rocprofiler-sdk
// intercepts the completion of every traced dispatch — an interrupt-backed
// signal and a handler-thread wakeup each, tens of microseconds on MI300X —
// and a graph replay is unrolled into ordinary dispatches, so there is no
// cheap granularity to fall back to as with CUPTI's graph mode. With `none`
// the KERNEL_DISPATCH service (and the CODE_OBJECT callback that only serves
// it) is never configured and the workload runs at native speed; memory copy
// and HIP sync tracing stay on. Published as the rocm_kernel_trace_mode gauge
// (0 none, 1 all) so an empty rocm_kernels_nanoseconds is never ambiguous.
enum RocmKernelTraceMode : uint32_t { kKernelTraceNone = 0, kKernelTraceAll = 1 };
uint32_t g_kernel_trace_mode = kKernelTraceNone;
// An unrecognized GRAPHSIGNAL_ROCM_KERNEL_TRACE value, kept until the writer's
// log exists so the fallback can be noted.
std::string g_kernel_trace_env_rejected;
graphsignal::Instrument* g_kernel_trace_mode_gauge = nullptr;
uint32_t kernel_trace_mode_from_env();

// rocprofiler handles created at tool_init.
rocprofiler_context_id_t        g_ctx_id{0};
rocprofiler_buffer_id_t         g_buffer_id{0};
rocprofiler_callback_thread_t   g_cb_thread{0};

// kernel_id -> compiled symbol name, built from CODE_OBJECT device-kernel
// symbol-register callbacks. Unlike CUPTI (name inline in the kernel record),
// ROCm delivers the name via a SYNCHRONOUS callback on the app's module-load
// thread while the buffer callback reads it on rocprofiler's thread, so this
// MUST be mutex-guarded. Populated on PHASE_LOAD; NEVER erased on PHASE_UNLOAD
// (we flush the buffer before unload, but late dispatch records must still
// resolve rather than fall back to "kernel_<id>").
constexpr size_t kMaxTrackedKernels = 65536;
std::unordered_map<uint64_t, std::string> g_kernel_syms;
std::mutex g_kernel_syms_mu;

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

// One mutex guards all lazily-registered instrument lookups below. Records are
// processed on rocprofiler's callback thread (or a flushing thread), so lookups
// must be thread-safe; the per-record cost is one uncontended lock.
std::mutex g_instruments_mu;

// GPU activity aggregates into profile instruments: one metric per activity
// family, frames keyed by kernel symbol / kind / HIP op. Frame cardinality is
// bounded by the writer (kMaxProfileFrames).
graphsignal::Instrument* g_kernels_profile = nullptr;
graphsignal::Instrument* g_memcpy_profile = nullptr;
graphsignal::Instrument* g_sync_profile = nullptr;

// rocm_memcpy_bytes per copy kind, tagged {"kind": <kind>}.
enum MemcpyKind {
  kMemcpyHostToDevice = 0,
  kMemcpyDeviceToHost,
  kMemcpyDeviceToDevice,
  kMemcpyHostToHost,
  kMemcpyOther,
  kMemcpyKindCount
};
const char* const kMemcpyKindStr[kMemcpyKindCount] = {
    "host_to_device", "device_to_host", "device_to_device", "host_to_host",
    "other"};
graphsignal::Instrument* g_memcpy_bytes[kMemcpyKindCount] = {};
bool g_memcpy_registered[kMemcpyKindCount] = {};

// The traced HIP synchronization ops.
constexpr uint32_t kHipSyncOps[] = {
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamSynchronize,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSynchronize,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipEventSynchronize,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord,
};
constexpr size_t kNumHipSyncOps = sizeof(kHipSyncOps) / sizeof(kHipSyncOps[0]);

int memcpyKindIndex(rocprofiler_memory_copy_operation_t op) {
  switch (op) {
    case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:   return kMemcpyHostToDevice;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:   return kMemcpyDeviceToHost;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE: return kMemcpyDeviceToDevice;
    case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:     return kMemcpyHostToHost;
    default:                                        return kMemcpyOther;
  }
}

int hipSyncOpIndex(uint32_t op) {
  for (size_t i = 0; i < kNumHipSyncOps; ++i) {
    if (kHipSyncOps[i] == op) return static_cast<int>(i);
  }
  return -1;
}

// The traced ops' function names ("hipStreamSynchronize", ...), resolved ONCE
// at tool_init by index into kHipSyncOps. The buffer callback must never call
// back into rocprofiler-sdk: the final flush runs from tool_fini, i.e. inside
// the SDK's own atexit finalization, and an SDK query from that callback
// (rocprofiler_query_buffer_tracing_kind_operation_name, as a lazy cache did)
// segfaults on rocprofiler-sdk 1.3.5 whenever a traced op first shows up in
// that last batch — torch records its stream events while tearing down.
std::string g_hip_sync_op_names[kNumHipSyncOps];

void resolve_hip_sync_op_names() {
  for (size_t i = 0; i < kNumHipSyncOps; ++i) {
    const char* name = nullptr;
    uint64_t name_len = 0;
    if (rocprofiler_query_buffer_tracing_kind_operation_name(
            ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
            static_cast<rocprofiler_tracing_operation_t>(kHipSyncOps[i]), &name,
            &name_len) == ROCPROFILER_STATUS_SUCCESS &&
        name) {
      g_hip_sync_op_names[i] = name;
    } else {
      g_hip_sync_op_names[i] = "hip_op_" + std::to_string(kHipSyncOps[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Record handling
// ---------------------------------------------------------------------------

void handle_kernel_dispatch(const rocprofiler_buffer_tracing_kernel_dispatch_record_t* k) {
  if (!k) return;
  if (k->end_timestamp <= k->start_timestamp) return;
  const uint64_t duration = k->end_timestamp - k->start_timestamp;

  // Resolve the compiled symbol name from the code-object map.
  const uint64_t kernel_id = k->dispatch_info.kernel_id;
  std::string name;
  {
    std::lock_guard<std::mutex> lock(g_kernel_syms_mu);
    auto it = g_kernel_syms.find(kernel_id);
    if (it != g_kernel_syms.end()) name = it->second;
  }
  if (name.empty()) name = "kernel_" + std::to_string(kernel_id);

  graphsignal::MetricsWriter::profile_add(g_kernels_profile, name.c_str(), duration);
}

void handle_memory_copy(const rocprofiler_buffer_tracing_memory_copy_record_t* m) {
  if (!m) return;
  if (m->end_timestamp <= m->start_timestamp) return;
  const uint64_t duration = m->end_timestamp - m->start_timestamp;

  const int kind = memcpyKindIndex(m->operation);
  // `bytes` was added to the memory-copy record in later rocprofiler-sdk
  // revisions; guard on the record's own size field for forward/backward compat.
  uint64_t bytes = 0;
  if (m->size >= offsetof(rocprofiler_buffer_tracing_memory_copy_record_t, bytes) +
                     sizeof(m->bytes)) {
    bytes = m->bytes;
  }

  graphsignal::Instrument* bytes_inst = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_instruments_mu);
    if (!g_memcpy_registered[kind]) {
      g_memcpy_bytes[kind] = g_writer->register_instrument(
          graphsignal::InstrumentType::Counter, "rocm_memcpy_bytes",
          {{"kind", kMemcpyKindStr[kind]}});
      g_memcpy_registered[kind] = true;
    }
    bytes_inst = g_memcpy_bytes[kind];
  }
  graphsignal::MetricsWriter::profile_add(
      g_memcpy_profile, kMemcpyKindStr[kind], duration);
  graphsignal::MetricsWriter::add(bytes_inst, bytes);
}

void handle_hip_api(const rocprofiler_buffer_tracing_hip_api_record_t* s) {
  if (!s) return;
  if (s->end_timestamp <= s->start_timestamp) return;
  const uint64_t duration = s->end_timestamp - s->start_timestamp;

  const int idx = hipSyncOpIndex(s->operation);
  if (idx < 0) return;

  graphsignal::MetricsWriter::profile_add(
      g_sync_profile, g_hip_sync_op_names[idx].c_str(), duration);
}

// rocprofiler buffer-tracing callback. Delivered on our assigned callback
// thread (or the thread calling rocprofiler_flush_buffer).
void buffer_callback(rocprofiler_context_id_t /*context*/,
                     rocprofiler_buffer_id_t /*buffer_id*/,
                     rocprofiler_record_header_t** headers, size_t num_headers,
                     void* /*user_data*/, uint64_t drop_count) {
  // Late callback after stop / during teardown: bail without touching shared
  // state (rocprofiler owns the record memory, so nothing to free).
  if (!g_running.load(std::memory_order_relaxed)) return;

  try {
    // Per-kind counters (debug only) so we can see, per callback, whether
    // KERNEL_DISPATCH records are actually arriving vs. the callback being
    // dominated by HIP sync/memcpy records. This is the ROCm analog of CUPTI's
    // bufferCompleted logging.
    uint32_t n_kernel = 0, n_memcpy = 0, n_hipapi = 0, n_other = 0;

    for (size_t i = 0; i < num_headers; ++i) {
      auto* header = headers[i];
      if (!header || header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
      switch (header->kind) {
        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
          ++n_kernel;
          handle_kernel_dispatch(
              static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload));
          break;
        case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
          ++n_memcpy;
          handle_memory_copy(
              static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload));
          break;
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API:
          ++n_hipapi;
          handle_hip_api(
              static_cast<const rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload));
          break;
        default:
          ++n_other;
          break;
      }
    }

    size_t tracked_kernels = 0;
    {
      std::lock_guard<std::mutex> lock(g_kernel_syms_mu);
      tracked_kernels = g_kernel_syms.size();
    }
    if (g_writer) {
      g_writer->debug("rocm buffer_callback: headers=%zu kernel=%u memcpy=%u "
                      "hip_api=%u other=%u drop=%llu tracked_kernels=%zu",
                      num_headers, n_kernel, n_memcpy, n_hipapi, n_other,
                      static_cast<unsigned long long>(drop_count),
                      tracked_kernels);
    }
  } catch (...) {
    // Never let an exception cross the rocprofiler C callback boundary.
  }
}

// ---------------------------------------------------------------------------
// Code-object callback: builds the kernel_id -> symbol-name map
// ---------------------------------------------------------------------------

void code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t* /*user_data*/,
                          void* /*callback_data*/) {
  if (!g_running.load(std::memory_order_relaxed)) return;
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
  if (record.operation !=
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    return;

  try {
    // Populate on LOAD; NEVER erase on UNLOAD (late dispatch records must still
    // resolve). Keep the callback fast and never call back into HIP/HSA.
    if (record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    const auto* data = static_cast<
        const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
        record.payload);
    if (!data || !data->kernel_name) return;

    std::lock_guard<std::mutex> lock(g_kernel_syms_mu);
    if (g_kernel_syms.size() >= kMaxTrackedKernels &&
        g_kernel_syms.find(data->kernel_id) == g_kernel_syms.end()) {
      // Bounded with clear-on-overflow; entries self-heal on the next module
      // load. Extremely unlikely to hit in practice.
      g_kernel_syms.clear();
    }
    g_kernel_syms[data->kernel_id] = data->kernel_name;
  } catch (...) {
    // Never let an exception cross the rocprofiler C callback boundary.
  }
}

// ---------------------------------------------------------------------------
// Lifecycle (driven by rocprofiler-sdk tool_init / tool_fini)
// ---------------------------------------------------------------------------

// Writer flush hook: deliver buffered rocprofiler records to buffer_callback
// before each periodic write, so the serialized instruments are fresh. Guarded
// on g_running — the hook is never allowed to call into a finalizing SDK.
void flush_hook(void* /*arg*/) {
  if (!g_running.load(std::memory_order_relaxed)) return;
  ROCP_CALL(rocprofiler_flush_buffer(g_buffer_id));
}

int tool_init(rocprofiler_client_finalize_t /*fini*/, void* /*tool_data*/) {
  bool expected = false;
  if (!g_running.compare_exchange_strong(expected, true)) return 0;

  try {
    g_writer = graphsignal::MetricsWriter::init(
        "rocm", nullptr, 0,
        g_debug_enabled.load(std::memory_order_relaxed));
    if (!g_writer) {
      // Writer unavailable (e.g. shm dir not creatable): stay a safe no-op —
      // no services are configured, the workload runs unaffected.
      g_running.store(false, std::memory_order_relaxed);
      return 0;
    }

    g_kernel_trace_mode = kernel_trace_mode_from_env();
    if (!g_kernel_trace_env_rejected.empty()) {
      g_writer->debug("GRAPHSIGNAL_ROCM_KERNEL_TRACE=%s is not none|all; using none",
                      g_kernel_trace_env_rejected.c_str());
    }

    {
      std::lock_guard<std::mutex> lock(g_instruments_mu);
      g_kernels_profile = g_writer->register_instrument(
          graphsignal::InstrumentType::Profile, "rocm_kernels_nanoseconds", {});
      g_memcpy_profile = g_writer->register_instrument(
          graphsignal::InstrumentType::Profile, "rocm_memcpy_nanoseconds", {});
      g_sync_profile = g_writer->register_instrument(
          graphsignal::InstrumentType::Profile, "rocm_sync_nanoseconds", {});
      g_kernel_trace_mode_gauge = g_writer->register_instrument(
          graphsignal::InstrumentType::Gauge, "rocm_kernel_trace_mode", {});
    }
    graphsignal::MetricsWriter::set(g_kernel_trace_mode_gauge,
                                    static_cast<double>(g_kernel_trace_mode));

    resolve_hip_sync_op_names();

    ROCP_CALL(rocprofiler_create_context(&g_ctx_id));

    // LOSSLESS is REQUIRED for a flushed buffer. rocprofiler uses double
    // buffering: flush() toggles buffer_idx between the two internal buffers and
    // records are written to the newly-active one. rocprofiler_create_buffer only
    // allocates the SECOND internal buffer when the policy is LOSSLESS — with
    // DISCARD just one buffer is allocated, so the first flush toggles writes to
    // a zero-capacity buffer and every subsequent record is dropped ("buffer too
    // small (size=0)"). LOSSLESS applies backpressure (emplace waits for a flush)
    // only when the buffer is genuinely full; the generous size + watermark
    // auto-flush + the Writer's ~1s flush hook keep that path cold in practice.
    // This matches the rocprofiler-sdk buffered-tracing samples.
    ROCP_CALL(rocprofiler_create_buffer(
        g_ctx_id, kBufferSize, kBufferWatermark,
        ROCPROFILER_BUFFER_POLICY_LOSSLESS, buffer_callback, nullptr, &g_buffer_id));

    // The hook is armed only after the buffer exists, so the Writer thread can
    // never flush an unallocated buffer id.
    g_writer->set_flush_hook(&flush_hook, nullptr);

    if (g_kernel_trace_mode == kKernelTraceAll) {
      ROCP_CALL(rocprofiler_configure_buffer_tracing_service(
          g_ctx_id, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
          g_buffer_id));
    }
    ROCP_CALL(rocprofiler_configure_buffer_tracing_service(
        g_ctx_id, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0,
        g_buffer_id));

    // Restrict HIP_RUNTIME_API tracing to the sync-op subset so host-API
    // overhead stays bounded to sync points (analogous to CUPTI's narrow
    // SYNCHRONIZATION activity — NOT full runtime-API tracing).
    rocprofiler_tracing_operation_t hip_sync_ops[kNumHipSyncOps];
    for (size_t i = 0; i < kNumHipSyncOps; ++i) {
      hip_sync_ops[i] = static_cast<rocprofiler_tracing_operation_t>(kHipSyncOps[i]);
    }
    ROCP_CALL(rocprofiler_configure_buffer_tracing_service(
        g_ctx_id, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, hip_sync_ops,
        kNumHipSyncOps, g_buffer_id));

    // CODE_OBJECT is a synchronous callback service (runs on the app thread at
    // module load) — used only to build the kernel_id -> symbol-name map, so
    // it is only needed when kernel dispatches are traced.
    if (g_kernel_trace_mode == kKernelTraceAll) {
      ROCP_CALL(rocprofiler_configure_callback_tracing_service(
          g_ctx_id, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
          code_object_callback, nullptr));
    }

    // Dedicated delivery thread for the (asynchronous) buffer tracing services.
    ROCP_CALL(rocprofiler_create_callback_thread(&g_cb_thread));
    ROCP_CALL(rocprofiler_assign_callback_thread(g_buffer_id, g_cb_thread));

    ROCP_CALL(rocprofiler_start_context(g_ctx_id));

    g_writer->debug("rocm tool_init: writing %s (kernel_trace_mode=%s)",
                    g_writer->file_path().c_str(),
                    g_kernel_trace_mode == kKernelTraceAll ? "all" : "none");
  } catch (...) {
    // Best-effort: profiling stays off; the workload runs unaffected.
    g_running.store(false, std::memory_order_relaxed);
    return 0;
  }
  return 0;
}

void tool_fini(void* /*tool_data*/) {
  // rocprofiler-sdk OWNS the lifecycle and calls us here — there is no explicit
  // finalize we invoke (unlike CUPTI's cuptiFinalize + std::atexit path).
  if (!g_running.load(std::memory_order_relaxed)) return;

  try {
    // Deliver any remaining buffered records to buffer_callback while g_running
    // is still true. We do NOT call rocprofiler_stop_context here: tool_fini runs
    // during rocprofiler-sdk finalization, which already stops and tears down all
    // contexts — an explicit stop races that teardown and returns "Context ID not
    // found" (status 2). Instead we just flip g_running=false below; any late
    // buffer_callback then bails on its own (it checks g_running first).
    // This flush is the last SDK call; the callback it drives makes none (see
    // resolve_hip_sync_op_names) — the SDK is mid-finalization here.
    ROCP_CALL(rocprofiler_flush_buffer(g_buffer_id));

    if (g_writer) g_writer->debug("rocm tool_fini: done");

    // No further callback or rocprofiler work past this point (the Writer's
    // flush hook checks g_running and bails).
    g_running.store(false, std::memory_order_relaxed);

    // Final write (without the flush hook) + writer thread join. The Writer
    // object is deliberately leaked: no destructor work on teardown.
    if (g_writer) g_writer->shutdown();
  } catch (...) {
    g_running.store(false, std::memory_order_relaxed);
  }
}

uint32_t read_env_bool(const char* name) {
  const char* s = std::getenv(name);
  if (!s || !*s) return 0;
  if (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0) return 1;
  return 0;
}

uint32_t kernel_trace_mode_from_env() {
  const char* s = std::getenv("GRAPHSIGNAL_ROCM_KERNEL_TRACE");
  if (!s || !*s) return kKernelTraceNone;
  std::string v(s);
  // trim + lowercase, matching the CLI's normalization
  while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
  size_t start = 0;
  while (start < v.size() && (v[start] == ' ' || v[start] == '\t')) ++start;
  v = v.substr(start);
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (v == "all") return kKernelTraceAll;
  if (v != "none") g_kernel_trace_env_rejected = s;
  return kKernelTraceNone;
}

} // namespace

void rocm_activity_set_debug_mode(uint32_t debug_mode) {
  g_debug_enabled.store(debug_mode != 0, std::memory_order_relaxed);
  if (g_writer) g_writer->set_debug(debug_mode != 0);
}

uint32_t rocm_activity_get_debug_mode() {
  return g_debug_enabled.load(std::memory_order_relaxed) ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// rocprofiler-sdk tool registration entry point.
// rocprofiler-register scans loaded tool libraries (via ROCP_TOOL_LIBRARIES /
// LD_PRELOAD) for this symbol, so it MUST be extern "C" with default ELF
// visibility (the ROCm analog of CUPTI's InitializeInjection). If the build ever
// adds -fvisibility=hidden, annotate with __attribute__((visibility("default"))).
// ---------------------------------------------------------------------------

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/, const char* /*runtime_version*/,
                      uint32_t /*priority*/, rocprofiler_client_id_t* id) {
  if (id) id->name = "graphsignal";
  g_debug_enabled.store(read_env_bool("GRAPHSIGNAL_DEBUG") != 0,
                        std::memory_order_relaxed);
  static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini,
      nullptr};
  return &cfg;
}
