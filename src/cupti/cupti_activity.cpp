#include "cupti_activity.h"

#include <cupti.h>
#include <cuda.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "metrics_writer.h"

#ifndef CUPTI_CALL
#define CUPTI_CALL(call)                                                        \
  do {                                                                          \
    CUptiResult _status = (call);                                               \
    if (_status != CUPTI_SUCCESS && g_writer) {                                 \
      const char *errstr = nullptr;                                             \
      cuptiGetResultString(_status, &errstr);                                   \
      g_writer->error("CUPTI error %d (%s) at %s:%d",                           \
              _status, errstr ? errstr : "unknown", __FILE__, __LINE__);        \
    }                                                                           \
  } while (0)
#endif

namespace {

// Keep buffers reasonably sized so CUPTI delivers records frequently.
constexpr size_t kBufferSize = 128 * 1024;
constexpr size_t kBufferAlign = 8;

std::atomic<bool> g_running{false};

// Owns the cumulative instruments, the retained log ring, and the background
// thread that serializes everything to /dev/shm/graphsignal_<pid>/cupti.json.
// Deliberately leaked at stop (no destructor work on teardown).
static graphsignal::MetricsWriter* g_writer = nullptr;

// ---------------------------------------------------------------------------
// Instrument lookups. Instruments are registered lazily on the first record of
// a given kernel/graph/kind and cached here so the per-record path is a map or
// array lookup. All registration runs under g_instruments_mu.
// ---------------------------------------------------------------------------

static std::mutex g_instruments_mu;

// GPU activity aggregates into profile instruments: one metric per activity
// family, frames keyed by kernel symbol / graph signature / kind. Frame
// cardinality is bounded by the writer (kMaxProfileFrames).
static graphsignal::Instrument* g_kernels_profile = nullptr;
static graphsignal::Instrument* g_graphs_profile = nullptr;
static graphsignal::Instrument* g_memcpy_profile = nullptr;
static graphsignal::Instrument* g_memset_profile = nullptr;
static graphsignal::Instrument* g_sync_profile = nullptr;

// Reports the CUDA graph tracing granularity in effect (0 = graph, 1 = node).
// Without it an empty cuda_graphs_nanoseconds is ambiguous: node mode, or a
// workload that launches no graphs at all.
static graphsignal::Instrument* g_graph_trace_mode_gauge = nullptr;

// Graph tracing granularity for this process, set once by
// cupti_activity_start. Read on the record path only to decide nothing — the
// mode is expressed by which activity kinds are enabled — so a plain value is
// enough.
static uint32_t g_graph_trace_mode = GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;

// An unrecognized GRAPHSIGNAL_CUDA_GRAPH_TRACE value, kept until the writer's
// log ring exists (the variable is parsed in InitializeInjection, before the
// writer is created). Fixed size: the native lib keeps all memory bounded.
static char g_graph_trace_env_invalid[64] = {0};

// graphId -> frame name (16-hex signature hash, or graph_<id> fallback).
static std::unordered_map<uint32_t, std::string> g_graph_frames;

// Per-kind byte counters, indexed by the raw CUPTI enum value; values past
// the known range share slot 0, whose tag string the *KindToStr default
// covers. `registered` distinguishes "not yet seen" from "registration
// failed" so a failed registration isn't retried per record.
struct KindInstrument {
  graphsignal::Instrument* inst = nullptr;
  bool registered = false;
};
constexpr size_t kNumMemcpyKinds = 12;
constexpr size_t kNumMemsetKinds = 8;
static KindInstrument g_memcpy_bytes[kNumMemcpyKinds];
static KindInstrument g_memset_bytes[kNumMemsetKinds];

// CUPTI resource-callback subscriber (CUPTI_CB_DOMAIN_RESOURCE). Used to observe
// CUDA graph instantiation so we can compute a stable structural signature for
// each executable graph once, off the launch hot path.
static CUpti_SubscriberHandle g_subscriber{};

// graphId -> structural signature ("kernel[name=<symbol>,calls=n];" for kernel
// nodes, "<node_type>[calls=n];" otherwise; sorted, joined). A GRAPH_TRACE
// activity record carries only a graphId; this map lets the trace handler tag
// the instrument by structure (which aggregates across processes/hosts) instead
// of the process-local graphId. The signature
// is registered under BOTH the CUgraph id and the CUgraphExec id, since the id
// carried by GRAPH_TRACE differs across CUDA versions. Bounded with
// clear-on-overflow so a graph-churning workload can't grow it without bound.
constexpr size_t kMaxTrackedGraphs = 8192;
static std::unordered_map<uint32_t, std::string> g_graph_signatures;
static std::mutex g_graph_sig_mu;

// Bounded free-list of CUPTI activity buffers. 128 KiB sits at glibc's mmap
// threshold, so allocating each buffer fresh on the callback path means an
// mmap/munmap syscall pair (with page-zeroing) or arena-lock contention shared
// with the workload's own malloc; the free-list keeps a few buffers resident.
constexpr size_t kMaxPooledBuffers = 16;
static std::mutex g_buffer_pool_mu;
static std::vector<uint8_t*> g_buffer_pool;

static uint8_t* buffer_pool_acquire() {
  {
    std::lock_guard<std::mutex> g(g_buffer_pool_mu);
    if (!g_buffer_pool.empty()) {
      uint8_t* p = g_buffer_pool.back();
      g_buffer_pool.pop_back();
      return p;
    }
  }
  void* p = nullptr;
  if (posix_memalign(&p, kBufferAlign, kBufferSize) != 0 || !p) return nullptr;
  return static_cast<uint8_t*>(p);
}

static void buffer_pool_release(uint8_t* buf) {
  if (!buf) return;
  {
    std::lock_guard<std::mutex> g(g_buffer_pool_mu);
    if (g_buffer_pool.size() < kMaxPooledBuffers) {
      g_buffer_pool.push_back(buf);
      return;
    }
  }
  std::free(buf);
}

static void buffer_pool_drain() {
  std::lock_guard<std::mutex> g(g_buffer_pool_mu);
  for (uint8_t* p : g_buffer_pool) std::free(p);
  g_buffer_pool.clear();
}

// ---------------------------------------------------------------------------

static const char* syncTypeToStr(CUpti_ActivitySynchronizationType t) {
  switch (t) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE: return "context";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:  return "stream";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:   return "event";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:   return "stream_wait_event";
    default: return "unknown";
  }
}

static const char* memsetKindToStr(CUpti_ActivityMemoryKind k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:         return "device";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:         return "pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:       return "pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:        return "managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:          return "array";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:  return "device_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC: return "managed_static";
    default:                                        return "unknown";
  }
}

static const char* memcpyKindToStr(CUpti_ActivityMemcpyKind k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "host_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "device_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "device_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "host_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "peer_to_peer";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "host_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "array_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "array_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "array_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "device_to_array";
    default: return "other";
  }
}

// Short, stable token per CUDA graph node type — keeps the signature compact
// and human-skimmable.
static const char* graphNodeTypeToStr(CUgraphNodeType t) {
  switch (t) {
    case CU_GRAPH_NODE_TYPE_KERNEL:           return "kernel";
    case CU_GRAPH_NODE_TYPE_MEMCPY:           return "memcpy";
    case CU_GRAPH_NODE_TYPE_MEMSET:           return "memset";
    case CU_GRAPH_NODE_TYPE_HOST:             return "host";
    case CU_GRAPH_NODE_TYPE_GRAPH:            return "child";
    case CU_GRAPH_NODE_TYPE_EMPTY:            return "empty";
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT:       return "wait";
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:     return "event";
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: return "semsignal";
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT:   return "semwait";
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC:        return "memalloc";
    case CU_GRAPH_NODE_TYPE_MEM_FREE:         return "memfree";
    default:                                  return "other";
  }
}

// Build a structural signature for a CUDA graph: the multiset of node tuples,
// sorted and joined. Kernel nodes are keyed by their compiled symbol name
// ("kernel[name=<symbol>,calls=n];"); every other node type is keyed by type
// only ("memset[calls=n];"). The kernel symbol encodes the specialization
// (template params/tile sizes/dtype) but NOT the launch grid, so graphs that
// differ only by batch/token size (decode buckets) share one signature. The
// signature is order-independent (sorted) so it is stable across runs/processes
// for a structurally identical graph. Best-effort: returns "" on any failure.
static std::string compute_graph_signature(CUgraph graph) {
  if (!graph) return std::string();

  size_t num_nodes = 0;
  if (cuGraphGetNodes(graph, nullptr, &num_nodes) != CUDA_SUCCESS) return std::string();
  if (num_nodes == 0) return std::string();

  // Bound the work: a few thousand nodes is far more than any plausible decode
  // graph; cap to keep the one-time enumeration cheap and the string bounded.
  constexpr size_t kMaxNodes = 8192;
  if (num_nodes > kMaxNodes) num_nodes = kMaxNodes;

  std::vector<CUgraphNode> nodes(num_nodes);
  size_t got = num_nodes;
  if (cuGraphGetNodes(graph, nodes.data(), &got) != CUDA_SUCCESS) return std::string();
  if (got < num_nodes) num_nodes = got;

  // tuple key (everything up to the count) -> count. The "calls=" separator is
  // baked into the key so the emit loop can append the count uniformly.
  std::map<std::string, uint64_t> tuples;
  for (size_t i = 0; i < num_nodes; ++i) {
    CUgraphNodeType ntype;
    if (cuGraphNodeGetType(nodes[i], &ntype) != CUDA_SUCCESS) continue;

    std::string key = graphNodeTypeToStr(ntype);
    if (ntype == CU_GRAPH_NODE_TYPE_KERNEL) {
      // Identify the kernel by its compiled symbol name. cuFuncGetName returns
      // the mangled name (alphanumeric/underscore — no signature delimiters),
      // which is shape-independent: decode graphs that differ only by grid map
      // to the same name. Available since CUDA 12.3; the GRAPH_TRACE path that
      // calls this requires CUDA 12.4, so the symbol is always present at
      // runtime here (guarded for older build toolkits only).
      const char* kname = nullptr;
      CUDA_KERNEL_NODE_PARAMS kp;
      std::memset(&kp, 0, sizeof(kp));
      if (cuGraphKernelNodeGetParams(nodes[i], &kp) == CUDA_SUCCESS && kp.func) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12030
        if (cuFuncGetName(&kname, kp.func) != CUDA_SUCCESS) kname = nullptr;
#endif
      }
      key += "[name=";
      key += (kname ? kname : "");
      key += ",calls=";
    } else {
      key += "[calls=";
    }
    tuples[key] += 1;
  }

  if (tuples.empty()) return std::string();

  std::string sig;
  for (const auto& kv : tuples) {  // std::map iterates in sorted key order
    sig += kv.first;
    sig += std::to_string(kv.second);
    sig += "];";
  }
  return sig;
}

// Register a graph's signature under a graphId (CUgraph id and/or CUgraphExec
// id). Bounded with clear-on-overflow.
static void register_graph_signature(uint32_t graph_id, const std::string& sig) {
  if (sig.empty()) return;
  std::lock_guard<std::mutex> lock(g_graph_sig_mu);
  if (g_graph_signatures.size() >= kMaxTrackedGraphs &&
      g_graph_signatures.find(graph_id) == g_graph_signatures.end()) {
    g_graph_signatures.clear();
  }
  g_graph_signatures[graph_id] = sig;
}

// Returns the signature for a graphId, or "" if none is known yet.
static std::string lookup_graph_signature(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(g_graph_sig_mu);
  auto it = g_graph_signatures.find(graph_id);
  return it == g_graph_signatures.end() ? std::string() : it->second;
}

// 16-hex lowercase FNV-1a64 of a string — the compact, cross-process-stable
// tag value for a graph's structural signature.
static void fnv1a64_hex(const std::string& s, char out[17]) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  std::snprintf(out, 17, "%016llx", static_cast<unsigned long long>(h));
}

// ---------------------------------------------------------------------------
// Lazy instrument registration
// ---------------------------------------------------------------------------

static graphsignal::Instrument* kind_instrument(KindInstrument& slot, graphsignal::InstrumentType type,
                                       const char* name,
                                       const char* tag_key, const char* tag_value) {
  std::lock_guard<std::mutex> g(g_instruments_mu);
  if (!slot.registered) {
    slot.registered = true;
    if (g_writer) {
      slot.inst = g_writer->register_instrument(type, name,
                                                {{tag_key, tag_value}});
    }
  }
  return slot.inst;
}

// Frame name for a graph: the structural signature (computed once at
// instantiation via the resource callback), hashed to a compact token so
// structurally identical graphs aggregate across processes/hosts. Falls back
// to the process-local graphId when no signature is known (e.g. the resource
// callback didn't fire, or graphId semantics didn't match).
static const std::string& graph_frame_name(uint32_t graph_id) {
  {
    std::lock_guard<std::mutex> g(g_instruments_mu);
    auto it = g_graph_frames.find(graph_id);
    if (it != g_graph_frames.end()) return it->second;
  }

  // Computed outside g_instruments_mu — lookup_graph_signature takes its own
  // lock.
  const std::string sig = lookup_graph_signature(graph_id);
  std::string frame;
  if (!sig.empty()) {
    char hex[17];
    fnv1a64_hex(sig, hex);
    frame = hex;
  } else {
    frame = "graph_" + std::to_string(graph_id);
  }

  std::lock_guard<std::mutex> g(g_instruments_mu);
  auto it = g_graph_frames.find(graph_id);
  if (it != g_graph_frames.end()) return it->second;
  if (g_graph_frames.size() >= kMaxTrackedGraphs) g_graph_frames.clear();
  if (g_writer) {
    g_writer->debug("graph %s signature: %s", frame.c_str(), sig.c_str());
  }
  return g_graph_frames.emplace(graph_id, std::move(frame)).first->second;
}

// ---------------------------------------------------------------------------
// Writer hooks
// ---------------------------------------------------------------------------

// Called by the writer thread before each periodic serialize so buffered CUPTI
// activity reaches the instruments first. The Writer never calls it on the
// final shutdown write, and the g_running guard keeps it from touching CUPTI
// while stop is tearing profiling down.
static void writer_flush_hook(void*) {
  if (!g_running.load(std::memory_order_relaxed)) return;
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}

// Copies DEVICE-storage probe blocks (probe.h DEVICE storage) to host memory
// via the driver API so user CUDA probes can be serialized. Per device, every
// copy is issued back to back into a pinned staging buffer on a private
// stream and waited for once (a copy into pageable memory would block on its
// own completion, turning N probes into N round trips). Safety rules, each one
// a way this path could otherwise hurt the workload:
//  - The probe's own device's primary context is used, and only when the
//    workload has already made it active: retaining an inactive one would
//    create a context (hundreds of MB) on a GPU the process never used.
//  - The copies run on a private NON-BLOCKING stream: copies on the legacy
//    default stream would wait for, and be waited on by, the workload's
//    blocking-stream kernels.
//  - The calls run in relaxed capture mode: the workload may be capturing a
//    CUDA graph in global mode, where a potentially unsafe call from any
//    other thread invalidates the capture and fails the engine's startup.
//  - Nothing is ever freed (freeing pinned memory can synchronize the
//    device): the context stays retained until process exit, and an outgrown
//    staging buffer is kept — capacity doubles, so the kept buffers total less
//    than the current one, and the registry cap bounds that at a few MB.
// Every driver call is checked; a failed copy skips its probe for this write,
// a failed wait skips the device's probes. Never touches CUDA during/after
// teardown (g_running).
static constexpr int kMaxProbeDevices = 64;
struct ProbeDevice {
  CUcontext ctx = nullptr;
  CUstream stream = nullptr;
  graphsignal_instrument_data* staging = nullptr;  // pinned host memory
  size_t staging_cap = 0;                          // in blocks
};
static std::mutex g_probe_mu;
static ProbeDevice g_probe_devices[kMaxProbeDevices];

// Switches this thread to relaxed capture mode for its lifetime.
struct RelaxedCaptureMode {
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
  bool active = cuThreadExchangeStreamCaptureMode(&mode) == CUDA_SUCCESS;
  ~RelaxedCaptureMode() {
    if (active) cuThreadExchangeStreamCaptureMode(&mode);  // restore
  }
};

// Reads the probes entries[idx[0..m)] of one device into out/ok. Caller holds
// g_probe_mu and has switched this thread to relaxed capture mode.
static void read_device_probes_on(int device_id, const graphsignal_probe_entry* const* entries,
                                  const std::vector<size_t>& idx,
                                  graphsignal_instrument_data* out, uint8_t* ok) {
  std::vector<uint8_t> queued(idx.size(), 0);  // allocated before any context push
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, device_id) != CUDA_SUCCESS) return;
  unsigned int flags = 0;
  int active = 0;
  if (cuDevicePrimaryCtxGetState(dev, &flags, &active) != CUDA_SUCCESS || !active) return;
  ProbeDevice& pd = g_probe_devices[device_id];
  if (!pd.ctx) {
    CUcontext ctx = nullptr;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return;
    pd.ctx = ctx;
  }
  if (cuCtxPushCurrent(pd.ctx) != CUDA_SUCCESS) return;
  if (!pd.stream && cuStreamCreate(&pd.stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
    pd.stream = nullptr;
  }
  if (pd.stream && pd.staging_cap < idx.size()) {
    size_t cap = pd.staging_cap ? pd.staging_cap : 1;
    while (cap < idx.size()) cap *= 2;
    void* p = nullptr;
    if (cuMemAllocHost(&p, cap * sizeof(graphsignal_instrument_data)) == CUDA_SUCCESS) {
      pd.staging = static_cast<graphsignal_instrument_data*>(p);
      pd.staging_cap = cap;
    }
  }
  if (pd.stream && pd.staging_cap >= idx.size()) {
    for (size_t k = 0; k < idx.size(); k++) {
      queued[k] = cuMemcpyDtoHAsync(&pd.staging[k], (CUdeviceptr)(uintptr_t)entries[idx[k]]->data,
                                    sizeof(graphsignal_instrument_data), pd.stream) == CUDA_SUCCESS;
    }
    if (cuStreamSynchronize(pd.stream) == CUDA_SUCCESS) {
      for (size_t k = 0; k < idx.size(); k++) {
        if (!queued[k]) continue;
        std::memcpy(&out[idx[k]], &pd.staging[k], sizeof(graphsignal_instrument_data));
        ok[idx[k]] = 1;
      }
    }
  }
  cuCtxPopCurrent(nullptr);
}

static void device_probe_reader(const graphsignal_probe_entry* const* entries, size_t n,
                                graphsignal_instrument_data* out, uint8_t* ok, void*) {
  if (!g_running.load(std::memory_order_relaxed)) return;
  try {
    if (!entries || !out || !ok || n == 0) return;
    std::lock_guard<std::mutex> g(g_probe_mu);
    RelaxedCaptureMode relaxed;
    if (!relaxed.active) return;
    std::vector<size_t> idx;
    uint64_t done = 0;  // devices already read (bit per ordinal)
    for (size_t i = 0; i < n; i++) {
      const int d = entries[i]->device_id;
      if (d < 0 || d >= kMaxProbeDevices || (done >> d) & 1ull) continue;
      done |= 1ull << d;
      idx.clear();
      for (size_t k = i; k < n; k++) {
        if (entries[k]->device_id == d) idx.push_back(k);
      }
      read_device_probes_on(d, entries, idx, out, ok);
    }
  } catch (...) {
  }
}

// ---------------------------------------------------------------------------
// CUPTI callbacks
// ---------------------------------------------------------------------------

// CUPTI resource callback. Observes CUDA graph instantiation so we can compute
// each graph's structural signature once, off the launch hot path, and key it
// by the ids that GRAPH_TRACE may use.
static void CUPTIAPI graph_resource_callback(void* /*userdata*/, CUpti_CallbackDomain domain,
                                             CUpti_CallbackId cbid, const CUpti_CallbackData* cbdata) {
  if (!g_running.load(std::memory_order_relaxed)) return;
  if (domain != CUPTI_CB_DOMAIN_RESOURCE) return;

  try {
    const auto* rd = reinterpret_cast<const CUpti_ResourceData*>(cbdata);
    if (!rd || !rd->resourceDescriptor) return;

    switch (cbid) {
      case CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED: {
        const auto* gd = reinterpret_cast<const CUpti_GraphData*>(rd->resourceDescriptor);
        const std::string sig = compute_graph_signature(gd->graph);
        if (sig.empty()) break;
        // Register under both the source graph id and the executable graph id;
        // GRAPH_TRACE's graphId differs across CUDA versions.
        uint32_t id = 0;
        if (gd->graph && cuptiGetGraphId(gd->graph, &id) == CUPTI_SUCCESS) {
          register_graph_signature(id, sig);
        }
        if (gd->graphExec && cuptiGetGraphExecId(gd->graphExec, &id) == CUPTI_SUCCESS) {
          register_graph_signature(id, sig);
        }
        break;
      }
      default:
        break;
    }
  } catch (...) {
    // Never let an exception cross the CUPTI C callback boundary.
  }
}

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *buffer = nullptr;
  *size = 0;
  *maxNumRecords = 0;

  // After stop / during teardown, hand CUPTI nothing.
  if (!g_running.load(std::memory_order_relaxed)) return;

  try {
    if (g_writer) g_writer->debug("cupti bufferRequested: size=%zu", kBufferSize);

    uint8_t* p = buffer_pool_acquire();
    if (!p) {
      if (g_writer) g_writer->error("cupti bufferRequested: alloc FAILED");
      return;
    }

    *buffer = p;
    *size = kBufferSize;
  } catch (...) {
    // Never let an exception cross the CUPTI C callback boundary.
    *buffer = nullptr;
    *size = 0;
  }
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize) {
  // Late callback after stop / during teardown: globals (the instrument maps)
  // may be mid-destruction. Release the buffer and bail without touching any
  // shared state. g_running is a static atomic, safe to read post-destruction.
  if (!g_running.load(std::memory_order_relaxed)) {
    std::free(buffer);
    return;
  }

  try {
  if (validSize > 0) {
    CUpti_Activity* record = nullptr;
    CUptiResult status;
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS && record) {
        switch (record->kind) {
          case CUPTI_ACTIVITY_KIND_KERNEL:
          case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel4*>(record);
            if (k->end <= k->start) break;
            graphsignal::MetricsWriter::profile_add(
                g_kernels_profile, k->name ? k->name : "", k->end - k->start);
            break;
          }
          case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: {
            // With GRAPH_TRACE enabled, kernels replayed via a CUDA graph are
            // not reported individually; the whole graph launch arrives as one
            // record. Use the base CUpti_ActivityGraphTrace struct — its
            // streamId/graphId/start/end prefix is layout-stable across the
            // GraphTrace/GraphTrace2 record revisions, same pattern as the
            // kernel/sync casts.
            const auto* g = reinterpret_cast<const CUpti_ActivityGraphTrace*>(record);
            if (g->start == 0 || g->end == 0 || g->end <= g->start) break;
            graphsignal::MetricsWriter::profile_add(
                g_graphs_profile, graph_frame_name(g->graphId).c_str(),
                g->end - g->start);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY: {
            const auto* m = reinterpret_cast<const CUpti_ActivityMemcpy*>(record);
            if (m->end <= m->start) break;
            const size_t kind = m->copyKind < kNumMemcpyKinds ? m->copyKind : 0;
            const char* kind_str =
                memcpyKindToStr(static_cast<CUpti_ActivityMemcpyKind>(m->copyKind));
            graphsignal::MetricsWriter::profile_add(
                g_memcpy_profile, kind_str, m->end - m->start);
            graphsignal::MetricsWriter::add(
                kind_instrument(g_memcpy_bytes[kind], graphsignal::InstrumentType::Counter,
                                "cuda_memcpy_bytes", "kind", kind_str),
                m->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMSET: {
            const auto* m = reinterpret_cast<const CUpti_ActivityMemset*>(record);
            if (m->end <= m->start) break;
            const size_t kind = m->memoryKind < kNumMemsetKinds ? m->memoryKind : 0;
            const char* kind_str =
                memsetKindToStr(static_cast<CUpti_ActivityMemoryKind>(m->memoryKind));
            graphsignal::MetricsWriter::profile_add(
                g_memset_profile, kind_str, m->end - m->start);
            graphsignal::MetricsWriter::add(
                kind_instrument(g_memset_bytes[kind], graphsignal::InstrumentType::Counter,
                                "cuda_memset_bytes", "kind", kind_str),
                m->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
            // Always use the base struct: it's CUPTI's layout-compatible prefix
            // (carries type/streamId/cudaEventId/start/end) and is defined across
            // CUDA 12 and all 13.x. CUpti_ActivitySynchronization2 is NOT present
            // in every CUDA 13.x cupti.h, so a CUDA_MAJOR-based cast both breaks
            // the build on those toolchains and, when cross-built against a
            // version that has it, produces an ABI mismatch with the runtime.
            const auto* s = reinterpret_cast<const CUpti_ActivitySynchronization*>(record);
            if (s->start == 0 && s->end == 0) break;  // timestamps unavailable
            if (s->end <= s->start) break;
            graphsignal::MetricsWriter::profile_add(
                g_sync_profile, syncTypeToStr(s->type), s->end - s->start);
            break;
          }
          default:
            break;
        }
      } else if (status != CUPTI_SUCCESS && status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
        const char* errstr = nullptr;
        cuptiGetResultString(status, &errstr);
        if (g_writer) {
          g_writer->error("cuptiActivityGetNextRecord error: %s",
                          errstr ? errstr : "unknown");
        }
      }
    } while (status == CUPTI_SUCCESS);
  }

  // Count any dropped records since the previous call (for the debug log only).
  // Use the callback-provided (context, streamId) to be compatible across CUDA
  // 12/13.
  size_t dropped = 0;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));

  if (g_writer) {
    g_writer->debug("cupti bufferCompleted: size=%zu validSize=%zu dropped=%zu",
                    size, validSize, dropped);
  }
  } catch (...) {
    // Never let an exception cross the CUPTI C callback boundary.
  }

  buffer_pool_release(buffer);
}

} // namespace

int cupti_activity_start(uint64_t write_interval_ns, uint32_t debug_mode,
                         uint32_t graph_trace_mode) {
  bool expected = false;
  if (!g_running.compare_exchange_strong(expected, true)) return 1;

  // Anything unexpected degrades to the default granularity rather than
  // leaving collection in an undefined state.
  g_graph_trace_mode = (graph_trace_mode == GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE)
                           ? GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE
                           : GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;

  // The Writer owns the instruments, the log ring, the process context, and
  // the serialize thread. init never throws; nullptr means profiling stays
  // off — the workload runs unaffected.
  graphsignal::MetricsWriter* writer =
      graphsignal::MetricsWriter::init("cupti", nullptr, write_interval_ns, debug_mode != 0);
  if (!writer) {
    std::fprintf(stderr,
                 "graphsignal: cupti writer init failed; profiling disabled\n");
    g_running.store(false, std::memory_order_relaxed);
    return 0;
  }
  g_writer = writer;

  // Fresh run — instrument pointers belong to the current Writer only.
  {
    std::lock_guard<std::mutex> g(g_instruments_mu);
    g_graph_frames.clear();
    for (auto& s : g_memcpy_bytes) s = KindInstrument{};
    for (auto& s : g_memset_bytes) s = KindInstrument{};
    g_kernels_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_kernels_nanoseconds", {});
    g_graphs_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_graphs_nanoseconds", {});
    g_memcpy_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_memcpy_nanoseconds", {});
    g_memset_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_memset_nanoseconds", {});
    g_sync_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_sync_nanoseconds", {});
    g_graph_trace_mode_gauge = writer->register_instrument(
        graphsignal::InstrumentType::Gauge, "cuda_graph_trace_mode", {});
  }
  graphsignal::MetricsWriter::set(g_graph_trace_mode_gauge,
                                  static_cast<double>(g_graph_trace_mode));
  {
    std::lock_guard<std::mutex> g(g_graph_sig_mu);
    g_graph_signatures.clear();
  }

  g_writer->set_flush_hook(writer_flush_hook, nullptr);
  g_writer->set_device_probe_reader(device_probe_reader, nullptr);

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));

  // Trace CUDA graphs at GRAPH granularity rather than per node. When
  // GRAPH_TRACE is enabled CUPTI stops instrumenting the individual nodes of a
  // launched graph (the graph is reported as a single CUPTI_ACTIVITY_KIND_-
  // GRAPH_TRACE record instead). Lower overhead — same approach Nsight Systems
  // uses with --cuda-graph-trace=graph. Available since CUDA 12.4; if an older
  // CUPTI rejects it, CUPTI_CALL just logs and graph launches go unrecorded.
  //
  // In NODE mode this enable is deliberately skipped: with GRAPH_TRACE off,
  // CUPTI keeps instrumenting the graph's nodes and their kernels arrive
  // through the CONCURRENT_KERNEL kind enabled above, landing in
  // cuda_kernels_nanoseconds by symbol like eager launches. Nothing else
  // changes, so an older CUPTI that rejects a kind cannot break the rest of
  // collection either way.
  if (g_graph_trace_mode == GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE));
  }

  // Observe CUDA graph instantiation so we can compute a stable structural
  // signature per executable graph (used to tag GRAPH_TRACE instruments across
  // processes/hosts). Resource callbacks are passive metadata notifications —
  // they don't instrument the launch path, so they don't reintroduce the
  // per-node graph instrumentation that GRAPH_TRACE avoids.
  if (cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)graph_resource_callback,
                     nullptr) == CUPTI_SUCCESS) {
    CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
  }

  std::atexit([]() { cupti_activity_stop(); });

  if (g_graph_trace_env_invalid[0]) {
    g_writer->debug(
        "GRAPHSIGNAL_CUDA_GRAPH_TRACE=%s is not graph|node; using graph",
        g_graph_trace_env_invalid);
  }
  g_writer->debug(
      "cupti_activity_start: write_interval_ns=%llu debug_mode=%u "
      "graph_trace_mode=%s",
      static_cast<unsigned long long>(write_interval_ns), debug_mode,
      g_graph_trace_mode == GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE ? "node" : "graph");
  return 1;
}

void cupti_activity_stop(void) {
  bool expected = true;
  if (!g_running.compare_exchange_strong(expected, false)) return;

  // Stop receiving resource callbacks before tearing CUPTI down. cuptiFinalize()
  // below also unregisters everything, but unsubscribing first is the explicit
  // counterpart to cuptiSubscribe in start.
  if (g_subscriber) {
    cuptiUnsubscribe(g_subscriber);
    g_subscriber = nullptr;
  }

  // Detach CUPTI before our memory can go away. The CUDA_INJECTION64_PATH lib
  // can be unloaded while the process keeps running CUDA; if CUPTI still holds
  // our bufferRequested/bufferCompleted pointers, its next callback jumps into
  // the unmapped library -> SIGSEGV (observed in libcupti during cuLaunchKernel
  // on a forward-compat driver). cuptiFinalize() unregisters all callbacks and
  // disables activities. g_running is already false, so any flush callback it
  // triggers bails immediately (the buffer is freed, no shared state touched).
  cuptiFinalize();

  // Final serialize + writer-thread join. The final write skips the flush hook
  // and the device probe reader refuses because g_running is false — no CUDA
  // on the teardown path. The Writer object is deliberately leaked: no
  // destructor work on teardown.
  if (g_writer) {
    g_writer->debug("cupti_activity_stop");
    g_writer->shutdown();
  }

  // Safe now: cuptiFinalize() has unregistered the callbacks, so nothing can
  // call buffer_pool_acquire/release afterward.
  buffer_pool_drain();
}

void cupti_activity_set_debug_mode(uint32_t enabled) {
  if (g_writer) g_writer->set_debug(enabled != 0);
}

uint32_t cupti_activity_get_debug_mode(void) {
  return (g_writer && g_writer->debug_enabled()) ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// CUDA injection entry point
// Called by the CUDA driver when CUDA_INJECTION64_PATH points to this library.
// ---------------------------------------------------------------------------

static std::atomic<bool> g_injection_active{false};

static uint32_t read_env_bool(const char* name) {
  const char* s = std::getenv(name);
  if (!s || !*s) return 0;
  if (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0) return 1;
  return 0;
}

// Returns the variable's value, or `def` when unset or empty. The pointer is
// getenv's — read it, never keep it.
static const char* read_env_string(const char* name, const char* def) {
  const char* s = std::getenv(name);
  if (!s || !*s) return def;
  return s;
}

extern "C" {

uint32_t cupti_activity_graph_trace_mode_from_env(void) {
  g_graph_trace_env_invalid[0] = '\0';
  const char* s = read_env_string("GRAPHSIGNAL_CUDA_GRAPH_TRACE", nullptr);
  if (!s) return GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;
  if (std::strcmp(s, "node") == 0) return GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE;
  if (std::strcmp(s, "graph") == 0) return GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;
  // Anything else: fall back to the default and keep the offending value for
  // the debug note cupti_activity_start emits once the writer exists.
  std::snprintf(g_graph_trace_env_invalid, sizeof(g_graph_trace_env_invalid),
                "%s", s);
  return GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;
}

CUptiResult InitializeInjection() {
  bool expected = false;
  if (!g_injection_active.compare_exchange_strong(expected, true,
                                                   std::memory_order_acq_rel))
    return CUPTI_SUCCESS;

  const uint32_t debug_mode = read_env_bool("GRAPHSIGNAL_DEBUG");
  const uint32_t graph_trace_mode = cupti_activity_graph_trace_mode_from_env();

  // Never let an exception (e.g. std::thread creation, bad_alloc) escape into
  // the CUDA driver's init path — that would std::terminate the workload.
  try {
    cupti_activity_start(/*write_interval_ns=*/0, debug_mode, graph_trace_mode);
  } catch (...) {
    // Best-effort: profiling stays off; the workload runs unaffected.
  }

  return CUPTI_SUCCESS;
}

} // extern "C"
