#include "../../cupti/cupti_activity.h"

#include <graphsignal/probe.h>
#include <graphsignal/probe_cuda.h>

#include <cuda_runtime.h>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Simple test framework
#define ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
      std::abort(); \
    } \
  } while (0)

#define ASSERT_EQ(a, b, msg) \
  do { \
    if ((a) != (b)) { \
      std::fprintf(stderr, "FAIL: %s:%d: %s (expected %llu, got %llu)\n", \
                   __FILE__, __LINE__, msg, \
                   static_cast<unsigned long long>(b), \
                   static_cast<unsigned long long>(a)); \
      std::abort(); \
    } \
  } while (0)

// ---------------------------------------------------------------------------
// Shm JSON helpers. The writer serializes full cumulative state to a single
// file, /dev/shm/graphsignal_<pid>/cupti.json, atomically on every interval.
// ---------------------------------------------------------------------------

static std::string shm_json_path() {
  char path[128];
  std::snprintf(path, sizeof(path), "/dev/shm/graphsignal_%d/cupti.json",
                static_cast<int>(getpid()));
  return std::string(path);
}

static std::string read_file(const std::string& path) {
  std::string content;
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return content;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz > 0) {
    content.resize(static_cast<size_t>(sz));
    size_t nread = std::fread(&content[0], 1, static_cast<size_t>(sz), f);
    content.resize(nread);
  }
  std::fclose(f);
  return content;
}

// Every metric object in the "metrics" array whose "name" equals `name`.
// Balanced-brace scan from the '{' that opens the object (metric objects nest
// only the "tags" object; the serialized strings carry no braces).
static std::vector<std::string> find_metric_blocks(const std::string& json,
                                                   const std::string& name) {
  std::vector<std::string> blocks;
  const std::string needle = "\"name\":\"" + name + "\"";
  size_t pos = json.find(needle);
  while (pos != std::string::npos) {
    size_t start = json.rfind('{', pos);
    if (start == std::string::npos) break;
    int depth = 0;
    size_t end = std::string::npos;
    for (size_t i = start; i < json.size(); ++i) {
      if (json[i] == '{') {
        depth++;
      } else if (json[i] == '}') {
        depth--;
        if (depth == 0) { end = i; break; }
      }
    }
    if (end == std::string::npos) break;
    blocks.push_back(json.substr(start, end - start + 1));
    pos = json.find(needle, pos + needle.size());
  }
  return blocks;
}

// First metric block with the given name carrying tag_key=tag_value; "" if none.
static std::string find_metric_with_tag(const std::string& json, const std::string& name,
                                        const std::string& tag_key,
                                        const std::string& tag_value) {
  const std::string tag = "\"" + tag_key + "\":\"" + tag_value + "\"";
  for (const auto& block : find_metric_blocks(json, name)) {
    if (block.find(tag) != std::string::npos) return block;
  }
  return std::string();
}

static uint64_t block_u64(const std::string& block, const char* field) {
  const std::string m = std::string("\"") + field + "\":";
  size_t p = block.find(m);
  if (p == std::string::npos) return 0;
  return std::strtoull(block.c_str() + p + m.size(), nullptr, 10);
}

// Extracts the "frames":{...} object of a profile metric block; "" if absent.
static std::string frames_object(const std::string& block) {
  const std::string m = "\"frames\":{";
  size_t p = block.find(m);
  if (p == std::string::npos) return std::string();
  size_t start = p + m.size() - 1;  // at '{'
  int depth = 0;
  bool in_str = false;
  for (size_t i = start; i < block.size(); ++i) {
    char c = block[i];
    if (in_str) {
      if (c == '\\') { i++; continue; }
      if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') {
      in_str = true;
    } else if (c == '{') {
      depth++;
    } else if (c == '}') {
      depth--;
      if (depth == 0) return block.substr(start, i - start + 1);
    }
  }
  return std::string();
}

struct Frame {
  std::string name;
  uint64_t value;
  uint64_t samples;
};

// Parses every "name":[value,samples] entry of a profile block's frames
// object, unescaping \" and \\ in frame names.
static std::vector<Frame> parse_frames(const std::string& block) {
  std::vector<Frame> out;
  const std::string fo = frames_object(block);
  size_t i = 1;  // skip '{'
  while (i < fo.size()) {
    if (fo[i] != '"') { ++i; continue; }
    ++i;
    std::string name;
    while (i < fo.size() && fo[i] != '"') {
      if (fo[i] == '\\' && i + 1 < fo.size()) {
        name += fo[i + 1];
        i += 2;
        continue;
      }
      name += fo[i];
      ++i;
    }
    ++i;  // closing quote
    if (i + 1 < fo.size() && fo[i] == ':' && fo[i + 1] == '[') {
      i += 2;
      char* after = nullptr;
      uint64_t value = std::strtoull(fo.c_str() + i, &after, 10);
      uint64_t samples = 0;
      if (after && *after == ',') {
        samples = std::strtoull(after + 1, nullptr, 10);
      }
      out.push_back({std::move(name), value, samples});
    }
  }
  return out;
}

// Value of the frame with the exact name; 0 when absent.
static uint64_t frame_value(const std::string& block, const std::string& name) {
  for (const Frame& f : parse_frames(block)) {
    if (f.name == name) return f.value;
  }
  return 0;
}

// Value of the first frame whose name contains `sub`; 0 when absent. Kernel
// frames carry raw mangled symbols, so lookups match by substring.
static uint64_t frame_value_containing(const std::string& block,
                                       const std::string& sub) {
  for (const Frame& f : parse_frames(block)) {
    if (f.name.find(sub) != std::string::npos) return f.value;
  }
  return 0;
}

// Sample count of the first frame whose name contains `sub`; 0 when absent.
static uint64_t frame_samples_containing(const std::string& block,
                                         const std::string& sub) {
  for (const Frame& f : parse_frames(block)) {
    if (f.name.find(sub) != std::string::npos) return f.samples;
  }
  return 0;
}

// True when some frame's name contains `sub`.
static bool has_frame_containing(const std::string& block,
                                 const std::string& sub) {
  for (const Frame& f : parse_frames(block)) {
    if (f.name.find(sub) != std::string::npos) return true;
  }
  return false;
}

// Value of a gauge metric block ("value":<double>); NaN-free double parse.
static double block_double(const std::string& block, const char* field) {
  const std::string m = std::string("\"") + field + "\":";
  size_t p = block.find(m);
  if (p == std::string::npos) return -1.0;
  return std::strtod(block.c_str() + p + m.size(), nullptr);
}

// Sums the sparse "counts":[...] array of a histogram block.
static uint64_t counts_total(const std::string& block) {
  const std::string m = "\"counts\":[";
  size_t p = block.find(m);
  if (p == std::string::npos) return 0;
  p += m.size();
  uint64_t total = 0;
  while (p < block.size() && block[p] != ']') {
    total += std::strtoull(block.c_str() + p, nullptr, 10);
    while (p < block.size() && block[p] != ',' && block[p] != ']') ++p;
    if (p < block.size() && block[p] == ',') ++p;
  }
  return total;
}

// Histogram sanity: sparse bins/counts arrays plus the exact aggregates
// (count = the sum of the counts, sum, min, max) whenever a value was observed.
static void assert_histogram_sane(const std::string& block, uint64_t min_count,
                                  const char* what) {
  ASSERT(!block.empty(), what);
  ASSERT(block.find("\"type\":\"histogram\"") != std::string::npos, what);
  ASSERT(block.find("\"bins\":[") != std::string::npos, what);
  const uint64_t total = counts_total(block);
  std::printf("%s: counts_total=%llu\n", what,
              static_cast<unsigned long long>(total));
  ASSERT(total >= min_count, what);
  if (total > 0) {
    char expect[64];
    std::snprintf(expect, sizeof(expect), "\"count\":%llu,", static_cast<unsigned long long>(total));
    ASSERT(block.find(expect) != std::string::npos,
           "histogram count must equal the sum of the bin counts");
    ASSERT(block.find("\"sum\":") != std::string::npos, "histogram must serialize sum");
    ASSERT(block.find("\"min\":") != std::string::npos, "histogram must serialize min");
    ASSERT(block.find("\"max\":") != std::string::npos, "histogram must serialize max");
  }
}

// ---------------------------------------------------------------------------
// Test kernels — busy waits spin on clock64() so durations are controllable.
// ---------------------------------------------------------------------------

__global__ void test_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * 2.0f + 1.0f;
  }
}

__global__ void busy_wait_kernel_A(int* sink, unsigned long long target_cycles) {
  unsigned long long start = clock64();
  while (clock64() - start < target_cycles) { /* spin */ }
  if (threadIdx.x == 0 && blockIdx.x == 0) *sink = 1;
}

__global__ void busy_wait_kernel_B(int* sink, unsigned long long target_cycles) {
  unsigned long long start = clock64();
  while (clock64() - start < target_cycles) { /* spin */ }
  if (threadIdx.x == 0 && blockIdx.x == 0) *sink = 2;
}

// Records a busy-wait duration into a DEVICE-storage probe instrument; the
// injection lib's device probe reader copies the block out via the driver API.
__global__ void probe_record_kernel(graphsignal_instrument_data* inst,
                                    unsigned long long target_cycles) {
  unsigned long long t0 = graphsignal_gtimer();
  unsigned long long start = clock64();
  while (clock64() - start < target_cycles) { /* spin */ }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    graphsignal_device_record(inst, graphsignal_gtimer() - t0);
  }
}

// ---------------------------------------------------------------------------
// CUDA graph tracing granularity (GRAPHSIGNAL_CUDA_GRAPH_TRACE=graph|node).
//
// Both cases run in a re-exec'd child of this binary: the mode is read from the
// environment before the library initializes and cannot be changed afterwards,
// and each child needs its own CUPTI session and its own
// /dev/shm/graphsignal_<pid>/cupti.json. fork() alone would inherit a
// half-initialized CUDA context, so the child execs /proc/self/exe with the
// scenario marker — same isolation intent as the probe test's forked cap case.
// ---------------------------------------------------------------------------

static const char* kGraphTraceMarker = "--graph-trace-scenario";

// Kernel A once and kernel B once, captured into a graph, replayed
// kGraphTraceReplays times, plus a single eager launch of A afterwards. The
// eager launch is what separates the modes: in graph mode it is the only thing
// cuda_kernels_nanoseconds may know about.
static const int kGraphTraceReplays = 5;

static int graph_trace_scenario() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("graph-trace child: CUDA not available, skipping\n");
    return 0;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "graph-trace child: cudaSetDevice failed\n");
    return 1;
  }

  // The env var is parsed by the library itself, so this exercises the real
  // parse path (including the fallback for an unrecognized value).
  const uint32_t mode = cupti_activity_graph_trace_mode_from_env();
  const bool node_mode = (mode == GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE);
  std::printf("graph-trace child: GRAPHSIGNAL_CUDA_GRAPH_TRACE=%s -> mode=%s\n",
              std::getenv("GRAPHSIGNAL_CUDA_GRAPH_TRACE")
                  ? std::getenv("GRAPHSIGNAL_CUDA_GRAPH_TRACE")
                  : "<unset>",
              node_mode ? "node" : "graph");

  const uint64_t write_interval_ns = 100'000'000ULL;  // 100ms
  ASSERT(cupti_activity_start(write_interval_ns, /*debug_mode=*/1, mode) == 1,
         "cupti_activity_start failed in graph-trace child");

  const unsigned long long kBusyCycles = 2'000'000ULL;
  int* d_sink = nullptr;
  ASSERT(cudaMalloc(&d_sink, sizeof(int)) == cudaSuccess,
         "cudaMalloc(d_sink) failed");

  cudaStream_t cap_stream = nullptr;
  ASSERT(cudaStreamCreate(&cap_stream) == cudaSuccess, "cap_stream create failed");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  ASSERT(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal) == cudaSuccess,
         "cudaStreamBeginCapture failed");
  busy_wait_kernel_A<<<1, 32, 0, cap_stream>>>(d_sink, kBusyCycles);
  busy_wait_kernel_B<<<1, 32, 0, cap_stream>>>(d_sink, kBusyCycles);
  ASSERT(cudaStreamEndCapture(cap_stream, &graph) == cudaSuccess,
         "cudaStreamEndCapture failed");
  ASSERT(cudaGraphInstantiate(&graph_exec, graph, 0) == cudaSuccess,
         "cudaGraphInstantiate failed");
  for (int i = 0; i < kGraphTraceReplays; ++i) {
    ASSERT(cudaGraphLaunch(graph_exec, cap_stream) == cudaSuccess,
           "cudaGraphLaunch failed");
  }
  ASSERT(cudaStreamSynchronize(cap_stream) == cudaSuccess, "graph stream sync failed");

  // One eager launch of A, outside any graph.
  busy_wait_kernel_A<<<1, 32, 0, cap_stream>>>(d_sink, kBusyCycles);
  ASSERT(cudaGetLastError() == cudaSuccess, "eager launch failed");
  ASSERT(cudaStreamSynchronize(cap_stream) == cudaSuccess, "eager sync failed");

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);

  // Several writer intervals; every write force-flushes CUPTI first.
  std::this_thread::sleep_for(std::chrono::milliseconds(700));

  const std::string json = read_file(shm_json_path());
  ASSERT(!json.empty(), "cupti.json should exist in the graph-trace child");

  // --- the mode gauge explains the rest of the payload.
  {
    const auto blocks = find_metric_blocks(json, "cuda_graph_trace_mode");
    ASSERT(blocks.size() == 1, "cuda_graph_trace_mode gauge missing");
    ASSERT(blocks[0].find("\"type\":\"gauge\"") != std::string::npos,
           "cuda_graph_trace_mode must serialize as a gauge");
    const double value = block_double(blocks[0], "value");
    std::printf("cuda_graph_trace_mode=%g\n", value);
    ASSERT(value == (node_mode ? 1.0 : 0.0),
           "cuda_graph_trace_mode must report the mode in effect");
  }

  const auto kernel_blocks = find_metric_blocks(json, "cuda_kernels_nanoseconds");
  ASSERT(kernel_blocks.size() == 1, "cuda_kernels_nanoseconds profile missing");
  const auto graph_blocks = find_metric_blocks(json, "cuda_graphs_nanoseconds");
  ASSERT(graph_blocks.size() == 1, "cuda_graphs_nanoseconds profile missing");

  const uint64_t samples_a = frame_samples_containing(kernel_blocks[0], "busy_wait_kernel_A");
  const uint64_t samples_b = frame_samples_containing(kernel_blocks[0], "busy_wait_kernel_B");
  const auto graph_frames = parse_frames(graph_blocks[0]);
  std::printf("graph-trace child: kernel samples A=%llu B=%llu, graph frames=%zu\n",
              static_cast<unsigned long long>(samples_a),
              static_cast<unsigned long long>(samples_b), graph_frames.size());

  if (node_mode) {
    // NODE mode: the graph's nodes are instrumented, so both kernels show up
    // as ordinary kernels — A once per replay plus the eager launch, B once
    // per replay — and nothing is reported at graph granularity.
    ASSERT_EQ(samples_a, static_cast<uint64_t>(kGraphTraceReplays + 1),
              "node mode: kernel A must be sampled once per replay plus the eager launch");
    ASSERT_EQ(samples_b, static_cast<uint64_t>(kGraphTraceReplays),
              "node mode: kernel B must be sampled once per replay");
    ASSERT(frame_value_containing(kernel_blocks[0], "busy_wait_kernel_A") > 0,
           "node mode: kernel A must carry a duration");
    ASSERT(frame_value_containing(kernel_blocks[0], "busy_wait_kernel_B") > 0,
           "node mode: kernel B must carry a duration");
    ASSERT(graph_frames.empty(),
           "node mode: cuda_graphs_nanoseconds must stay empty");
  } else {
    // GRAPH mode: one record per replay, aggregated into a single structural
    // frame; the graph's kernels are not reported individually, so only the
    // eager launch of A reaches cuda_kernels_nanoseconds and B is absent.
    ASSERT_EQ(graph_frames.size(), static_cast<size_t>(1),
              "graph mode: exactly one graph frame expected");
    ASSERT_EQ(graph_frames[0].samples, static_cast<uint64_t>(kGraphTraceReplays),
              "graph mode: the graph frame must count every replay");
    ASSERT(graph_frames[0].value > 0, "graph mode: graph frame duration must be > 0");
    ASSERT_EQ(samples_a, 1ull,
              "graph mode: only the eager launch of kernel A may be sampled");
    ASSERT(!has_frame_containing(kernel_blocks[0], "busy_wait_kernel_B"),
           "graph mode: a graph-only kernel must not appear in cuda_kernels_nanoseconds");
  }

  cudaStreamDestroy(cap_stream);
  cudaFree(d_sink);
  cupti_activity_stop();
  std::printf("graph-trace child (%s mode) passed\n", node_mode ? "node" : "graph");
  return 0;
}

// Runs the scenario in a re-exec'd child with GRAPHSIGNAL_CUDA_GRAPH_TRACE set
// to `env_value` (nullptr = leave it unset, i.e. the default mode).
static void run_graph_trace_child(const char* env_value) {
  std::printf("--- graph-trace child: GRAPHSIGNAL_CUDA_GRAPH_TRACE=%s\n",
              env_value ? env_value : "<unset>");
  pid_t pid = fork();
  ASSERT(pid >= 0, "fork must succeed");
  if (pid == 0) {
    if (env_value) {
      setenv("GRAPHSIGNAL_CUDA_GRAPH_TRACE", env_value, 1);
    } else {
      unsetenv("GRAPHSIGNAL_CUDA_GRAPH_TRACE");
    }
    char arg0[] = "cupti_activity_test";
    char arg1[] = "--graph-trace-scenario";
    char* const child_argv[] = {arg0, arg1, nullptr};
    execv("/proc/self/exe", child_argv);
    _exit(127);  // execv only returns on failure
  }
  int status = 0;
  ASSERT(waitpid(pid, &status, 0) == pid, "waitpid must return the child");
  ASSERT(WIFEXITED(status), "graph-trace child must exit normally");
  ASSERT_EQ(WEXITSTATUS(status), 0, "graph-trace child must exit 0");
}

// ---------------------------------------------------------------------------
// Device probes vs. CUDA graph capture. Engines (vLLM, SGLang) capture graphs
// in cudaStreamCaptureModeGlobal, where a potentially unsafe CUDA call from ANY
// thread invalidates the capture — the engine then fails at startup. Two paths
// must never do that: the writer thread's device probe reader (runs every
// interval, so it overlaps captures), and graphsignal_probe_register_cuda
// called on the capturing thread mid-capture (lazy registration in a kernel
// wrapper that first runs under capture). Re-exec'd child: its own CUPTI
// session with a short interval so the reader overlaps every capture window.
// ---------------------------------------------------------------------------

static const char* kCaptureMarker = "--capture-scenario";
static const int kCaptureRounds = 100;

static int capture_scenario() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("capture child: CUDA not available, skipping\n");
    return 0;
  }
  ASSERT(cudaSetDevice(0) == cudaSuccess, "capture child: cudaSetDevice failed");

  // Registered before capture, so the reader copies it on every write.
  graphsignal_probe_entry* pre = graphsignal_probe_register_cuda(
      "capture_pre_duration", NULL, NULL, 0);
  ASSERT(pre != nullptr, "capture child: pre-capture registration failed");

  const uint64_t write_interval_ns = 1'000'000ULL;  // 1ms: overlap every capture
  ASSERT(cupti_activity_start(write_interval_ns, /*debug_mode=*/0,
                              GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH) == 1,
         "cupti_activity_start failed in capture child");

  cudaStream_t s = nullptr;
  ASSERT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess,
         "capture child: stream create failed");

  // Phase A: the reader runs while captures are open.
  // Phase B: a new device probe is registered on the capturing thread mid-capture.
  int reader_failures = 0;
  int register_failures = 0;
  graphsignal_probe_entry* during = nullptr;
  for (int round = 0; round < 2 * kCaptureRounds; ++round) {
    const bool register_now = round >= kCaptureRounds;
    ASSERT(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal) == cudaSuccess,
           "capture child: begin capture failed");
    probe_record_kernel<<<1, 32, 0, s>>>(graphsignal_probe_device_data(pre), 1000);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));  // several writer ticks
    if (register_now) {
      during = graphsignal_probe_register_cuda("capture_during_duration", NULL, NULL, 0);
      probe_record_kernel<<<1, 32, 0, s>>>(graphsignal_probe_device_data(during), 1000);
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t rc = cudaStreamEndCapture(s, &graph);
    if (rc != cudaSuccess) {
      if (register_now) register_failures++; else reader_failures++;
      if (reader_failures + register_failures <= 3) {
        std::fprintf(stderr, "capture child: round %d capture failed: %s\n", round,
                     cudaGetErrorString(rc));
      }
      if (graph) cudaGraphDestroy(graph);
      cudaGetLastError();
      continue;
    }
    cudaGraphExec_t exec = nullptr;
    ASSERT(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess,
           "capture child: instantiate failed");
    ASSERT(cudaGraphLaunch(exec, s) == cudaSuccess, "capture child: launch failed");
    ASSERT(cudaStreamSynchronize(s) == cudaSuccess, "capture child: sync failed");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
  std::printf("capture child: reader-overlap failures=%d, register-in-capture failures=%d\n",
              reader_failures, register_failures);
  ASSERT_EQ(reader_failures, 0, "device probe reader must never invalidate a capture");
  ASSERT_EQ(register_failures, 0, "registration mid-capture must never invalidate it");
  ASSERT(during != nullptr, "registration mid-capture must succeed");

  // Both probes keep being read: every successful replay recorded once.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const std::string json = read_file(shm_json_path());
  const auto pre_blocks = find_metric_blocks(json, "capture_pre_duration");
  ASSERT(pre_blocks.size() == 1, "capture child: capture_pre_duration missing");
  ASSERT_EQ(block_u64(pre_blocks[0], "count"), static_cast<uint64_t>(2 * kCaptureRounds),
            "capture child: every replay must be recorded into the pre-capture probe");
  const auto during_blocks = find_metric_blocks(json, "capture_during_duration");
  ASSERT(during_blocks.size() == 1, "capture child: capture_during_duration missing");
  ASSERT_EQ(block_u64(during_blocks[0], "count"), static_cast<uint64_t>(kCaptureRounds),
            "capture child: every phase-B replay must be recorded");

  cudaStreamDestroy(s);
  cupti_activity_stop();
  std::printf("capture child passed\n");
  return 0;
}

static void run_capture_child() {
  std::printf("--- capture child: device probes vs. global-mode graph capture\n");
  pid_t pid = fork();
  ASSERT(pid >= 0, "fork must succeed");
  if (pid == 0) {
    char arg0[] = "cupti_activity_test";
    char arg1[] = "--capture-scenario";
    char* const child_argv[] = {arg0, arg1, nullptr};
    execv("/proc/self/exe", child_argv);
    _exit(127);
  }
  int status = 0;
  ASSERT(waitpid(pid, &status, 0) == pid, "waitpid must return the child");
  ASSERT(WIFEXITED(status), "capture child must exit normally");
  ASSERT_EQ(WEXITSTATUS(status), 0, "capture child must exit 0");
}

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], kGraphTraceMarker) == 0) {
    return graph_trace_scenario();
  }
  if (argc > 1 && std::strcmp(argv[1], kCaptureMarker) == 0) {
    return capture_scenario();
  }

  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::printf("CUDA not available, skipping CUDA operations test\n");
    return 0;
  }
  std::printf("CUDA available, testing with actual kernel and memcpy operations...\n");
  ASSERT(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice failed");

  // CUDA graph tracing granularity, before this process starts its own CUPTI
  // session: each case is a separate re-exec'd process with its own
  // environment, CUPTI session and shm file.
  run_graph_trace_child(nullptr);   // default: GRAPH granularity
  run_graph_trace_child("node");    // opt-in: NODE granularity
  run_graph_trace_child("nodes");   // invalid: falls back to GRAPH

  // Device probes must never break a workload's CUDA graph capture.
  run_capture_child();

  // HOST-storage probe, registered before profiling starts. The writer reads
  // the registry on every serialize, so it appears in cupti.json.
  graphsignal_probe_entry* host_probe = graphsignal_probe_register(
      "testapp_host_duration", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0);
  ASSERT(host_probe != nullptr, "host probe registration failed");
  for (int i = 0; i < 5; ++i) {
    graphsignal_record(host_probe, 1000000ull + static_cast<uint64_t>(i) * 1000);
  }

  // HOST-storage PROFILE probe: frames accumulate exact values and surface in
  // cupti.json through the same registry pass.
  graphsignal_probe_entry* host_profile = graphsignal_probe_register(
      "testapp_host_profile", GRAPHSIGNAL_PROFILE, NULL, NULL, 0);
  ASSERT(host_profile != nullptr, "host profile probe registration failed");
  graphsignal_profile_add_by_name(host_profile, "stage.prefill", 111);
  graphsignal_profile_add_by_name(host_profile, "stage.decode", 222);
  graphsignal_profile_add_by_name(host_profile, "stage.decode", 333);

  const uint64_t write_interval_ns = 100'000'000ULL;  // 100ms
  ASSERT(cupti_activity_start(write_interval_ns, /*debug_mode=*/1,
                              GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH) == 1,
         "cupti_activity_start failed");
  ASSERT(cupti_activity_get_debug_mode() == 1, "debug mode should be on");
  cupti_activity_set_debug_mode(0);
  ASSERT(cupti_activity_get_debug_mode() == 0, "debug mode should toggle off");
  cupti_activity_set_debug_mode(1);

  // ~2-5ms per busy-wait launch at typical SM clocks — long enough that every
  // sync below measurably blocks (zero-length sync records are skipped by the
  // lib), short enough to keep the test quick.
  const unsigned long long kBusyCycles = 5'000'000ULL;
  const int kKernelIters = 3;   // launches per busy-wait kernel symbol
  const int kGraphLaunches = 5; // graph launches (cumulative graph frame time)

  const int n = 1024;
  const size_t kCopyBytes = n * sizeof(float);  // 4096
  float* h_data = new float[n];
  for (int i = 0; i < n; i++) h_data[i] = static_cast<float>(i);
  float* d_data = nullptr;
  ASSERT(cudaMalloc(&d_data, kCopyBytes) == cudaSuccess, "cudaMalloc failed");
  int* d_sink = nullptr;
  ASSERT(cudaMalloc(&d_sink, sizeof(int)) == cudaSuccess, "cudaMalloc(d_sink) failed");

  // Memcpy with known byte totals: 2 x H2D, 1 x D2H. Asserted exactly against
  // the cumulative byte counters at the first read (nothing else in this phase
  // copies host<->device).
  const uint64_t kExpectedH2DBytes = 2 * kCopyBytes;
  const uint64_t kExpectedD2HBytes = kCopyBytes;
  ASSERT(cudaMemcpy(d_data, h_data, kCopyBytes, cudaMemcpyHostToDevice) == cudaSuccess,
         "cudaMemcpy HtoD 1 failed");
  ASSERT(cudaMemcpy(d_data, h_data, kCopyBytes, cudaMemcpyHostToDevice) == cudaSuccess,
         "cudaMemcpy HtoD 2 failed");
  ASSERT(cudaMemcpy(h_data, d_data, kCopyBytes, cudaMemcpyDeviceToHost) == cudaSuccess,
         "cudaMemcpy DtoH failed");

  // Memset on device memory.
  ASSERT(cudaMemset(d_data, 0, kCopyBytes) == cudaSuccess, "cudaMemset failed");

  // Two distinct busy-wait kernels, kKernelIters launches each, on an explicit
  // stream; cudaStreamSynchronize blocks on the tail -> a "stream" sync record.
  cudaStream_t stream = nullptr;
  ASSERT(cudaStreamCreate(&stream) == cudaSuccess, "cudaStreamCreate failed");
  for (int i = 0; i < kKernelIters; ++i) {
    busy_wait_kernel_A<<<1, 32, 0, stream>>>(d_sink, kBusyCycles);
    busy_wait_kernel_B<<<1, 32, 0, stream>>>(d_sink, kBusyCycles);
  }
  ASSERT(cudaGetLastError() == cudaSuccess, "busy-wait launches failed");
  ASSERT(cudaStreamSynchronize(stream) == cudaSuccess, "cudaStreamSynchronize failed");

  // cudaDeviceSynchronize behind a running kernel -> a "context" sync record.
  busy_wait_kernel_A<<<1, 32>>>(d_sink, kBusyCycles);
  ASSERT(cudaGetLastError() == cudaSuccess, "device-sync kernel launch failed");
  ASSERT(cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize failed");

  // cudaEventSynchronize on an event recorded behind a running kernel -> an
  // "event" sync record.
  cudaEvent_t done = nullptr;
  ASSERT(cudaEventCreate(&done) == cudaSuccess, "cudaEventCreate failed");
  busy_wait_kernel_B<<<1, 32, 0, stream>>>(d_sink, kBusyCycles);
  ASSERT(cudaGetLastError() == cudaSuccess, "event-sync kernel launch failed");
  ASSERT(cudaEventRecord(done, stream) == cudaSuccess, "cudaEventRecord failed");
  ASSERT(cudaEventSynchronize(done) == cudaSuccess, "cudaEventSynchronize failed");
  cudaEventDestroy(done);

  // CUDA graph capture + replay. With GRAPH_TRACE enabled the replayed kernels
  // are not reported individually; each launch arrives as one graph-level
  // record, aggregated into the cuda_graphs_nanoseconds profile frame named by the
  // FNV-1a64 hash of the graph's structural signature (computed at
  // instantiation via the resource callback).
  {
    cudaStream_t cap_stream = nullptr;
    ASSERT(cudaStreamCreate(&cap_stream) == cudaSuccess, "cap_stream create failed");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal) == cudaSuccess,
           "cudaStreamBeginCapture failed");
    for (int i = 0; i < 3; ++i) {
      test_kernel<<<4, 256, 0, cap_stream>>>(d_data, n);
    }
    ASSERT(cudaStreamEndCapture(cap_stream, &graph) == cudaSuccess,
           "cudaStreamEndCapture failed");
    ASSERT(cudaGraphInstantiate(&graph_exec, graph, 0) == cudaSuccess,
           "cudaGraphInstantiate failed");
    for (int i = 0; i < kGraphLaunches; ++i) {
      ASSERT(cudaGraphLaunch(graph_exec, cap_stream) == cudaSuccess,
             "cudaGraphLaunch failed");
    }
    ASSERT(cudaStreamSynchronize(cap_stream) == cudaSuccess, "graph stream sync failed");
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cap_stream);
  }

  // A few more host probe observations while profiling is live.
  for (int i = 0; i < 5; ++i) {
    graphsignal_record(host_probe, 2000000ull + static_cast<uint64_t>(i) * 1000);
  }

  // Let the writer run several intervals; each write force-flushes CUPTI first
  // (the flush hook), so all activity above lands in the instruments.
  std::this_thread::sleep_for(std::chrono::milliseconds(700));

  const std::string json1 = read_file(shm_json_path());
  std::printf("Read %zu bytes from %s\n", json1.size(), shm_json_path().c_str());
  ASSERT(!json1.empty(), "cupti.json should exist and be non-empty");
  ASSERT(json1.find("\"version\":1") != std::string::npos, "version must be 1");
  const uint64_t write_ts1 = block_u64(json1, "write_ts");
  ASSERT(write_ts1 > 0, "write_ts must be present");

  // --- cuda_kernels_nanoseconds: one profile instrument; frames keyed by the raw mangled
  // kernel symbol, values = cumulative duration ns.
  {
    const auto blocks = find_metric_blocks(json1, "cuda_kernels_nanoseconds");
    ASSERT(blocks.size() == 1, "cuda_kernels_nanoseconds profile missing (or duplicated)");
    ASSERT(blocks[0].find("\"type\":\"profile\"") != std::string::npos,
           "cuda_kernels_nanoseconds must serialize as a profile");
    const uint64_t dur_a = frame_value_containing(blocks[0], "busy_wait_kernel_A");
    const uint64_t dur_b = frame_value_containing(blocks[0], "busy_wait_kernel_B");
    std::printf("cuda_kernels_nanoseconds: A=%llu ns B=%llu ns\n",
                static_cast<unsigned long long>(dur_a),
                static_cast<unsigned long long>(dur_b));
    ASSERT(dur_a > 0, "cuda_kernels_nanoseconds frame for busy_wait_kernel_A missing");
    ASSERT(dur_b > 0, "cuda_kernels_nanoseconds frame for busy_wait_kernel_B missing");
  }

  // --- cuda_graphs_nanoseconds: one frame — the 16-hex hash of the graph's structural
  // signature — with the cumulative duration of all launches.
  {
    const auto blocks = find_metric_blocks(json1, "cuda_graphs_nanoseconds");
    ASSERT(blocks.size() == 1, "cuda_graphs_nanoseconds profile missing");
    ASSERT(blocks[0].find("\"type\":\"profile\"") != std::string::npos,
           "cuda_graphs_nanoseconds must serialize as a profile");
    const auto frames = parse_frames(blocks[0]);
    ASSERT(frames.size() == 1, "exactly one graph frame expected");
    std::printf("cuda_graphs_nanoseconds frame: %s=%llu ns\n", frames[0].name.c_str(),
                static_cast<unsigned long long>(frames[0].value));
    ASSERT(frames[0].value > 0, "graph frame duration must be > 0");
    ASSERT(frames[0].samples > 0, "graph frame must count its replays");
    ASSERT(frames[0].name.size() == 16,
           "graph frame must be a 16-hex signature hash");
    for (char c : frames[0].name) {
      ASSERT(std::isxdigit(static_cast<unsigned char>(c)) &&
             !std::isupper(static_cast<unsigned char>(c)),
             "graph frame must be lowercase hex");
    }
  }

  // --- cuda_memcpy_nanoseconds: per-kind duration frames plus exact cumulative byte
  // totals per direction.
  {
    const auto blocks = find_metric_blocks(json1, "cuda_memcpy_nanoseconds");
    ASSERT(blocks.size() == 1, "cuda_memcpy_nanoseconds profile missing");
    ASSERT(blocks[0].find("\"type\":\"profile\"") != std::string::npos,
           "cuda_memcpy_nanoseconds must serialize as a profile");
    ASSERT(frame_value(blocks[0], "host_to_device") > 0,
           "cuda_memcpy_nanoseconds frame for host_to_device missing");
    ASSERT(frame_value(blocks[0], "device_to_host") > 0,
           "cuda_memcpy_nanoseconds frame for device_to_host missing");

    const std::string h2d =
        find_metric_with_tag(json1, "cuda_memcpy_bytes", "kind", "host_to_device");
    ASSERT(!h2d.empty(), "cuda_memcpy_bytes[host_to_device] missing");
    ASSERT_EQ(block_u64(h2d, "value"), kExpectedH2DBytes,
              "host_to_device byte total mismatch");
    const std::string d2h =
        find_metric_with_tag(json1, "cuda_memcpy_bytes", "kind", "device_to_host");
    ASSERT(!d2h.empty(), "cuda_memcpy_bytes[device_to_host] missing");
    ASSERT_EQ(block_u64(d2h, "value"), kExpectedD2HBytes,
              "device_to_host byte total mismatch");
  }

  // --- cuda_memset_nanoseconds: duration frame + byte counter for device memory.
  {
    const auto blocks = find_metric_blocks(json1, "cuda_memset_nanoseconds");
    ASSERT(blocks.size() == 1, "cuda_memset_nanoseconds profile missing");
    ASSERT(frame_value(blocks[0], "device") > 0,
           "cuda_memset_nanoseconds frame for device missing");
    const std::string ms =
        find_metric_with_tag(json1, "cuda_memset_bytes", "kind", "device");
    ASSERT(!ms.empty(), "cuda_memset_bytes[device] missing");
    ASSERT(block_u64(ms, "value") >= kCopyBytes, "memset byte total too small");
  }

  // --- cuda_sync_nanoseconds: one frame per sync type exercised above.
  {
    const auto blocks = find_metric_blocks(json1, "cuda_sync_nanoseconds");
    ASSERT(blocks.size() == 1, "cuda_sync_nanoseconds profile missing");
    ASSERT(blocks[0].find("\"type\":\"profile\"") != std::string::npos,
           "cuda_sync_nanoseconds must serialize as a profile");
    ASSERT(frame_value(blocks[0], "stream") > 0,
           "cuda_sync_nanoseconds frame for stream missing");
    ASSERT(frame_value(blocks[0], "context") > 0,
           "cuda_sync_nanoseconds frame for context missing");
    ASSERT(frame_value(blocks[0], "event") > 0,
           "cuda_sync_nanoseconds frame for event missing");
  }

  // --- HOST-storage probes surface through the writer's registry pass.
  {
    const auto blocks = find_metric_blocks(json1, "testapp_host_duration");
    ASSERT(!blocks.empty(), "testapp_host_duration missing");
    assert_histogram_sane(blocks[0], 10, "testapp_host_duration");
  }
  {
    const auto blocks = find_metric_blocks(json1, "testapp_host_profile");
    ASSERT(!blocks.empty(), "testapp_host_profile missing");
    ASSERT(blocks[0].find("\"type\":\"profile\"") != std::string::npos,
           "host profile probe must serialize as a profile");
    ASSERT_EQ(frame_value(blocks[0], "stage.prefill"), 111ull,
              "host profile probe frame stage.prefill");
    ASSERT_EQ(frame_value(blocks[0], "stage.decode"), 555ull,
              "host profile probe frame stage.decode must accumulate 222+333");
    for (const Frame& f : parse_frames(blocks[0])) {
      ASSERT_EQ(f.samples, f.name == "stage.decode" ? 2ull : 1ull,
                "host profile probe frames must carry sample counts");
    }
  }

  // --- debug_mode=1: the retained log ring is serialized on every write.
  ASSERT(json1.find("\"log\":[{\"ts\":") != std::string::npos,
         "log array should contain captured entries");
  ASSERT(json1.find("cupti bufferCompleted") != std::string::npos,
         "log should contain cupti bufferCompleted lines");

  // =========================================================================
  // DEVICE-storage probe: registered now (its setup does its own H2D copy and
  // device memset, which is why the exact memcpy byte asserts above ran on the
  // first read), recorded from a kernel, then read out of device memory by the
  // lib's driver-API device probe reader on the next writes.
  // =========================================================================
  graphsignal_probe_entry* device_probe = graphsignal_probe_register_cuda(
      "testapp_device_duration", NULL, NULL, 0);
  ASSERT(device_probe != nullptr, "device probe registration failed");
  probe_record_kernel<<<1, 32>>>(graphsignal_probe_device_data(device_probe),
                                 kBusyCycles);
  ASSERT(cudaGetLastError() == cudaSuccess, "probe kernel launch failed");
  ASSERT(cudaDeviceSynchronize() == cudaSuccess, "probe kernel sync failed");

  std::this_thread::sleep_for(std::chrono::milliseconds(700));

  const std::string json2 = read_file(shm_json_path());
  ASSERT(!json2.empty(), "cupti.json should still be present");
  ASSERT(json2.find("\"version\":1") != std::string::npos, "version must stay 1");
  const uint64_t write_ts2 = block_u64(json2, "write_ts");
  ASSERT(write_ts2 > write_ts1, "write_ts must be monotone across reads");

  {
    const auto blocks = find_metric_blocks(json2, "testapp_device_duration");
    ASSERT(!blocks.empty(),
           "testapp_device_duration missing — device probe reader failed");
    assert_histogram_sane(blocks[0], 1, "testapp_device_duration");
    ASSERT(find_metric_blocks(json2, "testapp_host_duration").size() == 1,
           "testapp_host_duration missing from second read");
  }

  cudaStreamDestroy(stream);
  cudaFree(d_sink);
  cudaFree(d_data);
  delete[] h_data;

  // Stop tears CUPTI down and performs one final write (no CUDA involved on
  // that path); the file must survive.
  cupti_activity_stop();
  const std::string json3 = read_file(shm_json_path());
  ASSERT(!json3.empty(), "cupti.json must exist after stop (final write)");
  ASSERT(json3.find("\"version\":1") != std::string::npos,
         "final file must carry version 1");

  std::printf("All cupti_activity tests passed!\n");
  return 0;
}
