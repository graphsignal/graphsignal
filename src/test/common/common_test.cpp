// Pure-C++ tests for the internal writer layer (src/common/metrics_writer.h).
// No GPU, no gtest. Build:
//   c++ -std=c++17 -Wall -Wextra -Iinclude -Isrc/common -pthread \
//       -o build/common_test src/test/common/common_test.cpp

#include "metrics_writer.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

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

#define CASE(name) std::printf("CASE %s\n", name)

// A background writer interval long enough to never fire during the test, so
// only explicit write_now()/shutdown() writes happen (avoids races with the
// "*.tmp never left behind" scan and flush-hook counting).
constexpr uint64_t kLongIntervalNs = 3600ull * 1000000000ull;

// ---- helpers --------------------------------------------------------------

static std::string make_temp_dir() {
  const char* base = std::getenv("TMPDIR");
  std::string tmpl = (base && *base) ? base : "/tmp";
  if (tmpl.back() != '/') tmpl += '/';
  tmpl += "gs_common_test_XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char* r = mkdtemp(buf.data());
  ASSERT(r != NULL, "mkdtemp must succeed");
  return std::string(r);
}

static bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  ASSERT(f != NULL, "output file must exist and be readable");
  std::string out;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

static bool contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

// Parses the unsigned integer following `"key":`.
static uint64_t num_after(const std::string& json, const std::string& key) {
  std::string pat = "\"" + key + "\":";
  size_t p = json.find(pat);
  ASSERT(p != std::string::npos, "expected numeric key in JSON");
  return strtoull(json.c_str() + p + pat.size(), NULL, 10);
}

// Tiny well-formedness check: braces/brackets balanced (string-aware).
static void assert_well_formed(const std::string& j) {
  ASSERT(!j.empty() && j.front() == '{' && j.back() == '}',
         "JSON must be a single object");
  std::vector<char> stack;
  bool in_str = false;
  for (size_t i = 0; i < j.size(); i++) {
    char c = j[i];
    if (in_str) {
      if (c == '\\') { i++; continue; }
      if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') {
      in_str = true;
    } else if (c == '{' || c == '[') {
      stack.push_back(c);
    } else if (c == '}') {
      ASSERT(!stack.empty() && stack.back() == '{', "balanced braces");
      stack.pop_back();
    } else if (c == ']') {
      ASSERT(!stack.empty() && stack.back() == '[', "balanced brackets");
      stack.pop_back();
    }
  }
  ASSERT(!in_str, "no unterminated string");
  ASSERT(stack.empty(), "all braces/brackets closed");
}

// Extracts the full metric object `{"name":"<name>",...}` (handles the nested
// tags object) or returns "" if the metric is absent.
static std::string metric_object(const std::string& j, const std::string& name) {
  std::string pat = "{\"name\":\"" + name + "\"";
  size_t p = j.find(pat);
  if (p == std::string::npos) return "";
  int depth = 0;
  bool in_str = false;
  for (size_t i = p; i < j.size(); i++) {
    char c = j[i];
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
      if (depth == 0) return j.substr(p, i - p + 1);
    }
  }
  ASSERT(false, "metric object must terminate");
  return "";
}

// Like metric_object, but the object must also contain `needle` (e.g. a tag
// string) — for several objects of one metric name.
static std::string metric_object_with(const std::string& j, const std::string& name,
                                      const std::string& needle) {
  std::string key = "{\"name\":\"" + name + "\"";
  size_t pos = 0;
  while ((pos = j.find(key, pos)) != std::string::npos) {
    size_t end = j.find("}", pos);
    // the object may contain nested {} (tags): find the matching brace
    int depth = 0;
    size_t k = pos;
    for (; k < j.size(); k++) {
      if (j[k] == '{') depth++;
      else if (j[k] == '}') { depth--; if (depth == 0) break; }
    }
    end = k;
    std::string obj = j.substr(pos, end - pos + 1);
    if (obj.find(needle) != std::string::npos) return obj;
    pos = end;
  }
  return "";
}

static void unset_context_env() {
  static const char* kVars[] = {
      "RANK", "NCCL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK",
      "LOCAL_RANK", "NCCL_LOCAL_RANK", "SLURM_LOCALID",
      "OMPI_COMM_WORLD_LOCAL_RANK",
      "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS",
      "LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE",
      "MASTER_ADDR", "MASTER_PORT",
      "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_STEPID",
      "SLURM_NODEID", "SLURM_NNODES", "SLURM_JOB_NUM_NODES"};
  for (const char* v : kVars) unsetenv(v);
}

static void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

static void hook_count(void* arg) {
  (*static_cast<int*>(arg))++;
}

static void fake_device_reader(const graphsignal_probe_entry* const* entries, size_t n,
                               graphsignal_instrument_data* out, uint8_t* ok, void* arg) {
  (void)arg;
  // The test's "device" data blocks are really host memory: plain copies.
  for (size_t i = 0; i < n; i++) {
    std::memcpy(&out[i], entries[i]->data, sizeof(out[i]));
    ok[i] = 1;
  }
}

// ---- cases ----------------------------------------------------------------

static void test_init_lifecycle() {
  CASE("init/lifecycle");

  std::string dir = make_temp_dir();
  graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init("cupti", dir.c_str(),
                                   100000000ull /* 100ms */, true);
  ASSERT(w != NULL, "init must return a writer");

  std::string expected_path =
      dir + "/graphsignal_" + std::to_string(getpid()) + "/cupti.json";
  ASSERT(w->file_path() == expected_path,
         "file_path must be <dir>/graphsignal_<pid>/cupti.json");

  w->write_now(false);
  ASSERT(file_exists(w->file_path()), "write_now must create the file");

  std::string j = read_file(w->file_path());
  assert_well_formed(j);
  ASSERT(contains(j, "\"version\":1"), "JSON must carry version 1");
  ASSERT(contains(j, "\"pid\":"), "JSON must carry pid");
  ASSERT_EQ(num_after(j, "pid"), (uint64_t)getpid(), "pid must match");
  uint64_t start_ts = num_after(j, "start_ts");
  uint64_t write_ts = num_after(j, "write_ts");
  ASSERT(start_ts > 0, "start_ts must be set");
  ASSERT(start_ts <= write_ts, "start_ts must not exceed write_ts");

  w->shutdown();
}

// Device reader over fake "device" blocks in host memory: counts calls and
// probes, and refuses one block, so the skip path can be checked.
static const graphsignal_instrument_data* g_refused_block = NULL;
static int g_reader_calls = 0;
static size_t g_reader_probes = 0;
static void refusing_device_reader(const graphsignal_probe_entry* const* entries, size_t n,
                                   graphsignal_instrument_data* out, uint8_t* ok, void*) {
  g_reader_calls++;
  g_reader_probes += n;
  for (size_t i = 0; i < n; i++) {
    if (entries[i]->data == g_refused_block) continue;
    memcpy(&out[i], entries[i]->data, sizeof(out[i]));
    ok[i] = 1;
  }
}

static void test_device_probe_reader() {
  CASE("device probe reader: one call per write with every probe, a failed copy skips only that probe");

  std::string dir = make_temp_dir();
  graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init(
      "cupti", dir.c_str(), kLongIntervalNs, false);
  ASSERT(w != NULL, "init must return a writer");

  graphsignal_instrument_data* ok_block = static_cast<graphsignal_instrument_data*>(
      calloc(1, sizeof(graphsignal_instrument_data)));
  graphsignal_instrument_data* bad_block = static_cast<graphsignal_instrument_data*>(
      calloc(1, sizeof(graphsignal_instrument_data)));
  ASSERT(ok_block != NULL && bad_block != NULL, "calloc must succeed");
  ok_block->count = 1; ok_block->sum = 99; ok_block->min = ok_block->max = 99;
  ok_block->bins[graphsignal_probe_bin_index(99)]++;
  bad_block->min = UINT64_MAX;
  ASSERT(graphsignal_probe_register_storage("reader_ok", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0,
                                            ok_block, 0) != NULL, "reg ok");
  ASSERT(graphsignal_probe_register_storage("reader_refused", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0,
                                            bad_block, 0) != NULL, "reg refused");

  g_refused_block = bad_block;
  w->set_device_probe_reader(refusing_device_reader, NULL);
  g_reader_calls = 0;
  g_reader_probes = 0;
  w->write_now(false);
  std::string j = read_file(w->file_path());
  assert_well_formed(j);
  // Earlier cases registered other DEVICE probes in this process; every one
  // is read exactly once per write.
  uint64_t device_probes = 0;
  graphsignal_probe_registry* reg = graphsignal_probe_reader_attach();
  ASSERT(reg != NULL, "registry must be attached");
  for (uint64_t i = 0; i < graphsignal_probe_reader_count(reg); i++) {
    if (reg->entries[i].storage == GRAPHSIGNAL_STORAGE_DEVICE) device_probes++;
  }
  ASSERT_EQ(g_reader_calls, 1, "one reader call per write");
  ASSERT_EQ((uint64_t)g_reader_probes, device_probes, "the call carries every device probe");
  ASSERT(contains(metric_object(j, "reader_ok"), "\"count\":1,\"sum\":99,"),
         "a successful read serializes the probe");
  ASSERT(!contains(j, "reader_refused"), "a failed read skips the probe for this write");
  w->shutdown();
}

static graphsignal::MetricsWriter* g_w2 = NULL;  // reused by the idempotence + atomicity cases
static graphsignal::Instrument* g_counter = NULL;

static void test_instruments() {
  CASE("instrument serialization");

  std::string dir = make_temp_dir();
  g_w2 = graphsignal::MetricsWriter::init("cupti", dir.c_str(), kLongIntervalNs, false);
  ASSERT(g_w2 != NULL, "init must return a writer");

  graphsignal::Instrument* h = g_w2->register_instrument(
      graphsignal::InstrumentType::Histogram, "gpu_kernel_time",
      {{"kernel", "k\"esc\\x"}});
  ASSERT(h != NULL, "histogram registration must succeed");
  graphsignal::MetricsWriter::record(h, 100);
  graphsignal::MetricsWriter::record(h, 5000);
  graphsignal::MetricsWriter::record(h, 3);

  g_counter = g_w2->register_instrument(
      graphsignal::InstrumentType::Counter, "c_total", {});
  ASSERT(g_counter != NULL, "counter registration must succeed");
  graphsignal::MetricsWriter::add(g_counter, 42);

  graphsignal::Instrument* g = g_w2->register_instrument(
      graphsignal::InstrumentType::Gauge, "g_val", {});
  ASSERT(g != NULL, "gauge registration must succeed");
  graphsignal::MetricsWriter::set(g, 2.5);

  g_w2->write_now(false);
  std::string j = read_file(g_w2->file_path());
  assert_well_formed(j);

  std::string hist = metric_object(j, "gpu_kernel_time");
  ASSERT(!hist.empty(), "histogram metric must be serialized");
  ASSERT(contains(hist, "\"type\":\"histogram\""), "histogram type");
  // 3 -> bin 3 (lower 3); 100 -> bin 22 (lower 96); 5000 -> bin 44 (lower
  // 4096). Sparse ascending arrays of equal length, counts summing to 3.
  ASSERT(contains(hist, "\"bins\":[3,96,4096],\"counts\":[1,1,1]"),
         "sparse bins/counts arrays");
  // Exact aggregates ride next to the bins: 3 values, sum 5103, min 3, max 5000.
  ASSERT(contains(hist, "\"count\":3,\"sum\":5103,\"min\":3,\"max\":5000"),
         "histogram exact aggregates");
  // Quote and backslash in the tag value must be escaped.
  ASSERT(contains(hist, "\"tags\":{\"kernel\":\"k\\\"esc\\\\x\"}"),
         "tag value escaping");

  std::string cnt = metric_object(j, "c_total");
  ASSERT(!cnt.empty(), "counter metric must be serialized");
  ASSERT(contains(cnt, "\"type\":\"counter\""), "counter type");
  ASSERT(contains(cnt, "\"value\":42"), "counter value");

  std::string gau = metric_object(j, "g_val");
  ASSERT(!gau.empty(), "gauge metric must be serialized");
  ASSERT(contains(gau, "\"type\":\"gauge\""), "gauge type");
  ASSERT(contains(gau, "\"value\":2.5"), "gauge 2.5 stays 2.5");

  // A histogram nothing has recorded carries neither half: no bins, no
  // aggregates. The importer rejects that entry rather than storing an
  // all-null series, which is what keeps unrecorded instruments out of
  // /signals.
  graphsignal::Instrument* empty_hist = g_w2->register_instrument(
      graphsignal::InstrumentType::Histogram, "never_recorded", {});
  ASSERT(empty_hist != NULL, "histogram registration must succeed");
  g_w2->write_now(false);
  std::string ehist = metric_object(read_file(g_w2->file_path()), "never_recorded");
  ASSERT(!ehist.empty(), "unrecorded histogram is still a metric object");
  ASSERT(!contains(ehist, "\"bins\":"), "unrecorded histogram has no bins");
  ASSERT(!contains(ehist, "\"count\":"), "unrecorded histogram has no aggregates");
}

static void test_idempotent_registration() {
  CASE("idempotent registration");

  graphsignal::Instrument* again = g_w2->register_instrument(
      graphsignal::InstrumentType::Histogram, "gpu_kernel_time",
      {{"kernel", "k\"esc\\x"}});
  ASSERT(again != NULL, "re-registration must succeed");
  graphsignal::Instrument* first = g_w2->register_instrument(
      graphsignal::InstrumentType::Histogram, "gpu_kernel_time",
      {{"kernel", "k\"esc\\x"}});
  ASSERT(again == first, "same name+tags must return the same Instrument*");

  graphsignal::Instrument* other = g_w2->register_instrument(
      graphsignal::InstrumentType::Histogram, "gpu_kernel_time",
      {{"kernel", "other"}});
  ASSERT(other != NULL && other != again,
         "different tags must return a distinct Instrument*");
}

static void test_profile_instruments() {
  CASE("profile instrument serialization");

  std::string dir = make_temp_dir();
  graphsignal::MetricsWriter* w =
      graphsignal::MetricsWriter::init("prof", dir.c_str(), kLongIntervalNs, false);
  ASSERT(w != NULL, "init must return a writer");

  graphsignal::Instrument* p = w->register_instrument(
      graphsignal::InstrumentType::Profile, "gpu_kernels", {});
  ASSERT(p != NULL, "profile registration must succeed");
  graphsignal::MetricsWriter::profile_add(p, "kern\"a", 100);
  graphsignal::MetricsWriter::profile_add(p, "kern\"a", 150);
  graphsignal::MetricsWriter::profile_add(p, "kern_b", 7);

  // NULL/empty frame names and NULL instruments are safe no-ops.
  graphsignal::MetricsWriter::profile_add(p, NULL, 1);
  graphsignal::MetricsWriter::profile_add(p, "", 1);
  graphsignal::MetricsWriter::profile_add(NULL, "x", 1);

  w->write_now(false);
  std::string j = read_file(w->file_path());
  assert_well_formed(j);

  std::string prof = metric_object(j, "gpu_kernels");
  ASSERT(!prof.empty(), "profile metric must be serialized");
  ASSERT(contains(prof, "\"type\":\"profile\""), "profile type");
  ASSERT(contains(prof, "\"frames\":{"), "profile frames object");
  // A frame is [cumulative value, samples]; the quote in the frame name
  // must be escaped.
  ASSERT(contains(prof, "\"kern\\\"a\":[250,2]"),
         "escaped frame must accumulate 100+150 over two samples");
  ASSERT(contains(prof, "\"kern_b\":[7,1]"), "second frame exact value and count");
  ASSERT(!contains(prof, "\"kern\\\"a\":[100,"),
         "frame must not keep its pre-accumulation value");
  ASSERT(!contains(prof, "\"bins\":"), "profile must not serialize bins");
  ASSERT(!contains(prof, "\"value\":"), "profile must not serialize value");

  // Frame cap: 260 distinct frames -> exactly kMaxProfileFrames serialized;
  // frames past the cap are silently dropped.
  graphsignal::Instrument* pc = w->register_instrument(
      graphsignal::InstrumentType::Profile, "gpu_capped", {});
  ASSERT(pc != NULL, "capped profile registration must succeed");
  for (uint32_t i = 0; i < graphsignal::kMaxProfileFrames + 10; i++) {
    char name[64];
    std::snprintf(name, sizeof(name), "capframe.%u", i);
    graphsignal::MetricsWriter::profile_add(pc, name, 1);
  }
  // An existing frame still accumulates at the cap.
  graphsignal::MetricsWriter::profile_add(pc, "capframe.0", 1);

  w->write_now(false);
  j = read_file(w->file_path());
  assert_well_formed(j);
  std::string capped = metric_object(j, "gpu_capped");
  ASSERT(!capped.empty(), "capped profile must be serialized");
  size_t frames = 0;
  size_t pos = 0;
  while ((pos = capped.find("\"capframe.", pos)) != std::string::npos) {
    frames++;
    pos++;
  }
  ASSERT_EQ(frames, (size_t)graphsignal::kMaxProfileFrames,
            "exactly kMaxProfileFrames frames must be serialized");
  ASSERT(contains(capped, "\"capframe.0\":[2,2]"),
         "existing frame must keep accumulating at the cap");

  w->shutdown();
}

static void test_writer_cap() {
  CASE("writer instrument cap");

  std::string dir = make_temp_dir();
  graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init("cap", dir.c_str(), kLongIntervalNs, false);
  ASSERT(w != NULL, "init must return a writer");

  const uint32_t kOver = 4;
  uint32_t registered = 0, rejected = 0;
  for (uint32_t i = 0; i < graphsignal::kMaxInstruments + kOver; i++) {
    char name[64];
    std::snprintf(name, sizeof(name), "cap_metric_%u", i);
    graphsignal::Instrument* inst = w->register_instrument(
        graphsignal::InstrumentType::Counter, name, {});
    if (inst) {
      registered++;
    } else {
      rejected++;
    }
  }
  ASSERT_EQ(registered, graphsignal::kMaxInstruments,
            "exactly kMaxInstruments registrations must succeed");
  ASSERT_EQ(rejected, kOver, "registrations past the cap must return nullptr");
  ASSERT_EQ(w->dropped_instruments(), (uint64_t)kOver,
            "dropped counter must count rejected registrations");

  // An already-registered instrument is still found at the cap.
  graphsignal::Instrument* existing = w->register_instrument(
      graphsignal::InstrumentType::Counter, "cap_metric_0", {});
  ASSERT(existing != NULL, "existing instrument must be found at the cap");
  ASSERT_EQ(w->dropped_instruments(), (uint64_t)kOver,
            "finding an existing instrument must not count as dropped");

  w->write_now(false);
  std::string j = read_file(w->file_path());
  assert_well_formed(j);
  std::string dropped =
      metric_object(j, "graphsignal_writer_dropped_instruments");
  ASSERT(!dropped.empty(), "dropped_instruments metric must be serialized");
  ASSERT(contains(dropped, "\"value\":4"), "dropped_instruments value");

  w->shutdown();
}

static void test_overwrite_atomicity() {
  CASE("full-state overwrite + atomicity");

  g_w2->write_now(false);
  std::string j1 = read_file(g_w2->file_path());
  uint64_t ts1 = num_after(j1, "write_ts");
  uint64_t counter1 = num_after(metric_object(j1, "c_total"), "value");

  graphsignal::MetricsWriter::add(g_counter, 1);
  sleep_ms(10);
  g_w2->write_now(false);
  std::string j2 = read_file(g_w2->file_path());
  uint64_t ts2 = num_after(j2, "write_ts");
  uint64_t counter2 = num_after(metric_object(j2, "c_total"), "value");

  ASSERT(ts2 > ts1, "second write_ts must be strictly newer");
  ASSERT_EQ(counter1, 42ull, "counter before second read");
  ASSERT_EQ(counter2, 43ull, "counter after second read");
  ASSERT(counter2 >= counter1, "counters must be monotone across reads");

  // No *.tmp file left behind (write is tmp + rename).
  DIR* d = opendir(g_w2->dir_path().c_str());
  ASSERT(d != NULL, "writer dir must be readable");
  struct dirent* ent;
  while ((ent = readdir(d)) != NULL) {
    size_t len = std::strlen(ent->d_name);
    ASSERT(!(len >= 4 && std::strcmp(ent->d_name + len - 4, ".tmp") == 0),
           "no *.tmp file may be left in the dir");
  }
  closedir(d);

  g_w2->shutdown();
}

static void test_context_capture() {
  CASE("context capture");

  unset_context_env();
  setenv("RANK", "7", 1);
  setenv("SLURM_JOB_ID", "abc123", 1);
  std::string dir = make_temp_dir();
  graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init("ctx", dir.c_str(), kLongIntervalNs, false);
  ASSERT(w != NULL, "init must return a writer");
  w->write_now(false);
  std::string j = read_file(w->file_path());
  assert_well_formed(j);
  ASSERT(contains(j, "\"context\":{"), "context object must be present");
  ASSERT(contains(j, "\"rank\":\"7\""), "rank captured from RANK");
  ASSERT(contains(j, "\"slurm_job_id\":\"abc123\""),
         "slurm_job_id captured from SLURM_JOB_ID");
  w->shutdown();

  unset_context_env();
  std::string dir2 = make_temp_dir();
  graphsignal::MetricsWriter* w2 = graphsignal::MetricsWriter::init("ctx", dir2.c_str(), kLongIntervalNs, false);
  ASSERT(w2 != NULL, "init must return a writer");
  w2->write_now(false);
  std::string j2 = read_file(w2->file_path());
  assert_well_formed(j2);
  ASSERT(!contains(j2, "\"context\""),
         "context must be omitted entirely without relevant env vars");
  w2->shutdown();
}

static void test_log_ring() {
  CASE("log ring");

  // Gating: error always captured; debug only when enabled.
  {
    std::string dir = make_temp_dir();
    graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init("log", dir.c_str(), kLongIntervalNs,
                                     /*debug_enabled=*/false);
    ASSERT(w != NULL, "init must return a writer");
    w->error("boom %d", 1);
    w->debug("hidden-dbg");
    w->write_now(false);
    std::string j = read_file(w->file_path());
    assert_well_formed(j);
    ASSERT(contains(j, "\"msg\":\"boom 1\""), "error must always be captured");
    ASSERT(!contains(j, "hidden-dbg"),
           "debug must be absent when debug is disabled");

    w->set_debug(true);
    w->debug("shown-dbg");
    w->write_now(false);
    j = read_file(w->file_path());
    ASSERT(contains(j, "\"msg\":\"shown-dbg\""),
           "debug must be captured after set_debug(true)");
    w->shutdown();
  }

  // Overflow: 300 entries -> exactly 256 survive, oldest is #44; retained
  // across writes; ts non-decreasing in serialization order.
  {
    std::string dir = make_temp_dir();
    graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init("log", dir.c_str(), kLongIntervalNs, false);
    ASSERT(w != NULL, "init must return a writer");
    for (int i = 0; i < 300; i++) {
      w->error("m%d", i);
    }
    w->write_now(false);
    std::string j = read_file(w->file_path());
    assert_well_formed(j);

    size_t entries = 0;
    size_t pos = 0;
    uint64_t prev_ts = 0;
    while ((pos = j.find("{\"ts\":", pos)) != std::string::npos) {
      uint64_t ts = strtoull(j.c_str() + pos + 6, NULL, 10);
      ASSERT(ts >= prev_ts, "log ts must be non-decreasing in order");
      prev_ts = ts;
      entries++;
      pos++;
    }
    ASSERT_EQ(entries, (size_t)graphsignal::kMaxLogEntries,
              "log array must have exactly 256 entries");
    ASSERT(contains(j, "\"msg\":\"m44\""),
           "oldest surviving message must be #44");
    ASSERT(!contains(j, "\"msg\":\"m43\""), "message #43 must be evicted");
    ASSERT(!contains(j, "\"msg\":\"m0\""), "message #0 must be evicted");
    ASSERT(contains(j, "\"msg\":\"m299\""), "newest message must survive");

    // Never drained: a second write serializes the same retained ring.
    w->write_now(false);
    std::string j2 = read_file(w->file_path());
    size_t entries2 = 0;
    pos = 0;
    while ((pos = j2.find("{\"ts\":", pos)) != std::string::npos) {
      entries2++;
      pos++;
    }
    ASSERT_EQ(entries2, (size_t)graphsignal::kMaxLogEntries,
              "log entries must be retained across writes");
    ASSERT(contains(j2, "\"msg\":\"m44\""),
           "retained oldest message must still be #44");
    w->shutdown();
  }
}

static void test_flush_hook() {
  CASE("flush hook");

  std::string dir = make_temp_dir();
  graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init("hook", dir.c_str(), kLongIntervalNs, false);
  ASSERT(w != NULL, "init must return a writer");

  int calls = 0;
  w->set_flush_hook(hook_count, &calls);
  w->write_now(true);
  ASSERT_EQ(calls, 1, "write_now(true) must call the flush hook");
  w->write_now(false);
  ASSERT_EQ(calls, 1, "write_now(false) must not call the flush hook");
  w->shutdown();
  ASSERT_EQ(calls, 1, "the final shutdown write must not call the hook");
  ASSERT(file_exists(w->file_path()), "shutdown must have written the file");
}

static void test_probe_merge() {
  CASE("probe merge");

  std::string dir = make_temp_dir();
  graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init("probe", dir.c_str(), kLongIntervalNs, false);
  ASSERT(w != NULL, "init must return a writer");

  // HOST probe registered through the public header appears in the JSON.
  graphsignal_probe_entry* hp = graphsignal_probe_register(
      "user_h_probe", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0);
  ASSERT(hp != NULL, "host probe registration must succeed");
  graphsignal_record(hp, 42);

  // PROFILE probe: frames merged the same way as writer profiles.
  graphsignal_probe_entry* pp = graphsignal_probe_register(
      "user_p_probe", GRAPHSIGNAL_PROFILE, NULL, NULL, 0);
  ASSERT(pp != NULL, "profile probe registration must succeed");
  graphsignal_profile_add_by_name(pp, "frame.x", 5);
  graphsignal_profile_add_by_name(pp, "frame.x", 6);
  graphsignal_profile_add_by_name(pp, "frame.y", 2);

  w->write_now(false);
  std::string j = read_file(w->file_path());
  assert_well_formed(j);
  std::string hobj = metric_object(j, "user_h_probe");
  ASSERT(!hobj.empty(), "host probe must be merged into the writer JSON");
  ASSERT(contains(hobj, "\"type\":\"histogram\""), "host probe type");
  // 42 -> bin 17 (lower 40); bins/counts only, no aggregate fields.
  ASSERT(contains(hobj, "\"bins\":[40],\"counts\":[1]"),
         "host probe bins/counts");
  ASSERT(contains(hobj, "\"count\":1,\"sum\":42,\"min\":42,\"max\":42"),
         "host probe exact aggregates");

  std::string pobj = metric_object(j, "user_p_probe");
  ASSERT(!pobj.empty(), "profile probe must be merged into the writer JSON");
  ASSERT(contains(pobj, "\"type\":\"profile\""), "profile probe type");
  // Probe frames serialize in insertion order as [value, samples].
  ASSERT(contains(pobj, "\"frames\":{\"frame.x\":[11,2],\"frame.y\":[2,1]}"),
         "profile probe frames with exact values and sample counts");

  // DEVICE-storage probe: fake "device" block in host memory.
  graphsignal_instrument_data* fake = static_cast<graphsignal_instrument_data*>(
      calloc(1, sizeof(graphsignal_instrument_data)));
  ASSERT(fake != NULL, "calloc must succeed");
  fake->min = UINT64_MAX;
  // Simulate device-side recording of the values 3 and 7.
  fake->count = 2;
  fake->sum = 10;
  fake->min = 3;
  fake->max = 7;
  fake->bins[graphsignal_probe_bin_index(3)]++;
  fake->bins[graphsignal_probe_bin_index(7)]++;

  graphsignal_probe_entry* dp = graphsignal_probe_register_storage(
      "user_d_probe", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0, fake, 0);
  ASSERT(dp != NULL, "device probe registration must succeed");
  ASSERT_EQ(dp->storage, (uint32_t)GRAPHSIGNAL_STORAGE_DEVICE,
            "device probe storage");
  ASSERT(dp->device_id == 0, "device probe device_id");

  // Without a device reader the DEVICE probe is skipped.
  w->write_now(false);
  j = read_file(w->file_path());
  ASSERT(!contains(j, "user_d_probe"),
         "device probe must be skipped without a device reader");

  // With a reader that copies from the fake block, it appears.
  w->set_device_probe_reader(fake_device_reader, NULL);
  w->write_now(false);
  j = read_file(w->file_path());
  assert_well_formed(j);
  std::string dobj = metric_object(j, "user_d_probe");
  ASSERT(!dobj.empty(), "device probe must appear once a reader is set");
  // 3 -> bin 3 (lower 3); 7 -> bin 7 (lower 7); bins/counts only.
  ASSERT(contains(dobj, "\"bins\":[3,7],\"counts\":[1,1]"),
         "device probe bins/counts");
  ASSERT(contains(dobj, "\"count\":2,\"sum\":10,\"min\":3,\"max\":7"),
         "device probe exact aggregates");

  w->shutdown();
}

static void test_shutdown() {
  CASE("shutdown");

  std::string base = make_temp_dir();
  graphsignal::MetricsWriter* wa = graphsignal::MetricsWriter::init("cupti", base.c_str(), kLongIntervalNs, false);
  graphsignal::MetricsWriter* wb = graphsignal::MetricsWriter::init("rocm", base.c_str(), kLongIntervalNs, false);
  ASSERT(wa != NULL && wb != NULL, "both writers must initialize");
  ASSERT(wa->dir_path() == wb->dir_path(),
         "both writers must share one graphsignal_<pid> dir");
  ASSERT(wa->file_path() != wb->file_path(),
         "the two writers must produce distinct files");

  wa->write_now(false);
  wb->write_now(false);
  ASSERT(file_exists(wa->file_path()), "cupti.json must exist");
  ASSERT(file_exists(wb->file_path()), "rocm.json must exist");

  // shutdown() produces a final write.
  ASSERT(unlink(wa->file_path().c_str()) == 0, "unlink must succeed");
  ASSERT(!file_exists(wa->file_path()), "file must be gone before shutdown");
  wa->shutdown();
  ASSERT(file_exists(wa->file_path()),
         "shutdown must produce a final write of the file");
  assert_well_formed(read_file(wa->file_path()));

  // Double-shutdown is safe.
  wa->shutdown();

  wb->shutdown();
  ASSERT(file_exists(wb->file_path()), "rocm.json must survive shutdown");
}

int main() {
  unset_context_env();  // deterministic context regardless of ambient env

  test_init_lifecycle();
  test_instruments();
  test_idempotent_registration();
  test_profile_instruments();
  test_writer_cap();
  test_overwrite_atomicity();
  test_context_capture();
  test_log_ring();
  test_flush_hook();
  test_probe_merge();
  test_device_probe_reader();
  test_shutdown();

  std::printf("All common tests passed!\n");
  return 0;
}
