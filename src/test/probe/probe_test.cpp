// Pure-C++ tests for the public probe header (include/graphsignal/probe.h).
// No GPU, no gtest. Build:
//   c++ -std=c++17 -Wall -Wextra -Iinclude -pthread -o build/probe_test \
//       src/test/probe/probe_test.cpp

#include <graphsignal/probe.h>

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
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

// Count of unique instruments registered so far (for reader_count checks).
static uint64_t g_expected_registrations = 0;

static graphsignal_probe_entry* must_register(
    const char* name, graphsignal_instrument_type type,
    const char* const* tag_keys, const char* const* tag_vals, size_t ntags) {
  graphsignal_probe_entry* e =
      graphsignal_probe_register(name, type, tag_keys, tag_vals, ntags);
  ASSERT(e != NULL, "registration should succeed");
  g_expected_registrations++;
  return e;
}

static void test_bin_math() {
  CASE("bin math");

  const uint64_t values[] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 1000, 1000000ull,
      1ull << 40, 1ull << 62, UINT64_MAX};

  for (uint64_t v : values) {
    uint32_t idx = graphsignal_probe_bin_index(v);
    ASSERT(idx < GRAPHSIGNAL_PROBE_HIST_BINS, "bin index in range");
    ASSERT(graphsignal_probe_bin_lower(idx) <= v,
           "bin lower bound must not exceed the value");
    // The top reachable bin (251) covers everything up to UINT64_MAX, so the
    // strict upper-bound check applies below it; bin_lower(252) saturates.
    if (idx < 251u) {
      ASSERT(v < graphsignal_probe_bin_lower(idx + 1),
             "value must be below the next bin's lower bound");
    }
  }

  // Strict monotonicity of bin_lower over the reachable range; bins 252+ are
  // unreachable by bin_index and their lower bound saturates to UINT64_MAX,
  // giving readers a correct upper bound for bin 251.
  for (uint32_t i = 1; i <= 251; i++) {
    ASSERT(graphsignal_probe_bin_lower(i) > graphsignal_probe_bin_lower(i - 1),
           "bin_lower must be strictly monotone over reachable bins");
  }
  ASSERT_EQ(graphsignal_probe_bin_lower(252), UINT64_MAX,
            "bin_lower saturates to UINT64_MAX past the reachable bins");
  ASSERT(graphsignal_probe_bin_lower(252) > graphsignal_probe_bin_lower(251),
         "saturated bound stays above the top reachable bin");

  // Round-trip: every reachable bin's lower bound maps back to that bin.
  for (uint32_t i = 0; i <= 251; i++) {
    ASSERT_EQ(graphsignal_probe_bin_index(graphsignal_probe_bin_lower(i)),
              i, "bin_index(bin_lower(idx)) must round-trip");
  }
}

static void test_histogram_stats() {
  CASE("histogram stats");

  graphsignal_probe_entry* h =
      must_register("t_hist", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0);
  const uint64_t vals[] = {1, 2, 3, 100, 5000, 100000};
  uint64_t expected_sum = 0;
  for (uint64_t v : vals) {
    graphsignal_record(h, v);
    expected_sum += v;
  }

  graphsignal_instrument_data snap;
  ASSERT_EQ(graphsignal_probe_snapshot(h, &snap), 1,
            "snapshot of a HOST histogram must succeed");
  ASSERT_EQ(snap.count, (uint64_t)(sizeof(vals) / sizeof(vals[0])),
            "histogram count");
  ASSERT_EQ(snap.sum, expected_sum, "histogram sum");
  ASSERT_EQ(snap.min, 1ull, "histogram min");
  ASSERT_EQ(snap.max, 100000ull, "histogram max");

  uint64_t bin_total = 0;
  for (uint32_t i = 0; i < GRAPHSIGNAL_PROBE_HIST_BINS; i++) {
    bin_total += snap.bins[i];
  }
  ASSERT_EQ(bin_total, snap.count, "per-bin counts must sum to count");
}

static void test_counter_and_gauge() {
  CASE("counter and gauge semantics");

  graphsignal_probe_entry* c =
      must_register("t_counter", GRAPHSIGNAL_COUNTER, NULL, NULL, 0);
  graphsignal_add(c, 5);
  graphsignal_add(c, 7);
  ASSERT_EQ(c->data->value_bits, 12ull, "counter must accumulate");

  graphsignal_probe_entry* g =
      must_register("t_gauge", GRAPHSIGNAL_GAUGE, NULL, NULL, 0);
  const double gvals[] = {3.14, 0.0, -1.5};
  for (double v : gvals) {
    graphsignal_set(g, v);
    ASSERT(graphsignal_probe_gauge_value(g->data) == v,
           "gauge value must round-trip the exact double");
  }
  // Overwrite semantics: last set wins.
  graphsignal_set(g, 3.14);
  graphsignal_set(g, -1.5);
  ASSERT(graphsignal_probe_gauge_value(g->data) == -1.5,
         "gauge set must overwrite, not accumulate");
}

static void test_registration_idempotence() {
  CASE("registration idempotence and truncation");

  const char* tk[] = {"key"};
  const char* tv1[] = {"v1"};
  const char* tv2[] = {"v2"};

  graphsignal_probe_entry* e1 =
      must_register("t_same", GRAPHSIGNAL_COUNTER, tk, tv1, 1);
  graphsignal_probe_entry* e1b =
      graphsignal_probe_register("t_same", GRAPHSIGNAL_COUNTER, tk, tv1, 1);
  ASSERT(e1b == e1, "same (name, tags) must return the same entry pointer");

  graphsignal_probe_entry* e2 =
      must_register("t_same", GRAPHSIGNAL_COUNTER, tk, tv2, 1);
  ASSERT(e2 != e1, "different tag values must return distinct entries");

  // Metadata copied correctly.
  ASSERT(strcmp(e1->name, "t_same") == 0, "name copied");
  ASSERT_EQ(e1->ntags, 1u, "ntags copied");
  ASSERT(strcmp(e1->tag_keys[0], "key") == 0, "tag key copied");
  ASSERT(strcmp(e1->tag_values[0], "v1") == 0, "tag value copied");
  ASSERT(strcmp(e2->tag_values[0], "v2") == 0, "second tag value copied");
  ASSERT_EQ(e1->type, (uint32_t)GRAPHSIGNAL_COUNTER, "type copied");
  ASSERT_EQ(e1->storage, (uint32_t)GRAPHSIGNAL_STORAGE_HOST, "host storage");
  ASSERT(e1->device_id == -1, "host entries carry device_id -1");

  // Truncation at the ABI limits.
  std::string long_name(200, 'n');
  std::string long_key(40, 'k');
  std::string long_val(200, 'v');
  const char* tk2[] = {long_key.c_str()};
  const char* tv3[] = {long_val.c_str()};
  graphsignal_probe_entry* et = must_register(
      long_name.c_str(), GRAPHSIGNAL_GAUGE, tk2, tv3, 1);
  ASSERT_EQ(strlen(et->name), (size_t)(GRAPHSIGNAL_PROBE_MAX_NAME - 1),
            "over-long name truncated to MAX_NAME-1");
  ASSERT_EQ(strlen(et->tag_keys[0]), (size_t)(GRAPHSIGNAL_PROBE_MAX_TAG_KEY - 1),
            "over-long tag key truncated to MAX_TAG_KEY-1");
  ASSERT_EQ(strlen(et->tag_values[0]),
            (size_t)(GRAPHSIGNAL_PROBE_MAX_TAG_VALUE - 1),
            "over-long tag value truncated to MAX_TAG_VALUE-1");
  ASSERT(strncmp(et->name, long_name.c_str(),
                 GRAPHSIGNAL_PROBE_MAX_NAME - 1) == 0,
         "truncated name must be a prefix of the original");

  // Idempotence survives truncation: re-registering with the same over-long
  // strings returns the same entry instead of creating a duplicate.
  graphsignal_probe_entry* et2 = graphsignal_probe_register(
      long_name.c_str(), GRAPHSIGNAL_GAUGE, tk2, tv3, 1);
  ASSERT(et2 == et, "over-long registration must stay idempotent");

  // The non-allocating lookup the device helpers use before allocating.
  int full = 1;
  ASSERT(graphsignal_probe__find("t_same", tk, tv1, 1, &full) == e1,
         "find must return the registered entry");
  ASSERT_EQ(full, 0, "find must not report full below the cap");
  const uint64_t count_before = graphsignal_probe_reader_count(&__graphsignal_probe_registry_v1);
  ASSERT(graphsignal_probe__find("t_absent", NULL, NULL, 0, &full) == NULL,
         "find must return NULL for an absent instrument");
  ASSERT_EQ(full, 0, "absent below the cap is not full");
  ASSERT_EQ(graphsignal_probe_reader_count(&__graphsignal_probe_registry_v1), count_before,
            "find must never register");
}

static void test_null_safety() {
  CASE("NULL safety");

  graphsignal_record(NULL, 42);
  graphsignal_add(NULL, 42);
  graphsignal_set(NULL, 42.0);

  ASSERT(graphsignal_probe_register(NULL, GRAPHSIGNAL_COUNTER,
                                    NULL, NULL, 0) == NULL,
         "NULL name must return NULL");
  ASSERT(graphsignal_probe_register("", GRAPHSIGNAL_COUNTER,
                                    NULL, NULL, 0) == NULL,
         "empty name must return NULL");
}

static void test_registry_abi() {
  CASE("registry ABI");

  graphsignal_probe_registry* reg = &__graphsignal_probe_registry_v1;
  ASSERT_EQ(reg->magic, (uint32_t)GRAPHSIGNAL_PROBE_ABI_MAGIC,
            "registry magic set after first registration");
  ASSERT_EQ(reg->abi_version, (uint32_t)GRAPHSIGNAL_PROBE_ABI_VERSION,
            "registry abi_version");
  ASSERT(reg->start_ts > 0, "registry start_ts set");

  graphsignal_probe_registry* attached = graphsignal_probe_reader_attach();
  ASSERT(attached != NULL, "reader_attach must find the registry");
  ASSERT(attached == reg, "reader_attach must resolve to the same registry");
  ASSERT_EQ(graphsignal_probe_reader_count(attached), g_expected_registrations,
            "reader_count must match the number of unique registrations");
}

static void test_reader_under_concurrency() {
  CASE("reader under concurrency");

  graphsignal_probe_entry* h =
      must_register("t_conc", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0);

  constexpr int kThreads = 4;
  constexpr uint64_t kPerThread = 100000;
  std::atomic<int> finished{0};

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; t++) {
    workers.emplace_back([h, &finished]() {
      for (uint64_t i = 0; i < kPerThread; i++) {
        graphsignal_record(h, i);
      }
      finished.fetch_add(1, std::memory_order_release);
    });
  }

  // Snapshot repeatedly while the writers run: counts of a sampled sequence
  // must never decrease.
  graphsignal_instrument_data snap;
  uint64_t prev_count = 0;
  while (finished.load(std::memory_order_acquire) < kThreads) {
    ASSERT_EQ(graphsignal_probe_snapshot(h, &snap), 1, "snapshot must succeed");
    ASSERT(snap.count >= prev_count, "snapshot counts must never decrease");
    ASSERT(snap.count <= kThreads * kPerThread,
           "snapshot count must never exceed total records");
    prev_count = snap.count;
    std::this_thread::yield();
  }
  for (std::thread& t : workers) t.join();

  ASSERT_EQ(graphsignal_probe_snapshot(h, &snap), 1, "final snapshot");
  const uint64_t total = kThreads * kPerThread;
  ASSERT_EQ(snap.count, total, "final snapshot count");
  // Each thread records 0..99999: sum per thread = 99999*100000/2.
  const uint64_t expected_sum = (uint64_t)kThreads * (kPerThread * (kPerThread - 1) / 2);
  ASSERT_EQ(snap.sum, expected_sum, "final snapshot sum");
  uint64_t bin_total = 0;
  for (uint32_t i = 0; i < GRAPHSIGNAL_PROBE_HIST_BINS; i++) {
    bin_total += snap.bins[i];
  }
  ASSERT_EQ(bin_total, total, "final per-bin counts must sum to count");
}

static void test_profile_probes() {
  CASE("profile probes");

  graphsignal_probe_entry* p =
      must_register("t_profile", GRAPHSIGNAL_PROFILE, NULL, NULL, 0);
  ASSERT_EQ(p->type, (uint32_t)GRAPHSIGNAL_PROFILE, "profile type copied");
  ASSERT_EQ(p->storage, (uint32_t)GRAPHSIGNAL_STORAGE_HOST,
            "profile entries are host storage");

  graphsignal_profile_data* pd = graphsignal_probe_profile_data(p);
  ASSERT(pd != NULL, "profile entry must expose its profile data block");
  ASSERT_EQ(pd->frame_count, 0ull, "fresh profile has no frames");
  ASSERT_EQ(pd->dropped, 0ull, "fresh profile has no dropped frames");

  // Non-profile entries expose no profile data.
  graphsignal_probe_entry* c = must_register(
      "t_profile_counter", GRAPHSIGNAL_COUNTER, NULL, NULL, 0);
  ASSERT(graphsignal_probe_profile_data(c) == NULL,
         "non-profile entries must expose no profile data");
  ASSERT(graphsignal_probe_profile_data(NULL) == NULL,
         "NULL entry must expose no profile data");

  // find-or-add is idempotent per frame name.
  graphsignal_profile_frame* fa = graphsignal_profile_frame_get(p, "frame.a");
  ASSERT(fa != NULL, "frame_get must add a new frame");
  ASSERT(strcmp(fa->name, "frame.a") == 0, "frame name copied");
  graphsignal_profile_frame* fa2 = graphsignal_profile_frame_get(p, "frame.a");
  ASSERT(fa2 == fa, "same frame name must return the same frame pointer");
  graphsignal_profile_frame* fb = graphsignal_profile_frame_get(p, "frame.b");
  ASSERT(fb != NULL && fb != fa,
         "different frame names must return distinct frames");
  ASSERT_EQ(pd->frame_count, 2ull, "two distinct names -> two frames");

  // Accumulation: direct add and add-by-name land in the same frame.
  graphsignal_profile_add(fa, 5);
  graphsignal_profile_add(fa, 7);
  graphsignal_profile_add_by_name(p, "frame.a", 8);
  ASSERT_EQ(fa->value, 20ull, "frame value must accumulate exactly");
  ASSERT_EQ(fa->samples, 3ull, "every add must count one sample");
  graphsignal_profile_add_by_name(p, "frame.b", 3);
  ASSERT_EQ(fb->value, 3ull, "frames accumulate independently");
  ASSERT_EQ(fb->samples, 1ull, "samples count independently too");

  // record/add/set refuse profile entries: no crash, frames untouched. (The
  // profile block aliases the instrument-data layout, so a stray record would
  // corrupt frame state.)
  graphsignal_record(p, 42);
  graphsignal_add(p, 42);
  graphsignal_set(p, 42.0);
  ASSERT_EQ(pd->frame_count, 2ull, "record/add/set must not change frame_count");
  ASSERT_EQ(fa->value, 20ull, "record/add/set must not change frame values");
  ASSERT_EQ(fb->value, 3ull, "record/add/set must not change frame values");
  ASSERT_EQ(pd->dropped, 0ull, "record/add/set must not change dropped");

  // snapshot refuses profile entries.
  graphsignal_instrument_data snap;
  ASSERT_EQ(graphsignal_probe_snapshot(p, &snap), 0,
            "snapshot must return 0 for profile entries");

  // NULL safety across the profile API.
  graphsignal_profile_add(NULL, 1);
  graphsignal_profile_add_by_name(NULL, "frame.a", 1);
  graphsignal_profile_add_by_name(p, NULL, 1);
  ASSERT(graphsignal_profile_frame_get(NULL, "frame.a") == NULL,
         "frame_get on NULL entry must return NULL");
  ASSERT(graphsignal_profile_frame_get(p, NULL) == NULL,
         "NULL frame name must return NULL");
  ASSERT(graphsignal_profile_frame_get(p, "") == NULL,
         "empty frame name must return NULL");
  ASSERT_EQ(pd->frame_count, 2ull, "NULL-safe calls must not add frames");
  ASSERT_EQ(fa->value, 20ull, "NULL-safe calls must not change values");
  ASSERT_EQ(fa->samples, 3ull, "NULL-safe calls must not count samples");

  // Truncation idempotence: over-long frame names truncate to
  // MAX_FRAME_NAME-1 and keep resolving to the same frame.
  std::string long_name(2 * GRAPHSIGNAL_PROBE_MAX_FRAME_NAME, 'f');
  graphsignal_profile_frame* ft =
      graphsignal_profile_frame_get(p, long_name.c_str());
  ASSERT(ft != NULL, "over-long frame name must register");
  ASSERT_EQ(strlen(ft->name), (size_t)(GRAPHSIGNAL_PROBE_MAX_FRAME_NAME - 1),
            "over-long frame name truncated to MAX_FRAME_NAME-1");
  ASSERT(strncmp(ft->name, long_name.c_str(),
                 GRAPHSIGNAL_PROBE_MAX_FRAME_NAME - 1) == 0,
         "truncated frame name must be a prefix of the original");
  graphsignal_profile_frame* ft2 =
      graphsignal_profile_frame_get(p, long_name.c_str());
  ASSERT(ft2 == ft, "over-long frame lookup must stay idempotent");
  graphsignal_profile_add_by_name(p, long_name.c_str(), 11);
  ASSERT_EQ(ft->value, 11ull, "add_by_name must reach the truncated frame");
  ASSERT_EQ(pd->frame_count, 3ull, "one frame for the over-long name");

  // Frame cap: frames are per-entry, so filling a dedicated probe's frame
  // table pollutes the registry by exactly one entry — no fork needed.
  graphsignal_probe_entry* pc =
      must_register("t_profile_cap", GRAPHSIGNAL_PROFILE, NULL, NULL, 0);
  graphsignal_profile_data* pcd = graphsignal_probe_profile_data(pc);
  ASSERT(pcd != NULL, "cap probe must expose its profile data block");
  char fname[64];
  for (uint32_t i = 0; i < GRAPHSIGNAL_PROBE_MAX_PROFILE_FRAMES; i++) {
    std::snprintf(fname, sizeof(fname), "cap.frame.%u", i);
    ASSERT(graphsignal_profile_frame_get(pc, fname) != NULL,
           "frames up to the cap must register");
  }
  ASSERT_EQ(pcd->frame_count, (uint64_t)GRAPHSIGNAL_PROBE_MAX_PROFILE_FRAMES,
            "frame_count must reach the cap");
  ASSERT_EQ(pcd->dropped, 0ull, "no drops up to the cap");
  ASSERT(graphsignal_profile_frame_get(pc, "cap.frame.extra") == NULL,
         "a new frame past the cap must return NULL");
  ASSERT_EQ(pcd->dropped, 1ull, "the rejected frame must be counted in dropped");
  graphsignal_profile_add_by_name(pc, "cap.frame.extra2", 9);  // safe no-op
  ASSERT_EQ(pcd->dropped, 2ull, "every further rejected frame increments dropped");
  ASSERT_EQ(pcd->frame_count, (uint64_t)GRAPHSIGNAL_PROBE_MAX_PROFILE_FRAMES,
            "frame_count stays at the cap");
  // An existing frame is still found (and writable) at the cap.
  graphsignal_profile_frame* f0 = graphsignal_profile_frame_get(pc, "cap.frame.0");
  ASSERT(f0 != NULL, "existing frame must still be found at the cap");
  graphsignal_profile_add_by_name(pc, "cap.frame.0", 4);
  ASSERT_EQ(f0->value, 4ull, "existing frame must accumulate at the cap");
  ASSERT_EQ(pcd->dropped, 2ull,
            "finding an existing frame must not count as dropped");
}

// The cap test fills the process-global registry, so it runs in a forked
// child: the child's copy-on-write registry is polluted and thrown away with
// _exit; the parent's registry stays clean.
static void test_cardinality_cap_forked() {
  CASE("cardinality cap (forked child)");

  pid_t pid = fork();
  ASSERT(pid >= 0, "fork must succeed");

  if (pid == 0) {
    graphsignal_probe_registry* reg = &__graphsignal_probe_registry_v1;
    const uint64_t count_before = reg->count;
    const uint64_t dropped_before = reg->dropped;
    uint64_t created = 0;
    char name[64];
    graphsignal_probe_entry* e = NULL;
    for (uint64_t i = 0; i < GRAPHSIGNAL_PROBE_MAX_INSTRUMENTS + 8; i++) {
      std::snprintf(name, sizeof(name), "cap_instr_%" PRIu64, i);
      e = graphsignal_probe_register(name, GRAPHSIGNAL_COUNTER,
                                     NULL, NULL, 0);
      if (!e) break;
      created++;
    }
    if (e != NULL) _exit(2);  // never hit the cap
    if (created != GRAPHSIGNAL_PROBE_MAX_INSTRUMENTS - count_before) _exit(3);
    if (reg->count != GRAPHSIGNAL_PROBE_MAX_INSTRUMENTS) _exit(4);
    if (reg->dropped != dropped_before + 1) _exit(5);
    // Every further registration keeps incrementing the dropped counter.
    if (graphsignal_probe_register("cap_extra", GRAPHSIGNAL_COUNTER,
                                   NULL, NULL, 0) != NULL) {
      _exit(6);
    }
    if (reg->dropped != dropped_before + 2) _exit(7);
    // An already-registered instrument is still found at the cap.
    if (graphsignal_probe_register("cap_instr_0", GRAPHSIGNAL_COUNTER,
                                   NULL, NULL, 0) == NULL) {
      _exit(8);
    }
    if (reg->dropped != dropped_before + 2) _exit(9);
    // At the cap the lookup reports full (and counts the drop) for an absent
    // instrument, so device helpers never allocate a block they cannot register.
    int full = 0;
    if (graphsignal_probe__find("cap_absent", NULL, NULL, 0, &full) != NULL || !full) _exit(10);
    if (reg->dropped != dropped_before + 3) _exit(11);
    if (graphsignal_probe__find("cap_instr_0", NULL, NULL, 0, &full) == NULL || full) _exit(12);
    _exit(0);
  }

  int status = 0;
  ASSERT(waitpid(pid, &status, 0) == pid, "waitpid must return the child");
  ASSERT(WIFEXITED(status), "cap-test child must exit normally");
  ASSERT_EQ(WEXITSTATUS(status), 0, "cap-test child must exit 0");

  // The parent's registry is untouched by the child's cap loop.
  ASSERT_EQ(graphsignal_probe_reader_count(&__graphsignal_probe_registry_v1),
            g_expected_registrations,
            "parent registry must be unaffected by the forked cap test");
}

int main() {
  test_bin_math();
  test_histogram_stats();
  test_counter_and_gauge();
  test_registration_idempotence();
  test_null_safety();
  test_registry_abi();
  test_reader_under_concurrency();
  test_profile_probes();
  test_cardinality_cap_forked();

  std::printf("All probe tests passed!\n");
  return 0;
}
