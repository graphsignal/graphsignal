// Stand-in for an injected profiling library: the writer compiled into a
// shared object that the test executable loads with RTLD_LOCAL (libcuda
// loads CUDA_INJECTION64_PATH the same way).
#include "metrics_writer.h"

#include <cstring>

extern "C" {

// A probe registered from library code: it lands in the registry shared
// libraries unify on, not in the executable's own copy.
__attribute__((visibility("default"))) int gs_discovery_register_lib_probe() {
  graphsignal_probe_entry* e =
      graphsignal_probe_register("discovery_lib_probe", GRAPHSIGNAL_COUNTER, NULL, NULL, 0);
  graphsignal_add(e, 7);
  return e ? 0 : 1;
}

// One write into `dir`; copies the file path out.
__attribute__((visibility("default"))) int gs_discovery_write(const char* dir, char* path,
                                                             size_t cap) {
  graphsignal::MetricsWriter* w =
      graphsignal::MetricsWriter::init("discovery", dir, 3600ull * 1000000000ull, false);
  if (!w) return 1;
  w->write_now(false);
  std::snprintf(path, cap, "%s", w->file_path().c_str());
  w->shutdown();
  return 0;
}

}  // extern "C"
