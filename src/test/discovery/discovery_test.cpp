// Probes compiled into a main executable that is NOT linked with -rdynamic:
// the executable exports no symbols, so dlsym cannot see its registry and the
// writer must find it by scanning the executable's writable segments. Linux
// only (dl_iterate_phdr). Built and run by `make -f Makefile.cupti
// test-discovery`; the writer lives in a separate shared object loaded with
// RTLD_LOCAL, like the real injection library.
#include <graphsignal/probe.h>

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#define ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
      std::abort(); \
    } \
  } while (0)

static std::string read_file(const char* path) {
  std::string out;
  FILE* f = std::fopen(path, "r");
  if (!f) return out;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

int main(int argc, char** argv) {
  const char* lib = argc > 1 ? argv[1] : "build/libdiscovery_lib.so";

  graphsignal_probe_entry* e =
      graphsignal_probe_register("discovery_exe_probe", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0);
  ASSERT(e != NULL, "executable probe registration");
  graphsignal_record(e, 42);

  // The scenario is real only if the executable's registry is invisible.
  ASSERT(dlsym(RTLD_DEFAULT, GRAPHSIGNAL_PROBE_REGISTRY_SYMBOL) == NULL,
         "test executable must not export the registry (link without -rdynamic)");

  void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
  ASSERT(h != NULL, "dlopen of the writer library");
  typedef int (*register_fn)();
  typedef int (*write_fn)(const char*, char*, size_t);
  register_fn reg = (register_fn)dlsym(h, "gs_discovery_register_lib_probe");
  write_fn write = (write_fn)dlsym(h, "gs_discovery_write");
  ASSERT(reg && write, "writer library entry points");
  ASSERT(reg() == 0, "library probe registration");

  char tmpl[] = "/tmp/gs_discovery_XXXXXX";
  ASSERT(mkdtemp(tmpl) != NULL, "mkdtemp");
  char path[512];
  ASSERT(write(tmpl, path, sizeof(path)) == 0, "writer init/write");
  const std::string j = read_file(path);
  ASSERT(!j.empty(), "the writer must produce a file");

  ASSERT(j.find("\"name\":\"discovery_exe_probe\"") != std::string::npos,
         "the executable's probe must be found without -rdynamic");
  ASSERT(j.find("\"count\":1,\"sum\":42,") != std::string::npos,
         "the executable's probe must carry its values");
  ASSERT(j.find("\"name\":\"discovery_lib_probe\"") != std::string::npos,
         "the library registry must still be serialized alongside");
  const size_t first = j.find("\"name\":\"discovery_exe_probe\"");
  ASSERT(j.find("\"name\":\"discovery_exe_probe\"", first + 1) == std::string::npos,
         "each registry must be serialized once");

  std::printf("All discovery tests passed!\n");
  return 0;
}
