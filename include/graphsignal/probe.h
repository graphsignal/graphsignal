/*
 * Graphsignal probes — in-process metric instruments for GPU and host code.
 *
 * Copyright 2026 Graphsignal, Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Single vendorable header, no dependencies beyond libc. Requires C++17 or
 * later (g++, clang, nvcc, hipcc). Include it in your library or application,
 * register instruments, and record values; when the process runs under the
 * Graphsignal profiler, the injected library discovers the registry, reads
 * the instruments every second, and publishes them to the local /signals
 * endpoint.
 *
 *   #include <graphsignal/probe.h>
 *
 *   static graphsignal_probe_entry* g_batch_ns;
 *
 *   void init() {
 *       g_batch_ns = graphsignal_probe_register(
 *           "myengine_batch_duration_nanoseconds", GRAPHSIGNAL_HISTOGRAM,
 *           NULL, NULL, 0);
 *   }
 *
 *   void step() {
 *       uint64_t t0 = graphsignal_now_ns();
 *       ...work...
 *       graphsignal_record(g_batch_ns, graphsignal_now_ns() - t0);
 *   }
 *
 * The record path is lock-free: a handful of relaxed 64-bit atomic adds, no
 * allocation, no branches beyond a NULL check and the bin index math. Values
 * are unsigned 64-bit integers (nanoseconds, bytes, counts). Instruments are
 * cumulative forever; readers snapshot and diff.
 *
 * CUDA / HIP kernels: see <graphsignal/probe_cuda.h> and
 * <graphsignal/probe_rocm.h> for device-storage instruments recordable from
 * device code.
 *
 * Precision notes: histogram bins are log-linear with 4 sub-bins per power of
 * two (worst-case bucket width 25%); `sum` accumulates exactly in uint64 (a
 * duration instrument overflows after ~584 years of accumulated time).
 */

#ifndef GRAPHSIGNAL_PROBE_H
#define GRAPHSIGNAL_PROBE_H

#if !defined(__cplusplus) || __cplusplus < 201703L
#error "graphsignal/probe.h requires C++17 (-std=c++17)"
#endif

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

/* ---- limits (fixed: part of the registry ABI) --------------------------- */

#define GRAPHSIGNAL_PROBE_ABI_MAGIC 0x47535052u /* "GSPR" */
#define GRAPHSIGNAL_PROBE_ABI_VERSION 2u
#define GRAPHSIGNAL_PROBE_REGISTRY_SYMBOL "__graphsignal_probe_registry_v1"

#define GRAPHSIGNAL_PROBE_MAX_INSTRUMENTS 4096u
#define GRAPHSIGNAL_PROBE_HIST_BINS 256u
#define GRAPHSIGNAL_PROBE_MAX_NAME 128u
#define GRAPHSIGNAL_PROBE_MAX_TAGS 4u
#define GRAPHSIGNAL_PROBE_MAX_TAG_KEY 32u
#define GRAPHSIGNAL_PROBE_MAX_TAG_VALUE 128u
#define GRAPHSIGNAL_PROBE_MAX_PROFILE_FRAMES 250u
#define GRAPHSIGNAL_PROBE_MAX_FRAME_NAME 128u

typedef enum graphsignal_instrument_type {
    GRAPHSIGNAL_GAUGE = 0,
    GRAPHSIGNAL_COUNTER = 1,
    GRAPHSIGNAL_HISTOGRAM = 2,
    GRAPHSIGNAL_PROFILE = 3
} graphsignal_instrument_type;

typedef enum graphsignal_instrument_storage {
    GRAPHSIGNAL_STORAGE_HOST = 0,
    GRAPHSIGNAL_STORAGE_DEVICE = 1
} graphsignal_instrument_storage;

/* ---- instrument data (the hot block; also the device-memory layout) ----- */

typedef struct graphsignal_instrument_data {
    uint64_t count;      /* histogram observations */
    uint64_t sum;        /* sum of observed values */
    uint64_t min;        /* UINT64_MAX until first observation */
    uint64_t max;        /* 0 until first observation */
    uint64_t value_bits; /* gauge: double bit pattern; counter: total */
    uint64_t bins[GRAPHSIGNAL_PROBE_HIST_BINS];
} graphsignal_instrument_data;

/* Profile instruments map frame names to cumulative counter values (host
   storage only). A PROFILE entry's `data` pointer aliases this layout. */

typedef struct graphsignal_profile_frame {
    char name[GRAPHSIGNAL_PROBE_MAX_FRAME_NAME];
    uint64_t value;
    uint64_t samples; /* recordings that contributed to `value` */
} graphsignal_profile_frame;

typedef struct graphsignal_profile_data {
    uint64_t frame_count; /* published with release ordering */
    uint64_t dropped;     /* frames dropped past the cap */
    graphsignal_profile_frame frames[GRAPHSIGNAL_PROBE_MAX_PROFILE_FRAMES];
} graphsignal_profile_data;

/* ---- registry entry (cold metadata) -------------------------------------- */

typedef struct graphsignal_probe_entry {
    uint32_t type;      /* graphsignal_instrument_type */
    uint32_t storage;   /* graphsignal_instrument_storage */
    int32_t device_id;  /* device ordinal for DEVICE storage, else -1 */
    uint32_t ntags;
    char name[GRAPHSIGNAL_PROBE_MAX_NAME];
    char tag_keys[GRAPHSIGNAL_PROBE_MAX_TAGS][GRAPHSIGNAL_PROBE_MAX_TAG_KEY];
    char tag_values[GRAPHSIGNAL_PROBE_MAX_TAGS][GRAPHSIGNAL_PROBE_MAX_TAG_VALUE];
    /* HOST storage: host pointer to graphsignal_instrument_data.
       DEVICE storage: device pointer (read with cudaMemcpy/hipMemcpy). */
    graphsignal_instrument_data* data;
} graphsignal_probe_entry;

typedef struct graphsignal_probe_registry {
    uint32_t magic;
    uint32_t abi_version;
    uint64_t start_ts;    /* ns since epoch of the first registration */
    uint64_t dropped;     /* registrations dropped past the instrument cap */
    uint64_t reg_lock;    /* registration spinlock; never taken on record */
    uint64_t count;       /* valid entries; published with release ordering */
    graphsignal_probe_entry entries[GRAPHSIGNAL_PROBE_MAX_INSTRUMENTS];
} graphsignal_probe_registry;

/* One registry per process. Default visibility so the dynamic linker unifies
   vendored copies across shared objects and readers can dlsym it. */
extern "C" {
#if defined(_WIN32)
inline graphsignal_probe_registry __graphsignal_probe_registry_v1;
#else
__attribute__((visibility("default")))
inline graphsignal_probe_registry __graphsignal_probe_registry_v1;
#endif
}

/* ---- small utilities ------------------------------------------------------ */

static inline uint64_t graphsignal_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Log-linear bin index: exact for 0..3, then 4 sub-bins per power of two
   (worst-case bucket width 25%). Covers the full uint64 range in 256 bins. */
static inline uint32_t graphsignal_probe_bin_index(uint64_t v) {
    if (v < 4) return (uint32_t)v;
    uint32_t e = 63u - (uint32_t)__builtin_clzll(v);
    uint32_t idx = 4u * e + (uint32_t)((v >> (e - 2)) & 3u) - 4u;
    return idx < GRAPHSIGNAL_PROBE_HIST_BINS ? idx : GRAPHSIGNAL_PROBE_HIST_BINS - 1u;
}

/* Inclusive lower bound of a bin; monotonically increasing in idx.
   bin_index never produces indices above 251, so bins 252+ are unreachable;
   their lower bound saturates to UINT64_MAX (also the correct upper bound
   for bin 251 when a reader computes bin ranges as [lower(i), lower(i+1))). */
static inline uint64_t graphsignal_probe_bin_lower(uint32_t idx) {
    if (idx < 4) return (uint64_t)idx;
    if (idx >= 252) return UINT64_MAX;
    return (uint64_t)(4u + (idx & 3u)) << (idx / 4u - 1u);
}

/* ---- registration (cold path; may allocate, takes the registry lock) ----- */

static inline void graphsignal_probe__lock(graphsignal_probe_registry* reg) {
    while (__atomic_exchange_n(&reg->reg_lock, 1ull, __ATOMIC_ACQUIRE) != 0ull) {
    }
}

static inline void graphsignal_probe__unlock(graphsignal_probe_registry* reg) {
    __atomic_store_n(&reg->reg_lock, 0ull, __ATOMIC_RELEASE);
}

static inline void graphsignal_probe__copy_str(char* dst, size_t cap, const char* src) {
    if (!src) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

/* Stored strings are truncated at registration; matching compares the stored
   string against the caller's string truncated the same way, so registration
   stays idempotent for over-limit inputs. */
static inline int graphsignal_probe__str_matches(const char* stored,
                                                 const char* caller, size_t cap) {
    size_t n = strnlen(caller, cap - 1);
    return strncmp(stored, caller, n) == 0 && stored[n] == '\0';
}

static inline int graphsignal_probe__entry_matches(
        const graphsignal_probe_entry* e, const char* name,
        const char* const* tag_keys, const char* const* tag_vals, size_t ntags) {
    if (!graphsignal_probe__str_matches(e->name, name, GRAPHSIGNAL_PROBE_MAX_NAME)) return 0;
    if (e->ntags != (uint32_t)ntags) return 0;
    for (size_t i = 0; i < ntags; i++) {
        if (!graphsignal_probe__str_matches(e->tag_keys[i], tag_keys[i],
                                            GRAPHSIGNAL_PROBE_MAX_TAG_KEY)) return 0;
        if (!graphsignal_probe__str_matches(e->tag_values[i], tag_vals[i],
                                            GRAPHSIGNAL_PROBE_MAX_TAG_VALUE)) return 0;
    }
    return 1;
}

static inline size_t graphsignal_probe__clamp_ntags(
        const char* const* tag_keys, const char* const* tag_vals, size_t ntags) {
    if (ntags > GRAPHSIGNAL_PROBE_MAX_TAGS) ntags = GRAPHSIGNAL_PROBE_MAX_TAGS;
    if (ntags > 0 && (!tag_keys || !tag_vals)) ntags = 0;
    return ntags;
}

/* Caller holds the registry lock. */
static inline graphsignal_probe_entry* graphsignal_probe__find_locked(
        graphsignal_probe_registry* reg, const char* name,
        const char* const* tag_keys, const char* const* tag_vals, size_t ntags) {
    for (uint64_t i = 0; i < reg->count; i++) {
        if (graphsignal_probe__entry_matches(&reg->entries[i], name,
                                             tag_keys, tag_vals, ntags)) {
            return &reg->entries[i];
        }
    }
    return NULL;
}

/* Finds an already-registered instrument without registering or allocating
   anything; device helpers call it first so a repeat registration makes no
   device API calls. Sets *full (and counts the drop) when the instrument is
   absent and the registry has no room for it, so callers skip allocating. */
static inline graphsignal_probe_entry* graphsignal_probe__find(
        const char* name, const char* const* tag_keys, const char* const* tag_vals,
        size_t ntags, int* full) {
    *full = 0;
    if (!name || !name[0]) return NULL;
    ntags = graphsignal_probe__clamp_ntags(tag_keys, tag_vals, ntags);

    graphsignal_probe_registry* reg = &__graphsignal_probe_registry_v1;
    graphsignal_probe__lock(reg);
    graphsignal_probe_entry* e =
        graphsignal_probe__find_locked(reg, name, tag_keys, tag_vals, ntags);
    if (!e && reg->count >= GRAPHSIGNAL_PROBE_MAX_INSTRUMENTS) {
        reg->dropped++;
        *full = 1;
    }
    graphsignal_probe__unlock(reg);
    return e;
}

static inline graphsignal_probe_entry* graphsignal_probe__register_impl(
        const char* name, graphsignal_instrument_type type,
        const char* const* tag_keys, const char* const* tag_vals, size_t ntags,
        graphsignal_instrument_data* device_data, int device_id) {
    if (!name || !name[0]) return NULL;
    ntags = graphsignal_probe__clamp_ntags(tag_keys, tag_vals, ntags);

    graphsignal_probe_registry* reg = &__graphsignal_probe_registry_v1;
    graphsignal_probe__lock(reg);

    if (reg->magic != GRAPHSIGNAL_PROBE_ABI_MAGIC) {
        reg->magic = GRAPHSIGNAL_PROBE_ABI_MAGIC;
        reg->abi_version = GRAPHSIGNAL_PROBE_ABI_VERSION;
        reg->start_ts = graphsignal_now_ns();
    }

    graphsignal_probe_entry* existing =
        graphsignal_probe__find_locked(reg, name, tag_keys, tag_vals, ntags);
    if (existing) {
        graphsignal_probe__unlock(reg);
        return existing;
    }

    uint64_t count = reg->count;
    if (count >= GRAPHSIGNAL_PROBE_MAX_INSTRUMENTS) {
        reg->dropped++;
        graphsignal_probe__unlock(reg);
        return NULL;
    }

    graphsignal_instrument_data* data = device_data;
    if (data == NULL) {
        if (type == GRAPHSIGNAL_PROFILE) {
            data = (graphsignal_instrument_data*)
                calloc(1, sizeof(graphsignal_profile_data));
            if (!data) {
                reg->dropped++;
                graphsignal_probe__unlock(reg);
                return NULL;
            }
        } else {
            data = (graphsignal_instrument_data*)
                calloc(1, sizeof(graphsignal_instrument_data));
            if (!data) {
                reg->dropped++;
                graphsignal_probe__unlock(reg);
                return NULL;
            }
            data->min = UINT64_MAX;
        }
    }

    graphsignal_probe_entry* e = &reg->entries[count];
    e->type = (uint32_t)type;
    e->storage = device_data ? GRAPHSIGNAL_STORAGE_DEVICE : GRAPHSIGNAL_STORAGE_HOST;
    e->device_id = device_data ? device_id : -1;
    e->ntags = (uint32_t)ntags;
    graphsignal_probe__copy_str(e->name, GRAPHSIGNAL_PROBE_MAX_NAME, name);
    for (size_t i = 0; i < ntags; i++) {
        graphsignal_probe__copy_str(e->tag_keys[i], GRAPHSIGNAL_PROBE_MAX_TAG_KEY, tag_keys[i]);
        graphsignal_probe__copy_str(e->tag_values[i], GRAPHSIGNAL_PROBE_MAX_TAG_VALUE, tag_vals[i]);
    }
    e->data = data;

    /* Publish the entry after its fields are complete. */
    __atomic_store_n(&reg->count, count + 1, __ATOMIC_RELEASE);

    graphsignal_probe__unlock(reg);
    return e;
}

/* Registers (or finds) a host instrument. Idempotent per (name, tags): repeat
   calls return the same entry. Returns NULL when the name is missing, the
   instrument cap is reached (counted in registry->dropped), or allocation
   fails — recording on NULL is a safe no-op, so call sites need no check.
   Strings are copied and silently truncated to the ABI limits; at most
   GRAPHSIGNAL_PROBE_MAX_TAGS tags are kept. */
static inline graphsignal_probe_entry* graphsignal_probe_register(
        const char* name, graphsignal_instrument_type type,
        const char* const* tag_keys, const char* const* tag_vals, size_t ntags) {
    return graphsignal_probe__register_impl(
        name, type, tag_keys, tag_vals, ntags, NULL, -1);
}

/* Registers a DEVICE-storage instrument around a caller-allocated data block
   (device memory, zero-initialized with min=UINT64_MAX; probe_cuda.h /
   probe_rocm.h do the allocation for you). The block must stay allocated for
   the life of the process: the profiler copies it every interval. Idempotent
   like graphsignal_probe_register; when the entry already exists, the passed
   block is NOT adopted — check the returned entry's `data` to free yours. */
static inline graphsignal_probe_entry* graphsignal_probe_register_storage(
        const char* name, graphsignal_instrument_type type,
        const char* const* tag_keys, const char* const* tag_vals, size_t ntags,
        graphsignal_instrument_data* data, int device_id) {
    if (!data || type == GRAPHSIGNAL_PROFILE) return NULL;
    return graphsignal_probe__register_impl(
        name, type, tag_keys, tag_vals, ntags, data, device_id);
}

/* ---- recording (hot path: lock-free, allocation-free) -------------------- */

/* Histogram: observe one value. ~5 relaxed atomic RMWs. */
static inline void graphsignal_record(graphsignal_probe_entry* e, uint64_t value) {
    if (!e || e->storage != GRAPHSIGNAL_STORAGE_HOST ||
        e->type == GRAPHSIGNAL_PROFILE) return;
    graphsignal_instrument_data* d = e->data;
    __atomic_fetch_add(&d->count, 1ull, __ATOMIC_RELAXED);
    __atomic_fetch_add(&d->sum, value, __ATOMIC_RELAXED);
    __atomic_fetch_add(&d->bins[graphsignal_probe_bin_index(value)], 1ull,
                       __ATOMIC_RELAXED);
    uint64_t cur = __atomic_load_n(&d->min, __ATOMIC_RELAXED);
    while (value < cur &&
           !__atomic_compare_exchange_n(&d->min, &cur, value, 1,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
    cur = __atomic_load_n(&d->max, __ATOMIC_RELAXED);
    while (value > cur &&
           !__atomic_compare_exchange_n(&d->max, &cur, value, 1,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
}

/* Counter: add to the cumulative total. One relaxed atomic add. */
static inline void graphsignal_add(graphsignal_probe_entry* e, uint64_t delta) {
    if (!e || e->storage != GRAPHSIGNAL_STORAGE_HOST ||
        e->type == GRAPHSIGNAL_PROFILE) return;
    __atomic_fetch_add(&e->data->value_bits, delta, __ATOMIC_RELAXED);
}

/* Gauge: set the current value. One relaxed store. */
static inline void graphsignal_set(graphsignal_probe_entry* e, double value) {
    if (!e || e->storage != GRAPHSIGNAL_STORAGE_HOST ||
        e->type == GRAPHSIGNAL_PROFILE) return;
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    __atomic_store_n(&e->data->value_bits, bits, __ATOMIC_RELAXED);
}

static inline double graphsignal_probe_gauge_value(const graphsignal_instrument_data* d) {
    uint64_t bits = d->value_bits;
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* ---- profile instruments -------------------------------------------------- */

/* A PROFILE entry's data block. NULL for other types. */
static inline graphsignal_profile_data* graphsignal_probe_profile_data(
        const graphsignal_probe_entry* e) {
    if (!e || e->type != GRAPHSIGNAL_PROFILE ||
        e->storage != GRAPHSIGNAL_STORAGE_HOST || !e->data) {
        return NULL;
    }
    return (graphsignal_profile_data*)(void*)e->data;
}

/* Finds or adds a frame by name (cold path: takes the registry lock on a
   miss). Returns NULL for non-profile entries, missing names, or past the
   GRAPHSIGNAL_PROBE_MAX_PROFILE_FRAMES cap (counted in the block's `dropped`);
   adding on NULL is a safe no-op. Cache the returned frame for hot paths. */
static inline graphsignal_profile_frame* graphsignal_profile_frame_get(
        graphsignal_probe_entry* e, const char* frame_name) {
    graphsignal_profile_data* p = graphsignal_probe_profile_data(e);
    if (!p || !frame_name || !frame_name[0]) return NULL;

    uint64_t count = __atomic_load_n(&p->frame_count, __ATOMIC_ACQUIRE);
    for (uint64_t i = 0; i < count; i++) {
        if (graphsignal_probe__str_matches(p->frames[i].name, frame_name,
                                           GRAPHSIGNAL_PROBE_MAX_FRAME_NAME)) {
            return &p->frames[i];
        }
    }

    graphsignal_probe_registry* reg = &__graphsignal_probe_registry_v1;
    graphsignal_probe__lock(reg);
    count = p->frame_count;
    for (uint64_t i = 0; i < count; i++) {
        if (graphsignal_probe__str_matches(p->frames[i].name, frame_name,
                                           GRAPHSIGNAL_PROBE_MAX_FRAME_NAME)) {
            graphsignal_probe__unlock(reg);
            return &p->frames[i];
        }
    }
    if (count >= GRAPHSIGNAL_PROBE_MAX_PROFILE_FRAMES) {
        __atomic_fetch_add(&p->dropped, 1ull, __ATOMIC_RELAXED);
        graphsignal_probe__unlock(reg);
        return NULL;
    }
    graphsignal_profile_frame* f = &p->frames[count];
    graphsignal_probe__copy_str(f->name, GRAPHSIGNAL_PROBE_MAX_FRAME_NAME, frame_name);
    __atomic_store_n(&p->frame_count, count + 1, __ATOMIC_RELEASE);
    graphsignal_probe__unlock(reg);
    return f;
}

/* Adds to a frame's cumulative value and counts the recording. Two relaxed
   atomic adds; NULL-safe. */
static inline void graphsignal_profile_add(graphsignal_profile_frame* f,
                                           uint64_t delta) {
    if (!f) return;
    __atomic_fetch_add(&f->value, delta, __ATOMIC_RELAXED);
    __atomic_fetch_add(&f->samples, 1ull, __ATOMIC_RELAXED);
}

/* Convenience: find-or-add the frame, then add. Involves a frame-table scan —
   fine for moderate rates; cache the frame via graphsignal_profile_frame_get
   for hot paths. */
static inline void graphsignal_profile_add_by_name(graphsignal_probe_entry* e,
                                                   const char* frame_name,
                                                   uint64_t delta) {
    graphsignal_profile_add(graphsignal_profile_frame_get(e, frame_name), delta);
}

/* ---- reader API (in-process readers, e.g. the profiler's injected libs) -- */

/* Finds the process's registry. Returns NULL until some code in the process
   has registered a probe (or when no probe.h user is present at all). */
static inline graphsignal_probe_registry* graphsignal_probe_reader_attach(void) {
    graphsignal_probe_registry* reg = NULL;
#if defined(__unix__) || defined(__APPLE__)
    void* sym = dlsym(RTLD_DEFAULT, GRAPHSIGNAL_PROBE_REGISTRY_SYMBOL);
    if (sym) reg = (graphsignal_probe_registry*)sym;
#endif
    /* dlsym can miss the registry (hidden-visibility or stripped binaries);
       the local instance still covers probes registered in this link unit. */
    if (!reg || reg->magic != GRAPHSIGNAL_PROBE_ABI_MAGIC) {
        reg = &__graphsignal_probe_registry_v1;
    }
    if (reg->magic != GRAPHSIGNAL_PROBE_ABI_MAGIC ||
        reg->abi_version != GRAPHSIGNAL_PROBE_ABI_VERSION) {
        return NULL;
    }
    return reg;
}

/* Number of published entries (acquire pairs with registration's release). */
static inline uint64_t graphsignal_probe_reader_count(const graphsignal_probe_registry* reg) {
    return __atomic_load_n(&reg->count, __ATOMIC_ACQUIRE);
}

/* Copies a HOST instrument's live data into `out`. Reads are relaxed and the
   copy is not a consistent point-in-time cut across fields — instruments are
   cumulative, so a slightly torn snapshot only shifts samples into the next
   read. Returns 0 for DEVICE storage (read those with the platform memcpy). */
static inline int graphsignal_probe_snapshot(const graphsignal_probe_entry* e,
                                             graphsignal_instrument_data* out) {
    if (!e || e->storage != GRAPHSIGNAL_STORAGE_HOST || !e->data ||
        e->type == GRAPHSIGNAL_PROFILE) return 0;
    const graphsignal_instrument_data* d = e->data;
    out->count = __atomic_load_n(&d->count, __ATOMIC_RELAXED);
    out->sum = __atomic_load_n(&d->sum, __ATOMIC_RELAXED);
    out->min = __atomic_load_n(&d->min, __ATOMIC_RELAXED);
    out->max = __atomic_load_n(&d->max, __ATOMIC_RELAXED);
    out->value_bits = __atomic_load_n(&d->value_bits, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < GRAPHSIGNAL_PROBE_HIST_BINS; i++) {
        out->bins[i] = __atomic_load_n(&d->bins[i], __ATOMIC_RELAXED);
    }
    return 1;
}

#endif /* GRAPHSIGNAL_PROBE_H */
