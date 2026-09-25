/*
 * Graphsignal probes — ROCm/HIP device-side instruments.
 *
 * Copyright 2026 Graphsignal, Inc.
 * Licensed under the Apache License, Version 2.0. See probe.h for the
 * license terms and the overall probe model, and probe_cuda.h for the usage
 * pattern (this header is the HIP analog).
 *
 * Compile with hipcc (-std=c++17). Note: graphsignal_gtimer() returns
 * wall_clock64() ticks, not nanoseconds — convert with the device's
 * wall clock frequency (hipDeviceAttributeWallClockRate, kHz) or record raw
 * ticks consistently and interpret downstream.
 */

#ifndef GRAPHSIGNAL_PROBE_ROCM_H
#define GRAPHSIGNAL_PROBE_ROCM_H

#include <graphsignal/probe.h>

#include <hip/hip_runtime.h>

/* Registers a histogram instrument stored in device memory on the current
   HIP device. Idempotent per (name, tags): a repeat call finds the existing
   entry and makes no HIP calls. Returns NULL on cap/allocation failure;
   passing NULL downstream is safe.

   Safe to call from any thread while the application captures a HIP graph:
   the calls run in relaxed capture mode and the block is initialized on a
   private non-blocking stream, so neither the capturing stream nor the null
   stream is touched. A failed call leaves no pending HIP error behind for the
   application's next hipGetLastError(). Blocks are never freed — instruments
   live for the process, and hipFree can synchronize the device mid-capture —
   so a block whose setup failed, or that lost a registration race to another
   thread, stays allocated (2 KB, bounded by those events). */
static inline graphsignal_probe_entry* graphsignal_probe_register_rocm(
        const char* name, const char* const* tag_keys,
        const char* const* tag_vals, size_t ntags) {
    int full = 0;
    graphsignal_probe_entry* existing =
        graphsignal_probe__find(name, tag_keys, tag_vals, ntags, &full);
    if (existing || full || !name || !name[0]) return existing;

    const hipError_t pending = hipPeekAtLastError();
    int device_id = -1;
    graphsignal_instrument_data* dev_data = NULL;
    int ok = hipGetDevice(&device_id) == hipSuccess;
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    if (ok && hipThreadExchangeStreamCaptureMode(&mode) == hipSuccess) {
        hipStream_t stream = NULL;
        ok = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess;
        if (ok) {
            graphsignal_instrument_data init;
            memset(&init, 0, sizeof(init));
            init.min = UINT64_MAX;
            ok = hipMalloc((void**)&dev_data, sizeof(init)) == hipSuccess &&
                 hipMemcpyAsync(dev_data, &init, sizeof(init),
                                hipMemcpyHostToDevice, stream) == hipSuccess &&
                 hipStreamSynchronize(stream) == hipSuccess;
            (void)hipStreamDestroy(stream);
        }
        (void)hipThreadExchangeStreamCaptureMode(&mode);  /* restore the caller's mode */
    } else {
        ok = 0;
    }
    if (!ok) {
        if (pending == hipSuccess) (void)hipGetLastError();  /* clear only our error */
        return NULL;
    }
    return graphsignal_probe_register_storage(
        name, GRAPHSIGNAL_HISTOGRAM, tag_keys, tag_vals, ntags, dev_data, device_id);
}

/* Device pointer to pass into kernels. NULL-safe. */
static inline graphsignal_instrument_data* graphsignal_probe_device_data_rocm(
        const graphsignal_probe_entry* e) {
    if (!e || e->storage != GRAPHSIGNAL_STORAGE_DEVICE) return NULL;
    return e->data;
}

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)

/* Constant-rate device wall clock, in ticks (see header comment). */
__device__ __forceinline__ unsigned long long graphsignal_gtimer() {
    return (unsigned long long)wall_clock64();
}

/* Records one histogram observation from device code. NULL-safe. */
__device__ __forceinline__ void graphsignal_device_record(
        graphsignal_instrument_data* d, unsigned long long value) {
    if (d == NULL) return;
    atomicAdd((unsigned long long*)&d->count, 1ull);
    atomicAdd((unsigned long long*)&d->sum, value);
    unsigned int e = 63u - (unsigned int)__clzll((long long)(value | 1ull));
    unsigned int idx;
    if (value < 4ull) {
        idx = (unsigned int)value;
    } else {
        idx = 4u * e + (unsigned int)((value >> (e - 2u)) & 3ull) - 4u;
        if (idx >= GRAPHSIGNAL_PROBE_HIST_BINS) idx = GRAPHSIGNAL_PROBE_HIST_BINS - 1u;
    }
    atomicAdd((unsigned long long*)&d->bins[idx], 1ull);
    atomicMin((unsigned long long*)&d->min, value);
    atomicMax((unsigned long long*)&d->max, value);
}

#define GRAPHSIGNAL_RECORD(data, value) graphsignal_device_record((data), (value))

#endif /* __HIPCC__ */

#endif /* GRAPHSIGNAL_PROBE_ROCM_H */
