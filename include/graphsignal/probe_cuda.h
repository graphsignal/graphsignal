/*
 * Graphsignal probes — CUDA device-side instruments.
 *
 * Copyright 2026 Graphsignal, Inc.
 * Licensed under the Apache License, Version 2.0. See probe.h for the
 * license terms and the overall probe model.
 *
 * Compile with nvcc (-std=c++17). Device-storage instruments live in GPU
 * memory and are recorded from inside kernels; the profiler's injected
 * library reads them out.
 *
 *   #include <graphsignal/probe_cuda.h>
 *
 *   static graphsignal_probe_entry* g_tile_ns;
 *
 *   void init() {
 *       g_tile_ns = graphsignal_probe_register_cuda(
 *           "mykernels_tile_duration_nanoseconds", NULL, NULL, 0);
 *   }
 *
 *   __global__ void my_kernel(..., graphsignal_instrument_data* tile_ns) {
 *       unsigned long long t0 = graphsignal_gtimer();
 *       ...tile work...
 *       if (threadIdx.x == 0) {  // one recording lane per block
 *           GRAPHSIGNAL_RECORD(tile_ns, graphsignal_gtimer() - t0);
 *       }
 *   }
 *
 *   // launch: my_kernel<<<...>>>(..., graphsignal_probe_device_data(g_tile_ns));
 *
 * The caller picks the recording lane (thread 0 of a block, one lane per
 * warp, ...) — every thread that executes GRAPHSIGNAL_RECORD records one
 * sample. The record path is a few relaxed atomicAdds in device memory.
 */

#ifndef GRAPHSIGNAL_PROBE_CUDA_H
#define GRAPHSIGNAL_PROBE_CUDA_H

#include <graphsignal/probe.h>

#include <cuda_runtime.h>

/* Registers a histogram instrument stored in device memory on the current
   CUDA device. Idempotent per (name, tags): a repeat call finds the existing
   entry and makes no CUDA calls. Returns NULL on cap/allocation failure;
   passing NULL downstream is safe. Host-side graphsignal_record() is a no-op
   for device instruments — record from device code.

   Safe to call from any thread while the application captures a CUDA graph:
   the calls run in relaxed capture mode and the block is initialized on a
   private non-blocking stream, so neither the capturing stream nor the legacy
   default stream is touched. A failed call leaves no pending CUDA error behind
   for the application's next cudaGetLastError(). Blocks are never freed —
   instruments live for the process, and cudaFree can synchronize the device
   mid-capture — so a block whose setup failed, or that lost a registration
   race to another thread, stays allocated (2 KB, bounded by those events). */
static inline graphsignal_probe_entry* graphsignal_probe_register_cuda(
        const char* name, const char* const* tag_keys,
        const char* const* tag_vals, size_t ntags) {
    int full = 0;
    graphsignal_probe_entry* existing =
        graphsignal_probe__find(name, tag_keys, tag_vals, ntags, &full);
    if (existing || full || !name || !name[0]) return existing;

    const cudaError_t pending = cudaPeekAtLastError();
    int device_id = -1;
    graphsignal_instrument_data* dev_data = NULL;
    int ok = cudaGetDevice(&device_id) == cudaSuccess;
    enum cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    if (ok && cudaThreadExchangeStreamCaptureMode(&mode) == cudaSuccess) {
        cudaStream_t stream = NULL;
        ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
        if (ok) {
            graphsignal_instrument_data init;
            memset(&init, 0, sizeof(init));
            init.min = UINT64_MAX;
            ok = cudaMalloc((void**)&dev_data, sizeof(init)) == cudaSuccess &&
                 cudaMemcpyAsync(dev_data, &init, sizeof(init),
                                 cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                 cudaStreamSynchronize(stream) == cudaSuccess;
            cudaStreamDestroy(stream);
        }
        cudaThreadExchangeStreamCaptureMode(&mode);  /* restore the caller's mode */
    } else {
        ok = 0;
    }
    if (!ok) {
        if (pending == cudaSuccess) (void)cudaGetLastError();  /* clear only our error */
        return NULL;
    }
    return graphsignal_probe_register_storage(
        name, GRAPHSIGNAL_HISTOGRAM, tag_keys, tag_vals, ntags, dev_data, device_id);
}

/* Device pointer to pass into kernels. NULL-safe. */
static inline graphsignal_instrument_data* graphsignal_probe_device_data(
        const graphsignal_probe_entry* e) {
    if (!e || e->storage != GRAPHSIGNAL_STORAGE_DEVICE) return NULL;
    return e->data;
}

#if defined(__CUDACC__)

/* Globally synchronized nanosecond clock, readable from any thread. */
__device__ __forceinline__ unsigned long long graphsignal_gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
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

#endif /* __CUDACC__ */

#endif /* GRAPHSIGNAL_PROBE_CUDA_H */
