"""Synthetic `/signals` payloads for the top view tests.

`build_payload(t)` returns the payload the watcher would serve `t` seconds
after the instance started, with every cumulative value growing linearly in
`t` so that any two payloads diff to known per-second rates:

  kernels: flash_fwd 314 ms/s (1,980 calls/s), gemm 221+10 ms/s across two
           processes, rms_norm 49 ms/s; memcpy 20 ms/s at 5 MB/s; sync 60 ms/s
  engine:  prompt 11,200 tok/s, generation 1,840 tok/s; TTFT bins fill at
           [10, 30, 5, 1]/s over bounds [0.1, 0.2, 0.5, 1.0]
"""

START_NS = 1_700_000_000 * 1_000_000_000
NS = 1_000_000_000

FLASH_FWD = ('_ZN5flash16flash_fwd_kernelINS_21Flash_fwd_kernel_traitsILi128ELi128ELi64ELi4ELb0ELb0E'
             'N7cutlass10bfloat16_tENS_15Flash_kernel_traitsILi128ELi128ELi64ELi4ES3_EEEELb0ELb0ELb1E'
             'Lb1ELb0ELb1ELb1EEEv16Flash_fwd_params')
GEMM = ('sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_'
        'execute_segment_k_off_kernel__5x_cublas')
RMS_NORM = '_ZN4vllm15rms_norm_kernelIN3c108BFloat16EEEvPT_PKS3_S6_fii'
GRAPH_SIG = '3f2a9c1e00112233'

ROOT_PID = '41822'
CHILD_PID = '41900'
DEVICE_UUID = 'GPU-aaaa'
DEVICE_TAGS = {'device.uuid': DEVICE_UUID, 'device.bus_id': '0000:01:00.0'}


def _metric(name, mtype, stats, tags=None, ts=None):
    return {'name': name, 'type': mtype, 'tags': dict(tags or {}), 'stats': stats,
            'updated_ns': ts}


def _profile(name, frames, pid, ts):
    return _metric(name, 'profile', {
        'frames': sorted(
            [{'name': f, 'value': v, 'samples': s} for f, (v, s) in frames.items()],
            key=lambda fr: fr['value'], reverse=True)},
        tags={'process.pid': pid}, ts=ts)


def _histogram(name, bins, per_second_counts, t, ts, tags=None):
    counts = [int(c * t) for c in per_second_counts]
    count = sum(counts)
    total = sum(b * c for b, c in zip(bins, counts)) * 0.8
    return _metric(name, 'histogram', {
        'count': count, 'sum': total, 'min': None, 'max': None,
        'mean': (total / count) if count else None,
        'p50': None, 'p95': None,
        'bins': list(bins), 'counts': counts},
        tags=tags or {'process.pid': ROOT_PID}, ts=ts)


def build_payload(t, engine='vllm', profiles=True, graph_mode='node', devices=1,
                  errors=True, extra_metrics=None):
    ts = START_NS + int(t * NS)
    metrics = []

    device_tag_sets = []
    for i in range(devices):
        tags = {'device.uuid': f'GPU-{i:04x}' if devices > 1 else DEVICE_UUID,
                'device.bus_id': f'0000:{i + 1:02x}:00.0'}
        device_tag_sets.append(tags)
        gauges = {
            'gpu_utilization_percent': 94.0 - i,
            'gpu_memory_used_bytes': 98.1e9,
            'gpu_memory_total_bytes': 119.7e9,
            'gpu_memory_utilization_percent': 71.0,
            'gpu_power_usage_watts': 85.0,
            'gpu_power_management_limit_watts': 140.0,
            'gpu_temperature_celsius': 61.0,
            'gpu_clock_sm_megahertz': 1800.0,
            'gpu_pcie_throughput_rx_kilobytes_per_second': 1200.0,
            'gpu_pcie_throughput_tx_kilobytes_per_second': 400.0,
            'gpu_clocks_throttle_reasons': 0,
        }
        for name, value in gauges.items():
            metrics.append(_metric(name, 'gauge', {'value': value}, tags=tags, ts=ts))
        metrics.append(_metric('gpu_xid_critical_errors', 'counter', {'total': 0}, tags=tags, ts=ts))

    if profiles:
        metrics.append(_metric('cuda_graph_trace_mode', 'gauge',
                               {'value': 1 if graph_mode == 'node' else 0},
                               tags={'process.pid': ROOT_PID}, ts=ts))
        metrics.append(_profile('cuda_kernels_nanoseconds', {
            FLASH_FWD: (int(314e6 * t), int(1980 * t)),
            GEMM: (int(221e6 * t), int(8430 * t)),
            RMS_NORM: (int(49e6 * t), int(16860 * t)),
        }, ROOT_PID, ts))
        metrics.append(_profile('cuda_kernels_nanoseconds', {
            GEMM: (int(10e6 * t), int(100 * t)),
        }, CHILD_PID, ts))
        if graph_mode == 'graph':
            metrics.append(_profile('cuda_graphs_nanoseconds', {
                GRAPH_SIG: (int(180e6 * t), int(2000 * t)),
            }, ROOT_PID, ts))
        else:
            metrics.append(_profile('cuda_graphs_nanoseconds', {}, ROOT_PID, ts))
        metrics.append(_profile('cuda_memcpy_nanoseconds', {
            'host_to_device': (int(20e6 * t), int(100 * t)),
        }, ROOT_PID, ts))
        metrics.append(_metric('cuda_memcpy_bytes', 'counter', {'total': int(5e6 * t)},
                               tags={'kind': 'host_to_device', 'process.pid': ROOT_PID}, ts=ts))
        metrics.append(_profile('cuda_sync_nanoseconds', {
            'stream_synchronize': (int(60e6 * t), int(50 * t)),
        }, ROOT_PID, ts))

    if engine == 'vllm':
        tags = {'model_name': 'm', 'process.pid': ROOT_PID}
        metrics.append(_metric('vllm:num_requests_running', 'gauge', {'value': 48}, tags, ts))
        metrics.append(_metric('vllm:num_requests_waiting', 'gauge', {'value': 3}, tags, ts))
        metrics.append(_metric('vllm:kv_cache_usage_perc', 'gauge', {'value': 0.71}, tags, ts))
        metrics.append(_metric('vllm:prefix_cache_hits', 'counter', {'total': int(420 * t)}, tags, ts))
        metrics.append(_metric('vllm:prefix_cache_queries', 'counter', {'total': int(1000 * t)}, tags, ts))
        metrics.append(_metric('vllm:prompt_tokens', 'counter', {'total': int(11200 * t)}, tags, ts))
        metrics.append(_metric('vllm:generation_tokens', 'counter', {'total': int(1840 * t)}, tags, ts))
        metrics.append(_histogram('vllm:time_to_first_token_seconds',
                                  [0.1, 0.2, 0.5, 1.0], [10, 30, 5, 1], t, ts, tags))
        metrics.append(_histogram('vllm:time_per_output_token_seconds',
                                  [0.01, 0.025, 0.05, 0.1], [5, 40, 4, 1], t, ts, tags))
        metrics.append(_histogram('vllm:e2e_request_latency_seconds',
                                  [1.0, 2.5, 5.0, 10.0], [8, 30, 6, 2], t, ts, tags))
    elif engine == 'sglang':
        tags = {'model_name': 'm'}
        metrics.append(_metric('sglang:num_running_reqs', 'gauge', {'value': 12}, tags, ts))
        metrics.append(_metric('sglang:gen_throughput', 'gauge', {'value': 950.0}, tags, ts))
        metrics.append(_metric('sglang:token_usage', 'gauge', {'value': 0.5}, tags, ts))

    metrics.extend(extra_metrics or [])

    resources = [
        {'kind': 'host', 'tags': {'host.name': 'dgx-spark-01'}, 'attributes': {'platform.name': 'Linux'},
         'first_seen_ts': START_NS // NS, 'last_seen_ts': ts // NS},
        {'kind': 'process', 'tags': {'process.pid': ROOT_PID},
         'attributes': {'process.command_line': 'vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001'},
         'first_seen_ts': START_NS // NS, 'last_seen_ts': ts // NS},
        {'kind': 'process', 'tags': {'process.pid': CHILD_PID, 'process.root_pid': ROOT_PID},
         'attributes': {'process.command_line': 'VLLM::EngineCore'},
         'first_seen_ts': START_NS // NS + 1, 'last_seen_ts': ts // NS},
    ]
    for i, tags in enumerate(device_tag_sets):
        resources.append({'kind': 'device', 'tags': {'device.uuid': tags['device.uuid']},
                          'attributes': {'device.name': 'NVIDIA GB10'},
                          'first_seen_ts': START_NS // NS, 'last_seen_ts': ts // NS})

    return {
        'profiler': {'version': '1.1.0'},
        'payload_ns': ts,
        'start_ns': START_NS,
        'context': {'instance.id': 'abc', 'host.name': 'dgx-spark-01'},
        'metrics': metrics,
        'errors': ([{'level': 'warning',
                     'message': 'Profiler 1.1.0 is outdated, latest 1.2.0',
                     'exception': None, 'tags': {}, 'log_ns': ts - NS}] if errors else []),
        'resources': resources,
    }
