import unittest

from graphsignal.top.state import TopState
from test.top import fixtures as fx


def state_after(ts, window=5.0, **kwargs):
    state = TopState(window=window, endpoint='127.0.0.1:18259')
    for t in ts:
        state.update(fx.build_payload(t, **kwargs))
    return state


class SnapshotWindowTest(unittest.TestCase):
    def test_first_poll_has_no_base(self):
        state = state_after([10])
        self.assertIsNotNone(state.latest)
        self.assertIsNone(state.base)
        self.assertIsNone(state.dt)

    def test_base_is_at_least_window_old(self):
        state = state_after(range(10, 21), window=5.0)
        # polls at 10..20 one second apart: base 15, latest 20
        self.assertEqual(state.dt, 5.0)

    def test_base_is_oldest_when_window_not_yet_filled(self):
        state = state_after([10, 11, 12], window=5.0)
        self.assertEqual(state.dt, 2.0)

    def test_duplicate_payload_is_ignored(self):
        state = TopState(window=5.0)
        payload = fx.build_payload(10)
        state.update(payload)
        state.update(payload)
        self.assertIsNone(state.base)

    def test_restart_clears_history(self):
        state = state_after([10, 11, 12])
        restarted = fx.build_payload(1)
        restarted['start_ns'] = fx.START_NS + 50 * fx.NS
        restarted['payload_ns'] = restarted['start_ns'] + fx.NS
        state.update(restarted)
        self.assertIsNone(state.base)
        self.assertEqual(len(state.series(f'gpu.util.{fx.DEVICE_UUID}')), 1)

    def test_unavailable_flag_clears_on_update(self):
        state = TopState()
        state.mark_unavailable('connection refused')
        self.assertEqual(state.header().unavailable, 'connection refused')
        state.update(fx.build_payload(1))
        self.assertIsNone(state.header().unavailable)


class HeaderTest(unittest.TestCase):
    def test_header_fields(self):
        header = state_after([10, 15]).header()
        self.assertEqual(header.endpoint, '127.0.0.1:18259')
        self.assertEqual(header.pid, fx.ROOT_PID)
        self.assertTrue(header.command_line.startswith('vllm serve'))
        self.assertEqual(header.host, 'dgx-spark-01')
        self.assertEqual(header.profiler_version, '1.1.0')
        self.assertEqual(header.uptime_s, 15.0)
        self.assertEqual(header.graph_trace_mode, 'node')
        self.assertEqual(header.window_s, 5.0)
        self.assertFalse(header.rocm)

    def test_graph_mode_and_no_profiles(self):
        self.assertEqual(state_after([1], graph_mode='graph').header().graph_trace_mode, 'graph')
        self.assertIsNone(state_after([1], profiles=False).header().graph_trace_mode)

    def test_empty_state_header(self):
        header = TopState().header()
        self.assertIsNone(header.pid)
        self.assertIsNone(header.uptime_s)


class DevicesTest(unittest.TestCase):
    def test_single_device_view(self):
        devices = state_after([1, 2, 3]).devices()
        self.assertEqual(len(devices), 1)
        d = devices[0]
        self.assertEqual(d.index, 0)
        self.assertEqual(d.name, 'NVIDIA GB10')
        self.assertEqual(d.utilization, 94.0)
        self.assertEqual(d.utilization_history, [94.0, 94.0, 94.0])
        self.assertAlmostEqual(d.memory_fraction, 98.1 / 119.7)
        self.assertEqual(d.power, 85.0)
        self.assertEqual(d.power_limit, 140.0)
        self.assertEqual(d.temperature, 61.0)
        self.assertEqual(d.sm_clock_mhz, 1800.0)
        self.assertEqual(d.pcie_rx_bytes_per_s, 1_200_000.0)
        self.assertEqual(d.xid_errors, 0)
        self.assertFalse(d.throttled)
        self.assertEqual(d.throttle_history, [False, False, False])

    def test_devices_ordered_by_bus_id(self):
        devices = state_after([1], devices=3).devices()
        self.assertEqual([d.uuid for d in devices], ['GPU-0000', 'GPU-0001', 'GPU-0002'])
        self.assertEqual([d.utilization for d in devices], [94.0, 93.0, 92.0])

    def test_missing_gauges_are_none(self):
        payload = fx.build_payload(1)
        payload['metrics'] = [m for m in payload['metrics']
                              if m['name'] not in ('gpu_power_usage_watts', 'gpu_memory_total_bytes')]
        state = TopState()
        state.update(payload)
        d = state.devices()[0]
        self.assertIsNone(d.power)
        self.assertIsNone(d.memory_fraction)


class EngineTest(unittest.TestCase):
    def test_vllm_rates_over_window(self):
        engine = state_after(range(10, 21)).engine()
        self.assertEqual(engine.engine, 'vllm')
        self.assertEqual(engine.running, 48)
        self.assertEqual(engine.waiting, 3)
        self.assertAlmostEqual(engine.kv_cache_pct, 71.0)
        self.assertAlmostEqual(engine.prefix_hit_pct, 42.0)
        self.assertAlmostEqual(engine.prompt_tokens_per_s, 11200.0)
        self.assertAlmostEqual(engine.gen_tokens_per_s, 1840.0)
        # per-poll series: first poll has no previous → None, then 10 rates
        self.assertIsNone(engine.gen_history[0])
        self.assertEqual(len(engine.gen_history), 11)
        self.assertAlmostEqual(engine.gen_history[-1], 1840.0)

    def test_stale_scrape_does_not_dent_the_sparkline(self):
        # The watcher scraped at t=10 and t=12 but the view polled at 10, 11
        # and 12: the t=11 payload carries the t=10 counter unchanged.
        state = TopState(window=5.0)
        state.update(fx.build_payload(10))
        stale = fx.build_payload(11)
        fresh10 = {m['name']: m for m in fx.build_payload(10)['metrics']}
        for m in stale['metrics']:
            if m['name'] in ('vllm:prompt_tokens', 'vllm:generation_tokens'):
                m['stats'] = fresh10[m['name']]['stats']
                m['updated_ns'] = fresh10[m['name']]['updated_ns']
        state.update(stale)
        state.update(fx.build_payload(12))
        engine = state.engine()
        # first poll None, stale poll repeats nothing yet (None), fresh poll
        # covers 2 s of counter growth over 2 s of scrape interval: no spike
        self.assertEqual(engine.gen_history[0], None)
        self.assertAlmostEqual(engine.gen_history[2], 1840.0)
        state.update(fx.build_payload(13))
        stale = fx.build_payload(14)
        fresh13 = {m['name']: m for m in fx.build_payload(13)['metrics']}
        for m in stale['metrics']:
            if m['name'] in ('vllm:prompt_tokens', 'vllm:generation_tokens'):
                m['stats'] = fresh13[m['name']]['stats']
                m['updated_ns'] = fresh13[m['name']]['updated_ns']
        state.update(stale)
        # a stale poll repeats the previous rate instead of charting zero
        self.assertAlmostEqual(state.engine().gen_history[-1], 1840.0)
        self.assertAlmostEqual(state.engine().gen_history[-2], 1840.0)

    def test_interval_histogram_quantiles(self):
        engine = state_after(range(10, 21)).engine()
        # TTFT fills [10, 30, 5, 1]/s over bounds [0.1, 0.2, 0.5, 1.0]:
        # 46 observations per second; p50 lands in the 0.2 bin, p95 in 0.5.
        self.assertEqual(engine.ttft.p50, 0.2)
        self.assertEqual(engine.ttft.p95, 0.5)
        self.assertEqual(engine.ttft.count, 46 * 5)
        self.assertIsNotNone(engine.ttft.mean)
        self.assertEqual(engine.tpot.p50, 0.025)
        self.assertEqual(engine.e2e.p95, 5.0)

    def test_first_poll_has_no_rates(self):
        engine = state_after([10]).engine()
        self.assertIsNone(engine.prompt_tokens_per_s)
        self.assertIsNone(engine.ttft.p50)
        self.assertEqual(engine.running, 48)

    def test_cumulative_mode_returns_totals(self):
        engine = state_after([10, 15]).engine(cumulative=True)
        self.assertEqual(engine.gen_tokens_per_s, 1840 * 15)
        self.assertEqual(engine.ttft.count, 46 * 15)
        self.assertEqual(engine.ttft.p50, 0.2)

    def test_sglang_gauge_throughput_fallback(self):
        engine = state_after([1, 2], engine='sglang').engine()
        self.assertEqual(engine.engine, 'sglang')
        self.assertEqual(engine.running, 12)
        self.assertEqual(engine.gen_tokens_per_s, 950.0)
        self.assertIsNone(engine.prompt_tokens_per_s)
        self.assertAlmostEqual(engine.kv_cache_pct, 50.0)

    def test_no_engine(self):
        self.assertIsNone(state_after([1, 2], engine=None).engine())


class GpuTimeTest(unittest.TestCase):
    def test_busy_fractions(self):
        view = state_after(range(10, 21)).gpu_time()
        self.assertTrue(view.available)
        # (314 + 221 + 10 + 49) ms/s on one device = 59.4 %
        self.assertAlmostEqual(view.kernels, 59.4)
        self.assertAlmostEqual(view.graphs, 0.0)
        self.assertAlmostEqual(view.memcpy, 2.0)
        self.assertIsNone(view.memset)
        self.assertAlmostEqual(view.sync, 6.0)
        self.assertAlmostEqual(view.idle, 100 - 59.4 - 2.0)

    def test_divided_by_device_count(self):
        view = state_after(range(10, 21), devices=2).gpu_time()
        self.assertAlmostEqual(view.kernels, 29.7)

    def test_unavailable_without_profiles_or_base(self):
        self.assertFalse(state_after([1, 2], profiles=False).gpu_time().available)
        self.assertFalse(state_after([1]).gpu_time().available)

    def test_cumulative(self):
        view = state_after([10, 15]).gpu_time(cumulative=True)
        self.assertAlmostEqual(view.kernels, 59.4)


class TableTest(unittest.TestCase):
    def test_kernel_rows_sorted_and_named(self):
        rows = state_after(range(10, 21)).table('kernels')
        self.assertEqual([r.label for r in rows],
                         ['flash attention prefill', 'matmul bf16 128x128x64 tn', 'rms norm'])
        flash = rows[0]
        self.assertEqual(flash.category, 'attention')
        self.assertAlmostEqual(flash.share_pct, 31.4)
        self.assertAlmostEqual(flash.ns_per_s, 314e6)
        self.assertAlmostEqual(flash.calls_per_s, 1980.0)
        self.assertAlmostEqual(flash.avg_ns, 314e6 / 1980)
        self.assertEqual(flash.raw, fx.FLASH_FWD)
        self.assertEqual(flash.kernel.short, 'flash_fwd_kernel')

    def test_frames_summed_across_processes(self):
        rows = state_after(range(10, 21)).table('kernels')
        gemm = rows[1]
        self.assertAlmostEqual(gemm.ns_per_s, 231e6)
        self.assertEqual(set(gemm.per_process), {fx.ROOT_PID, fx.CHILD_PID})
        self.assertAlmostEqual(gemm.per_process[fx.CHILD_PID][0], 10e6 * 5)

    def test_memcpy_rows_carry_bytes(self):
        rows = state_after(range(10, 21)).table('memcpy')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].label, 'memcpy host to device')
        self.assertEqual(rows[0].category, 'memcpy')
        self.assertAlmostEqual(rows[0].bytes_per_s, 5e6)
        self.assertAlmostEqual(rows[0].share_pct, 2.0)

    def test_memset_folds_into_memcpy_tab(self):
        state = TopState(window=5.0)
        for t in (10, 15):
            payload = fx.build_payload(t)
            payload['metrics'].append({
                'name': 'cuda_memset_nanoseconds', 'type': 'profile', 'tags': {'process.pid': fx.ROOT_PID},
                'stats': {'frames': [{'name': 'device', 'value': int(2e6 * t), 'samples': 10 * t}]},
                'updated_ns': 0})
            payload['metrics'].append({
                'name': 'cuda_memset_bytes', 'type': 'counter', 'tags': {'kind': 'device', 'process.pid': fx.ROOT_PID},
                'stats': {'total': int(1e6 * t)}, 'updated_ns': 0})
            state.update(payload)
        rows = state.table('memcpy')
        self.assertEqual([r.label for r in rows], ['memcpy host to device', 'memset device'])
        self.assertEqual(rows[1].category, 'memset')
        self.assertAlmostEqual(rows[1].bytes_per_s, 1e6)
        self.assertAlmostEqual(rows[1].ns_per_s, 2e6)
        self.assertFalse(state.has_tab('memset'))
        self.assertAlmostEqual(state.gpu_time().memset, 0.2)

    def test_graph_rows(self):
        state = state_after(range(10, 21), graph_mode='graph')
        rows = state.table('graphs')
        self.assertEqual(rows[0].label, 'graph 3f2a9c1e…')
        self.assertAlmostEqual(rows[0].calls_per_s, 2000.0)
        self.assertTrue(state.has_tab('graphs'))
        self.assertTrue(state.has_tab('memcpy'))

    def test_cumulative_rows(self):
        rows = state_after([10, 15]).table('kernels', cumulative=True)
        self.assertEqual(rows[0].value_ns, int(314e6 * 15))
        self.assertAlmostEqual(rows[0].share_pct, 31.4)

    def test_first_poll_has_no_rows(self):
        self.assertEqual(state_after([10]).table('kernels'), [])

    def test_no_profiles(self):
        self.assertEqual(state_after([1, 2], profiles=False).table('kernels'), [])

    def test_errors(self):
        errors = state_after([1]).errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['level'], 'warning')
