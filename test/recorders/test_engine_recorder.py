import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import graphsignal.watcher
import graphsignal.recorders.log_recorder as log_recorder_mod
from graphsignal.recorders.engine_recorder import (
    EngineRecorder, _HealthTimes, build_engine_resource, parse_config_line,
    redact_engine_info)
from test.http_server import HttpTestServer
from test.test_utils import configure_test_watcher, find_metric

_HEALTH_COUNTERS = ('engine_startup_nanoseconds', 'engine_uptime_nanoseconds',
                    'engine_downtime_nanoseconds', 'engine_down_events')

_GAUGE_BODY = """# HELP vllm_num_requests_running Number of running requests.
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{model_name="m"} %s
"""

_COUNTER_BODY = """# HELP vllm_request_success Count of successful requests.
# TYPE vllm_request_success counter
vllm_request_success{model_name="m"} %s
"""

_HISTOGRAM_BODY = """# HELP vllm_e2e_latency Request latency.
# TYPE vllm_e2e_latency histogram
vllm_e2e_latency_bucket{le="0.5",model_name="m"} %(b1)s
vllm_e2e_latency_bucket{le="1.0",model_name="m"} %(b2)s
vllm_e2e_latency_bucket{le="+Inf",model_name="m"} %(count)s
vllm_e2e_latency_count{model_name="m"} %(count)s
vllm_e2e_latency_sum{model_name="m"} %(sum)s
"""


def _recorder(**kwargs):
    kwargs.setdefault('root_pid', 123)
    kwargs.setdefault('pid', 123)
    recorder = EngineRecorder(**kwargs)
    recorder.setup()
    # Skip the initial detection delay so the first on_tick scrapes.
    if recorder._scraper is not None:
        recorder._scraper._next_attempt_ts = 0
    return recorder


class EngineRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _tick(self, recorder, body):
        with patch.object(recorder._scraper, '_fetch', return_value=body) as fetch_m:
            recorder.on_tick()
        return fetch_m

    def test_no_metrics_port_has_no_scraper(self):
        recorder = _recorder(metrics_port=None)
        self.assertIsNone(recorder._scraper)
        recorder.on_tick()  # nothing to scrape, must not raise

    def test_fetches_only_configured_endpoint(self):
        recorder = _recorder(metrics_port=8000)
        self.assertEqual(recorder._scraper.endpoint, 'http://127.0.0.1:8000/metrics')
        fetch_m = self._tick(recorder, _GAUGE_BODY % '3.0')
        fetch_m.assert_called_once_with('http://127.0.0.1:8000/metrics')

    def test_fetches_configured_path_and_host(self):
        recorder = _recorder(metrics_port=8000,
                             metrics_path='/prometheus/metrics',
                             metrics_host='localhost')
        fetch_m = self._tick(recorder, _GAUGE_BODY % '3.0')
        fetch_m.assert_called_once_with(
            'http://localhost:8000/prometheus/metrics')

    def test_gauges_overwrite_datapoint(self):
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, _GAUGE_BODY % '3.0')
        self._tick(recorder, _GAUGE_BODY % '5.0')

        metric = find_metric(self.watcher.metric_store().export(),
                             'vllm_num_requests_running', {'model_name': 'm'})
        self.assertEqual(metric.type, 'gauge')
        self.assertEqual(metric.datapoint['value'], 5.0)

    def test_counters_pass_through_raw_cumulative_totals(self):
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, _COUNTER_BODY % '100.0')

        metric = find_metric(self.watcher.metric_store().export(),
                             'vllm_request_success', {'model_name': 'm'})
        self.assertEqual(metric.type, 'counter')
        self.assertEqual(metric.datapoint['total'], 100.0)

        # Second scrape: the raw total is emitted again (no delta state);
        # the store keeps the latest snapshot.
        self._tick(recorder, _COUNTER_BODY % '150.0')
        metric = find_metric(self.watcher.metric_store().export(),
                             'vllm_request_success', {'model_name': 'm'})
        self.assertEqual(metric.datapoint['total'], 150.0)

    def test_histogram_family_emits_one_histogram_with_bins_and_totals(self):
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, _HISTOGRAM_BODY % {
            'b1': '4', 'b2': '7', 'count': '10', 'sum': '25.5'})

        exported = self.watcher.metric_store().export()
        histogram = find_metric(exported, 'vllm_e2e_latency',
                                {'model_name': 'm'}, metric_type='histogram')
        self.assertIsNotNone(histogram)
        # Cumulative le buckets converted: 0.5→4, 1.0→3, +Inf remainder 3
        # folded into the last finite bound.
        self.assertEqual(histogram.datapoint['bins'], [0.5, 1.0])
        self.assertEqual(histogram.datapoint['counts'], [4, 6])
        # The exact totals from _count/_sum ride on the same datapoint.
        self.assertEqual(histogram.datapoint['count'], 10)
        self.assertEqual(histogram.datapoint['sum'], 25.5)
        # Prometheus exposes no extremes.
        self.assertNotIn('min', histogram.datapoint)
        self.assertNotIn('max', histogram.datapoint)
        # One metric, not one per half.
        self.assertEqual(
            len([m for m in exported if m.name == 'vllm_e2e_latency']), 1)

        # Every scrape re-emits raw cumulative values; latest snapshots kept.
        self._tick(recorder, _HISTOGRAM_BODY % {
            'b1': '6', 'b2': '10', 'count': '15', 'sum': '40.0'})
        exported = self.watcher.metric_store().export()
        histogram = find_metric(exported, 'vllm_e2e_latency',
                                {'model_name': 'm'}, metric_type='histogram')
        self.assertEqual(histogram.datapoint['bins'], [0.5, 1.0])
        self.assertEqual(histogram.datapoint['counts'], [6, 9])
        self.assertEqual(histogram.datapoint['count'], 15)
        self.assertEqual(histogram.datapoint['sum'], 40.0)

    def test_summary_family_emits_a_histogram_without_bins(self):
        body = """# HELP rpc_duration_seconds RPC duration.
# TYPE rpc_duration_seconds summary
rpc_duration_seconds{quantile="0.5"} 0.05
rpc_duration_seconds_count 12
rpc_duration_seconds_sum 1.5
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        exported = self.watcher.metric_store().export()
        histogram = find_metric(exported, 'rpc_duration_seconds',
                                metric_type='histogram')
        self.assertIsNotNone(histogram)
        self.assertEqual(histogram.datapoint['count'], 12)
        # Fractional seconds survive: a truncated sum would read 1.
        self.assertEqual(histogram.datapoint['sum'], 1.5)
        self.assertNotIn('bins', histogram.datapoint)

    def test_skips_gauge_histogram_bucket_labels(self):
        body = """# HELP sglang:routing_key_running_req_count Distribution of routing keys.
# TYPE sglang:routing_key_running_req_count gauge
sglang:routing_key_running_req_count{gt="0",le="1",model_name="m"} 2.0
sglang:routing_key_running_req_count{gt="1",le="2",model_name="m"} 1.0
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="m"} 12.5
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        exported = self.watcher.metric_store().export()
        self.assertIsNone(
            find_metric(exported, 'sglang:routing_key_running_req_count'))
        metric = find_metric(exported, 'sglang:gen_throughput')
        self.assertEqual(metric.datapoint['value'], 12.5)
        self.assertNotIn('gt', metric.tags)
        self.assertNotIn('le', metric.tags)

    def test_strips_pid_label(self):
        body = """# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{pid="1973",model_name="m"} 4.0
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        metric = find_metric(self.watcher.metric_store().export(),
                             'sglang:num_running_reqs')
        self.assertNotIn('pid', metric.tags)
        self.assertEqual(metric.tags.get('model_name'), 'm')

    def test_skips_non_finite_gauge_values(self):
        body = """# HELP sglang:fwd_occupancy Forward pass GPU occupancy percentage.
# TYPE sglang:fwd_occupancy gauge
sglang:fwd_occupancy NaN
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput 12.5
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        exported = self.watcher.metric_store().export()
        self.assertIsNone(find_metric(exported, 'sglang:fwd_occupancy'))
        self.assertIsNotNone(find_metric(exported, 'sglang:gen_throughput'))

    def test_non_prometheus_body_is_not_emitted(self):
        recorder = _recorder(metrics_port=8000)
        with patch.object(recorder._scraper, '_fetch',
                          return_value='<html>nope</html>'), \
             patch.object(recorder, '_parse_and_emit') as parse_m:
            recorder.on_tick()
        parse_m.assert_not_called()
        self.assertFalse(recorder._scraper.verified)

    def test_fetch_failure_backs_off_without_raising(self):
        recorder = _recorder(metrics_port=8000)
        with patch.object(recorder._scraper, '_fetch',
                          side_effect=OSError('refused')):
            recorder.on_tick()  # must not raise
        self.assertFalse(recorder._scraper.verified)



class HealthTimesTest(unittest.TestCase):
    def test_states_and_attribution(self):
        t = _HealthTimes(start_ns=0)
        # Refused while the model loads: startup, not downtime.
        t.record(False, 5)
        t.record(False, 10)
        # The interval ending in the first healthy probe is still startup.
        t.record(True, 15)
        self.assertEqual((t.startup_ns, t.uptime_ns, t.downtime_ns), (15, 0, 0))
        t.record(True, 20)
        # 503: held, not counted yet.
        t.record(False, 25)
        self.assertEqual((t.downtime_ns, t.down_events), (0, 0))
        # Next probe, engine still running: the held gap counts as down and
        # the outage counts; this probe's gap goes to the state it observes.
        t.record(True, 30)
        self.assertEqual((t.uptime_ns, t.downtime_ns, t.down_events), (10, 5, 1))
        # Timeout, then still down: each failure commits at the next probe,
        # and it is one outage, not two.
        t.record(False, 35)
        t.record(False, 40)
        self.assertEqual(t.startup_ns, 15)
        self.assertEqual(t.uptime_ns, 10)
        self.assertEqual(t.downtime_ns, 10)
        self.assertEqual(t.down_events, 2)
        # The last failure is still held: committed + held = elapsed.
        self.assertEqual(t.held_ns, 5)
        self.assertEqual(t.startup_ns + t.uptime_ns + t.downtime_ns + t.held_ns, 40)

    def test_graceful_stop_is_not_an_outage(self):
        # SIGTERM: the port closes, one probe is refused, then the process
        # exits and no further probe runs.
        t = _HealthTimes(start_ns=0)
        t.record(True, 5)
        t.record(True, 10)
        t.record(False, 15)
        self.assertEqual((t.downtime_ns, t.down_events), (0, 0))

    def test_failure_before_exit_is_dropped(self):
        # The process was gone by the next probe: a crash or a slow stop
        # ending in exit, reported by errors and the exit record instead.
        t = _HealthTimes(start_ns=0)
        t.record(True, 5)
        t.record(False, 10)
        t.record(False, 15, engine_alive=False)
        self.assertEqual((t.downtime_ns, t.down_events), (0, 0))

    def test_outage_while_running_counts(self):
        t = _HealthTimes(start_ns=0)
        t.record(True, 5)
        for now in (10, 15, 20):
            t.record(False, now)
        t.record(True, 25)
        self.assertEqual((t.downtime_ns, t.down_events, t.uptime_ns), (15, 1, 5))

    def test_clock_going_backwards_adds_nothing(self):
        t = _HealthTimes(start_ns=100)
        t.record(True, 50)
        self.assertEqual(t.startup_ns, 0)


def _dead_pid():
    proc = subprocess.Popen(['true'])
    proc.wait()
    return proc.pid


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class EngineRecorderHealthTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            recorder.shutdown()
        graphsignal.watcher.shutdown()

    def _recorder(self, **kwargs):
        kwargs.setdefault('health_port', 8000)
        kwargs.setdefault('pid', 123)
        recorder = EngineRecorder(root_pid=123, **kwargs)
        self.recorders.append(recorder)
        return recorder

    def _counters(self):
        exported = self.watcher.metric_store().export()
        totals = {}
        for name in _HEALTH_COUNTERS:
            metric = find_metric(exported, name, metric_type='counter')
            totals[name] = None if metric is None else metric.datapoint['total']
        return totals

    def test_health_url(self):
        self.assertEqual(self._recorder()._health_url,
                         'http://127.0.0.1:8000/health')
        self.assertEqual(
            self._recorder(health_port=30000, health_path='ready',
                           health_host='::1')._health_url,
            'http://[::1]:30000/ready')
        self.assertIsNone(self._recorder(health_port=None)._health_url)

    def test_no_health_port_emits_nothing(self):
        recorder = self._recorder(health_port=None)
        recorder.setup()
        recorder.on_tick()
        self.assertIsNone(recorder._health_thread)
        self.assertEqual(self._counters(), {n: None for n in _HEALTH_COUNTERS})

    def test_counters_published_at_zero_before_any_probe(self):
        recorder = self._recorder(health_interval_sec=3600)
        recorder.setup()
        recorder.on_tick()
        self.assertEqual(self._counters(), {n: 0 for n in _HEALTH_COUNTERS})

    def test_probe_loop_accrues_startup_then_uptime(self):
        recorder = self._recorder(health_interval_sec=0.01)
        with patch.object(recorder, '_probe_health', return_value=True):
            recorder.setup()
            self.assertTrue(_wait_for(lambda: recorder._health.uptime_ns > 0))
        recorder.on_tick()
        counters = self._counters()
        self.assertGreater(counters['engine_startup_nanoseconds'], 0)
        self.assertGreater(counters['engine_uptime_nanoseconds'], 0)
        self.assertEqual(counters['engine_downtime_nanoseconds'], 0)
        self.assertEqual(counters['engine_down_events'], 0)

    def test_hung_probe_does_not_delay_on_tick(self):
        release = threading.Event()
        recorder = self._recorder(health_interval_sec=0.01)
        started = threading.Event()

        def hung_probe():
            started.set()
            release.wait(5)
            return True

        with patch.object(recorder, '_probe_health', side_effect=hung_probe):
            recorder.setup()
            self.assertTrue(started.wait(2))
            t0 = time.monotonic()
            recorder.on_tick()
            self.assertLess(time.monotonic() - t0, 0.5)
            release.set()

    def test_finalize_stops_accrual(self):
        recorder = self._recorder(health_interval_sec=0.01)
        with patch.object(recorder, '_probe_health', return_value=False):
            recorder.setup()
            self.assertTrue(_wait_for(lambda: recorder._health.startup_ns > 0))
            recorder.finalize()
            recorder._health_thread.join(2)
            frozen = (recorder._health.startup_ns, recorder._health.uptime_ns,
                      recorder._health.downtime_ns)
            time.sleep(0.05)
        self.assertEqual(frozen, (recorder._health.startup_ns,
                                  recorder._health.uptime_ns,
                                  recorder._health.downtime_ns))
        # Never healthy: everything went to startup.
        self.assertEqual(frozen[1:], (0, 0))
        self.assertEqual(self._counters()['engine_startup_nanoseconds'], frozen[0])

    def test_probe_http_2xx_is_healthy(self):
        server = HttpTestServer()
        server.set_response_code(200)
        server.start()
        server.wait_ready()
        recorder = self._recorder(health_port=server.get_port(),
                                  health_host='localhost')
        self.assertTrue(recorder._probe_health())
        self.assertEqual(server.get_request_path(), '/health')

    def test_probe_http_503_is_unhealthy(self):
        server = HttpTestServer()
        server.set_response_code(503)
        server.start()
        server.wait_ready()
        recorder = self._recorder(health_port=server.get_port(),
                                  health_host='localhost', health_path='/ready')
        try:
            self.assertFalse(recorder._probe_health())
        finally:
            server.set_response_code(200)

    def _route_server(self, routes):
        """A local server answering GET <path> with routes[path] (404 when
        absent), recording the paths it was asked for."""
        requested = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requested.append(self.path)
                self.send_response(routes.get(self.path, 404))
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1], requested

    def test_target_alive(self):
        self.assertTrue(self._recorder(pid=os.getpid())._target_alive())
        self.assertFalse(self._recorder(pid=_dead_pid())._target_alive())
        self.assertTrue(self._recorder(pid=None)._target_alive())

    def test_probe_falls_back_on_404_and_stays_there(self):
        # An SGLang release without /ready (0.5.10) answers 404; /ping is up.
        port, requested = self._route_server({'/ping': 200})
        recorder = self._recorder(health_port=port, health_path='/ready,/ping')
        self.assertTrue(recorder._probe_health())
        self.assertTrue(recorder._probe_health())
        self.assertEqual(requested, ['/ready', '/ping', '/ping'])
        self.assertEqual(recorder._health_url, f'http://127.0.0.1:{port}/ping')

    def test_probe_keeps_primary_path_on_503(self):
        # A 503 from /ready is an answer (not ready), not a missing route.
        port, requested = self._route_server({'/ready': 503, '/ping': 200})
        recorder = self._recorder(health_port=port, health_path='/ready,/ping')
        self.assertFalse(recorder._probe_health())
        self.assertEqual(requested, ['/ready'])
        self.assertEqual(recorder._health_url, f'http://127.0.0.1:{port}/ready')

    def test_last_path_404_is_unhealthy(self):
        port, requested = self._route_server({})
        recorder = self._recorder(health_port=port, health_path='/health')
        self.assertFalse(recorder._probe_health())
        self.assertEqual(requested, ['/health'])

    def test_probe_connection_refused_is_unhealthy(self):
        recorder = self._recorder(health_port=_free_port())
        self.assertFalse(recorder._probe_health())


# vLLM's cache settings as an info gauge; a second data-parallel engine's
# series follows, which is not kept.
_CONFIG_INFO_BODY = """# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",enable_prefix_caching="True",engine="%(engine)s",long_label="%(long)s"} 1.0
"""
_CONFIG_INFO_BODY_SECOND = """vllm:cache_config_info{block_size="16",enable_prefix_caching="True",engine="1",long_label="other"} 1.0
"""

# vLLM 0.19's start-up line, trimmed, with each kind of value it prints.
_VLLM_CONFIG_LINE = (
    "(EngineCore pid=1985) INFO 11-17 15:48:23 [core.py:105] Initializing a V1 LLM "
    "engine (v0.19.0+6bc3197f) with config: model='Qwen/Qwen2.5-0.5B-Instruct', "
    "speculative_config=None, dtype=torch.bfloat16, max_seq_len=32768, "
    "tensor_parallel_size=1, enforce_eager=False, kv_cache_dtype=auto, "
    "structured_outputs_config=StructuredOutputsConfig(backend='auto', "
    "reasoning_parser='', enable_in_reasoning=False), seed=0, "
    "served_model_name=Qwen/Qwen2.5-0.5B-Instruct, enable_prefix_caching=True, "
    "compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, "
    "'custom_ops': ['none'], 'cache_dir': ''}")
_MARKER = ' with config: '


class ParseConfigLineTest(unittest.TestCase):
    def test_vllm_line(self):
        fields = parse_config_line(_VLLM_CONFIG_LINE.split(_MARKER, 1)[1])
        self.assertEqual(fields['model'], 'Qwen/Qwen2.5-0.5B-Instruct')
        self.assertIsNone(fields['speculative_config'])
        self.assertEqual(fields['dtype'], 'torch.bfloat16')
        self.assertEqual(fields['max_seq_len'], 32768)
        self.assertEqual(fields['tensor_parallel_size'], 1)
        self.assertIs(fields['enforce_eager'], False)
        self.assertIs(fields['enable_prefix_caching'], True)
        self.assertEqual(fields['served_model_name'], 'Qwen/Qwen2.5-0.5B-Instruct')
        # Nested reprs stay whole, commas and all.
        self.assertEqual(fields['structured_outputs_config'],
                         "StructuredOutputsConfig(backend='auto', reasoning_parser='', "
                         "enable_in_reasoning=False)")
        self.assertEqual(fields['compilation_config'],
                         "{'mode': <CompilationMode.VLLM_COMPILE: 3>, "
                         "'custom_ops': ['none'], 'cache_dir': ''}")
        self.assertEqual(len(fields), 12)

    def test_comma_inside_a_quoted_value(self):
        self.assertEqual(parse_config_line("a='x, b=1', c=2"), {'a': 'x, b=1', 'c': 2})

    def test_floats_and_empty(self):
        self.assertEqual(parse_config_line("u=0.9, e=, s=''"), {'u': 0.9, 'e': '', 's': ''})


# A trimmed SGLang /get_server_info answer, with every kind of secret planted.
_SGLANG_INFO = {
    'version': '0.5.10+516d57ac',
    'model_path': 'Qwen/Qwen2.5-0.5B-Instruct',
    'tokenizer_path': 'Qwen/Qwen2.5-0.5B-Instruct',
    'max_running_requests': 8,
    'max_total_tokens': None,
    'disable_radix_cache': False,
    'api_key': 'sk-live-123',
    'admin_api_key': None,
    'ssl_keyfile_password': 'hunter2',
    'hf_token': '',
    'encoder_urls': ['https://user:pass@encoder.internal/v1'],
    'mm_process_config': {'access_key': 'AKIA123', 'image': {'max_pixels': 1024}},
    'internal_states': [{'last_gen_throughput': 12.5}],
}


class EngineResourceTest(unittest.TestCase):
    def test_redacts_credentials_at_any_depth(self):
        redacted = redact_engine_info(_SGLANG_INFO)
        self.assertEqual(redacted['api_key'], '[redacted]')
        self.assertEqual(redacted['ssl_keyfile_password'], '[redacted]')
        self.assertEqual(redacted['mm_process_config']['access_key'], '[redacted]')
        self.assertEqual(redacted['encoder_urls'],
                         ['https://[redacted]@encoder.internal/v1'])
        # Unset is configuration, not a secret.
        self.assertIsNone(redacted['admin_api_key'])
        self.assertEqual(redacted['hf_token'], '')
        # Names that only contain a credential word are left alone.
        self.assertEqual(redacted['tokenizer_path'], 'Qwen/Qwen2.5-0.5B-Instruct')
        self.assertIsNone(redacted['max_total_tokens'])
        self.assertEqual(redacted['mm_process_config']['image'], {'max_pixels': 1024})

    def test_no_secret_reaches_the_payload(self):
        _tags, attributes = build_engine_resource('sglang', _SGLANG_INFO)
        payload = attributes['engine.config']
        for secret in ('sk-live-123', 'hunter2', 'AKIA123', 'user:pass'):
            self.assertNotIn(secret, payload)

    def test_tags_and_payload(self):
        tags, attributes = build_engine_resource('sglang', _SGLANG_INFO)
        self.assertEqual(tags, {'engine.name': 'sglang', 'engine.version': '0.5.10+516d57ac'})
        config = json.loads(attributes['engine.config'])
        self.assertEqual(config['max_running_requests'], 8)
        # Runtime state is not configuration.
        self.assertNotIn('internal_states', config)

    def test_payload_ignores_key_order_and_runtime_state(self):
        reordered = dict(reversed(list(_SGLANG_INFO.items())))
        reordered['internal_states'] = [{'last_gen_throughput': 99.0}]
        self.assertEqual(build_engine_resource('sglang', _SGLANG_INFO),
                         build_engine_resource('sglang', reordered))

    def test_identity_survives_a_config_change(self):
        # Tags are the resource's identity: a restart on new flags (or with a
        # new seed and KV pool) replaces the configuration, never adds a resource.
        changed = dict(_SGLANG_INFO, max_running_requests=64, random_seed=12345)
        before, after = (build_engine_resource('sglang', _SGLANG_INFO),
                         build_engine_resource('sglang', changed))
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])

    def test_version_only_answer(self):
        # vLLM's /version.
        tags, attributes = build_engine_resource('vllm', {'version': '0.11.0'})
        self.assertEqual(tags['engine.version'], '0.11.0')
        self.assertEqual(attributes['engine.config'], '{"version":"0.11.0"}')

    def test_non_object_answer_is_none(self):
        self.assertIsNone(build_engine_resource('sglang', ['not', 'an', 'object']))

    def test_config_info_labels_join_the_config(self):
        labels = {'gpu_memory_utilization': '0.9', 'enable_prefix_caching': 'True',
                  'hf_token': 'hf_live'}
        _tags, attributes = build_engine_resource(
            'vllm', {'version': '0.19.0'}, {'vllm:cache_config_info': labels})
        config = json.loads(attributes['engine.config'])
        self.assertEqual(config['vllm:cache_config_info']['gpu_memory_utilization'], '0.9')
        # Labels are redacted like the rest of the configuration.
        self.assertEqual(config['vllm:cache_config_info']['hf_token'], '[redacted]')


class EngineRecorderInfoTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.now_ns = 0

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _server(self, routes):
        """GET <path> answers routes[path] as (status, body); 404 otherwise."""
        requested = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requested.append(self.path)
                status, body = routes.get(self.path, (404, b''))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1], requested

    def _recorder(self, port, **kwargs):
        kwargs.setdefault('engine', 'sglang')
        return EngineRecorder(root_pid=123, pid=123, health_port=port,
                              clock_ns=lambda: self.now_ns, **kwargs)

    def _engine_resources(self):
        return [r for r in self.watcher.resource_store().export()
                if r['kind'] == 'engine']

    def test_info_url_from_the_engine_spec(self):
        self.assertEqual(self._recorder(30000)._engine_info_url,
                         'http://127.0.0.1:30000/get_server_info')
        vllm = self._recorder(8000, engine='vllm')
        self.assertEqual(vllm._engine_info_url, 'http://127.0.0.1:8000/version')
        self.assertEqual(vllm._config_line_marker, _MARKER)

    def test_no_engine_or_unlisted_engine_no_url(self):
        for engine in (None, 'trtllm'):
            recorder = self._recorder(8000, engine=engine)
            self.assertIsNone(recorder._engine_info_url)
            self.assertIsNone(recorder._config_line_marker)
            # Nothing to fetch: a healthy probe records no resource.
            recorder._record_engine_resource()
        self.assertEqual(self._engine_resources(), [])

    def test_records_engine_resource_whole(self):
        info = dict(_SGLANG_INFO, padding='x' * 5000)
        port, requested = self._server(
            {'/get_server_info': (200, json.dumps(info).encode())})
        recorder = self._recorder(port)
        recorder._record_engine_resource()

        resources = self._engine_resources()
        self.assertEqual(len(resources), 1)
        resource = resources[0]
        self.assertEqual(resource['tags']['engine.name'], 'sglang')
        self.assertEqual(resource['tags']['engine.version'], '0.5.10+516d57ac')
        config = json.loads(resource['attributes']['engine.config'])
        # Over the generic attribute cap, and stored whole.
        self.assertEqual(len(config['padding']), 5000)
        self.assertEqual(config['api_key'], '[redacted]')
        self.assertEqual(requested, ['/get_server_info'])

    def test_fetches_once_and_refreshes_on_period(self):
        port, requested = self._server(
            {'/get_server_info': (200, json.dumps(_SGLANG_INFO).encode())})
        recorder = self._recorder(port, engine_info_refresh_sec=60)
        recorder._record_engine_resource()
        first_seen = self._engine_resources()[0]['last_seen_ts']

        self.now_ns = int(30e9)
        recorder._record_engine_resource()
        self.assertEqual(self._engine_resources()[0]['last_seen_ts'], first_seen)

        self.now_ns = int(61e9)
        time.sleep(0.001)
        recorder._record_engine_resource()
        self.assertGreater(self._engine_resources()[0]['last_seen_ts'], first_seen)
        # Re-recorded from what was fetched, never fetched again.
        self.assertEqual(requested, ['/get_server_info'])

    def test_4xx_ends_attempts(self):
        port, requested = self._server({'/get_server_info': (401, b'')})
        recorder = self._recorder(port)
        recorder._record_engine_resource()
        recorder._record_engine_resource()
        self.assertEqual(requested, ['/get_server_info'])
        self.assertEqual(self._engine_resources(), [])

    def test_5xx_retries_on_next_healthy_probe(self):
        routes = {'/get_server_info': (503, b'')}
        port, requested = self._server(routes)
        recorder = self._recorder(port)
        recorder._record_engine_resource()
        self.assertEqual(self._engine_resources(), [])

        routes['/get_server_info'] = (200, json.dumps(_SGLANG_INFO).encode())
        recorder._record_engine_resource()
        self.assertEqual(len(self._engine_resources()), 1)
        self.assertEqual(requested, ['/get_server_info', '/get_server_info'])

    def test_non_json_answer_ends_attempts(self):
        port, requested = self._server({'/get_server_info': (200, b'<html>')})
        recorder = self._recorder(port)
        recorder._record_engine_resource()
        recorder._record_engine_resource()
        self.assertEqual(requested, ['/get_server_info'])
        self.assertEqual(self._engine_resources(), [])

    def _scrape(self, recorder, body):
        recorder._scraper._next_attempt_ts = 0
        with patch.object(recorder._scraper, '_fetch', return_value=body):
            recorder.on_tick()

    def test_scrape_keeps_config_info_labels_whole(self):
        long_value = 'x' * 400
        recorder = self._recorder(8000, metrics_port=8000)
        self._scrape(recorder, _CONFIG_INFO_BODY % {'engine': '0', 'long': long_value}
                     + _CONFIG_INFO_BODY_SECOND)
        self.assertEqual(recorder._config_info, {'vllm:cache_config_info': {
            'block_size': '16', 'enable_prefix_caching': 'True', 'engine': '0',
            'long_label': long_value}})
        self.assertTrue(recorder._config_info_scraped)

    def test_waits_for_the_first_scrape(self):
        port, requested = self._server(
            {'/version': (200, b'{"version": "0.19.0"}')})
        # The start-up line is already printed: only the scrape is waited for.
        self._console(123, [_VLLM_CONFIG_LINE])
        recorder = self._recorder(port, engine='vllm', metrics_port=port)
        recorder._record_engine_resource()
        self.assertEqual(self._engine_resources(), [])

        self._scrape(recorder, _CONFIG_INFO_BODY % {'engine': '0', 'long': 'v'})
        recorder._record_engine_resource()
        resources = self._engine_resources()
        self.assertEqual(len(resources), 1)
        config = json.loads(resources[0]['attributes']['engine.config'])
        self.assertEqual(config['version'], '0.19.0')
        self.assertEqual(config['vllm:cache_config_info']['block_size'], '16')
        # The info answer is fetched once, whatever the wait.
        self.assertEqual(requested, ['/version'])

    def test_records_without_config_info_past_the_wait(self):
        port, _requested = self._server(
            {'/version': (200, b'{"version": "0.19.0"}')})
        # The start-up line is already printed: only the scrape is missing.
        self._console(123, [_VLLM_CONFIG_LINE])
        recorder = self._recorder(port, engine='vllm', metrics_port=port,
                                  engine_config_info_wait_sec=60)
        recorder._record_engine_resource()
        self.now_ns = int(59e9)
        recorder._record_engine_resource()
        self.assertEqual(self._engine_resources(), [])

        self.now_ns = int(60e9)
        recorder._record_engine_resource()
        config = json.loads(self._engine_resources()[0]['attributes']['engine.config'])
        self.assertEqual(set(config), {'version', 'startup_config'})

    def _console(self, pid, lines):
        """Console segments as the supervisor writes them, under a temp base."""
        # addCleanup rather than enterContext, which needs Python 3.11.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = tmp.name
        shm_base = patch.object(log_recorder_mod, '_SHM_BASE', base)
        shm_base.start()
        self.addCleanup(shm_base.stop)
        log_dir = os.path.join(base, f'graphsignal_log_{pid}')
        os.mkdir(log_dir)
        path = os.path.join(log_dir, 'log_0.jsonl')

        def write(more):
            with open(path, 'a') as f:
                for line in more:
                    f.write(json.dumps({'ts': 1, 'stream': 'stdout', 'line': line}) + '\n')
        write(lines)
        return write

    def test_startup_config_joins_the_config(self):
        port, _requested = self._server({'/version': (200, b'{"version": "0.19.0"}')})
        self._console(123, ['INFO starting', _VLLM_CONFIG_LINE, 'INFO ready'])
        recorder = self._recorder(port, engine='vllm')
        recorder._record_engine_resource()
        config = json.loads(self._engine_resources()[0]['attributes']['engine.config'])
        self.assertEqual(config['version'], '0.19.0')
        self.assertEqual(config['startup_config']['max_seq_len'], 32768)
        self.assertEqual(config['startup_config']['tensor_parallel_size'], 1)

    def test_waits_for_the_startup_line(self):
        port, _requested = self._server({'/version': (200, b'{"version": "0.19.0"}')})
        write = self._console(123, ['INFO starting'])
        recorder = self._recorder(port, engine='vllm')
        recorder._record_engine_resource()
        self.assertEqual(self._engine_resources(), [])

        write(['INFO loading', _VLLM_CONFIG_LINE])
        recorder._record_engine_resource()
        config = json.loads(self._engine_resources()[0]['attributes']['engine.config'])
        self.assertEqual(config['startup_config']['dtype'], 'torch.bfloat16')

    def test_records_without_startup_line_past_the_wait(self):
        port, _requested = self._server({'/version': (200, b'{"version": "0.19.0"}')})
        recorder = self._recorder(port, engine='vllm',
                                  engine_config_info_wait_sec=60)
        # No console directory at all: a target not started by graphsignal-run.
        with patch.object(log_recorder_mod, '_SHM_BASE', '/nonexistent-graphsignal-test'):
            recorder._record_engine_resource()
            self.assertEqual(self._engine_resources(), [])
            self.now_ns = int(60e9)
            recorder._record_engine_resource()
        config = json.loads(self._engine_resources()[0]['attributes']['engine.config'])
        self.assertEqual(config, {'version': '0.19.0'})

    def test_recorded_after_a_healthy_probe_only(self):
        port, requested = self._server(
            {'/get_server_info': (200, json.dumps(_SGLANG_INFO).encode())})
        recorder = self._recorder(port, health_interval_sec=0.01)
        self.addCleanup(recorder.shutdown)
        with patch.object(recorder, '_probe_health', return_value=False):
            recorder.setup()
            time.sleep(0.1)
            self.assertEqual(requested, [])
        with patch.object(recorder, '_probe_health', return_value=True):
            self.assertTrue(_wait_for(lambda: self._engine_resources()))


if __name__ == '__main__':
    unittest.main()
