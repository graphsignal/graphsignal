import unittest
import graphsignal.watcher
from graphsignal import version
from graphsignal.signals.routes import build_payload, quantiles_from_bins
from test.test_utils import configure_test_watcher
import json
import time
import urllib.error
import urllib.request
from graphsignal.signals.routes import SignalsEndpoint
from test.test_utils import configure_test_watcher, free_port


class QuantilesFromBinsTest(unittest.TestCase):
    def test_nearest_rank_over_bins(self):
        self.assertEqual(quantiles_from_bins([0, 10, 20, 30], [1, 4, 3, 2]),
                         {'p50': 10, 'p95': 30})

    def test_all_in_first_bin(self):
        self.assertEqual(quantiles_from_bins([5, 10], [10, 0]),
                         {'p50': 5, 'p95': 5})

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(quantiles_from_bins([], []))
        self.assertIsNone(quantiles_from_bins([1, 2], [1]))
        self.assertIsNone(quantiles_from_bins(None, None))
        self.assertIsNone(quantiles_from_bins([1], None))
        # All-zero counts: no samples to rank.
        self.assertIsNone(quantiles_from_bins([1], [0]))


class BuildPayloadTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _get_metric(self, payload, name, metric_type=None):
        for record in payload['metrics']:
            if record['name'] != name:
                continue
            if metric_type is not None and record['type'] != metric_type:
                continue
            return record
        return None

    def test_header_and_context(self):
        payload = build_payload()
        self.assertEqual(payload['profiler'], {'version': version.__version__})
        self.assertGreater(payload['payload_ns'], 0)
        # Counters accumulate from the instance start, and the payload says so.
        self.assertGreater(payload['start_ns'], 0)
        self.assertLessEqual(payload['start_ns'], payload['payload_ns'])
        self.assertIn('instance.id', payload['context'])

    def test_gauge_stats_last_set_value(self):
        self.watcher.set_gauge('g1', 3.0, measurement_ts=1010, tags={'t1': '1'})
        self.watcher.set_gauge('g1', 20.0, measurement_ts=1020, tags={'t1': '1'})

        record = self._get_metric(build_payload(), 'g1')
        self.assertIsNotNone(record)
        self.assertEqual(record['type'], 'gauge')
        self.assertEqual(record['tags'], {'t1': '1'})
        self.assertEqual(record['updated_ns'], 1020)
        self.assertEqual(record['stats'], {'value': 20.0})

    def test_counter_stats(self):
        self.watcher.set_counter('c1', 100, measurement_ts=1000)
        self.watcher.set_counter('c1', 142, measurement_ts=2000)

        record = self._get_metric(build_payload(), 'c1')
        self.assertEqual(record['type'], 'counter')
        self.assertEqual(record['stats'], {'total': 142})
        self.assertEqual(record['updated_ns'], 2000)

    def test_histogram_stats_without_bins(self):
        # A Prometheus summary: exact totals, no buckets. mean is exact;
        # quantiles have no bins to come from and stay null.
        self.watcher.set_histogram('s1', count=4, sum_val=95.0,
                                   min_val=1.5, max_val=60.0,
                                   measurement_ts=3000)

        record = self._get_metric(build_payload(), 's1')
        self.assertEqual(record['type'], 'histogram')
        self.assertEqual(record['updated_ns'], 3000)
        self.assertEqual(record['stats'], {
            'count': 4,
            'sum': 95.0,
            'min': 1.5,
            'max': 60.0,
            'mean': 23.75,
            'p50': None,
            'p95': None,
            'bins': None,
            'counts': None,
        })

    def test_histogram_stats_zero_count_mean_is_null(self):
        self.watcher.set_histogram('s1', count=0, sum_val=0.0,
                                   measurement_ts=3000)

        stats = self._get_metric(build_payload(), 's1')['stats']
        self.assertEqual(stats['count'], 0)
        self.assertEqual(stats['sum'], 0.0)
        self.assertIsNone(stats['mean'])

    def test_histogram_stats_mean_and_quantiles(self):
        self.watcher.set_histogram('h1', bins=[5, 20, 50], counts=[1, 2, 1],
                                   measurement_ts=3000)

        record = self._get_metric(build_payload(), 'h1')
        self.assertEqual(record['type'], 'histogram')
        self.assertEqual(record['updated_ns'], 3000)
        # mean = (5*1 + 20*2 + 50*1) / 4 = 23.75; nearest-rank quantiles;
        # no exact aggregates supplied -> null (not measured).
        self.assertEqual(record['stats'], {
            'count': None,
            'sum': None,
            'min': None,
            'max': None,
            'mean': 23.75,
            'p50': 20,
            'p95': 50,
            'bins': [5, 20, 50],
            'counts': [1, 2, 1],
        })

    def test_histogram_stats_exact_aggregates(self):
        # Bins say 4 values in [5, 20, 20, 50]; the writer's exact aggregates
        # say the true values summed to 101 -> mean 25.25, not the bin mean.
        self.watcher.set_histogram('h2', bins=[5, 20, 50], counts=[1, 2, 1],
                                   measurement_ts=3000, count=4, sum_val=101,
                                   min_val=6, max_val=59)
        stats = self._get_metric(build_payload(), 'h2')['stats']
        self.assertEqual(stats['count'], 4)
        self.assertEqual(stats['sum'], 101)
        self.assertEqual(stats['min'], 6)
        self.assertEqual(stats['max'], 59)
        self.assertEqual(stats['mean'], 25.25)
        self.assertEqual(stats['p50'], 20)
        self.assertEqual(stats['p95'], 50)

    def test_profile_stats_frames_sorted_by_value_desc(self):
        self.watcher.set_profile('p1', frames={'b': 7, 'a': 100, 'c': 50},
                                 samples={'b': 1, 'a': 12},
                                 measurement_ts=3000)

        record = self._get_metric(build_payload(), 'p1')
        self.assertEqual(record['type'], 'profile')
        self.assertEqual(record['updated_ns'], 3000)
        # A frame nothing counted reports zero samples.
        self.assertEqual(record['stats'], {'frames': [
            {'name': 'a', 'value': 100, 'samples': 12},
            {'name': 'c', 'value': 50, 'samples': 0},
            {'name': 'b', 'value': 7, 'samples': 1},
        ]})

    def test_metric_records_sorted_by_name_type_tags(self):
        self.watcher.set_counter('m1', 1, measurement_ts=1000, tags={'t1': '1'})
        self.watcher.set_histogram('m1', bins=[1], counts=[1],
                                   measurement_ts=1000, tags={'t1': '1'})
        self.watcher.set_gauge('a1', 1.0, measurement_ts=1000, tags={'t1': 'b'})
        self.watcher.set_gauge('a1', 1.0, measurement_ts=1000, tags={'t1': 'a'})

        keys = [(r['name'], r['type'], sorted(r['tags'].items()))
                for r in build_payload()['metrics']]
        self.assertEqual(keys, sorted(keys))
        names_types = [(r['name'], r['type'])
                       for r in build_payload()['metrics']]
        self.assertEqual(names_types, [
            ('a1', 'gauge'), ('a1', 'gauge'),
            ('m1', 'counter'), ('m1', 'histogram')])

    def test_errors_section(self):
        self.watcher.log_message('something broke', level='error',
                                 tags={'t1': '1'})
        self.watcher.log_message('all fine', level='info')

        messages = [e['message'] for e in build_payload()['errors']]
        self.assertIn('something broke', messages)
        self.assertNotIn('all fine', messages)

    def test_resources_section(self):
        self.watcher.update_resource(
            'process', tags={'process.pid': '1'},
            attributes={'process.command_line': 'python app.py'},
            first_seen_ts=1_000_000_000, last_seen_ts=2_000_000_000)

        payload = build_payload()
        process = [r for r in payload['resources'] if r['kind'] == 'process']
        self.assertEqual(len(process), 1)
        self.assertEqual(process[0]['tags'], {'process.pid': '1'})
        self.assertEqual(process[0]['attributes'],
                         {'process.command_line': 'python app.py'})
        # The store keeps nanoseconds; the payload's `_ts` fields are seconds.
        self.assertEqual(process[0]['first_seen_ts'], 1)
        self.assertEqual(process[0]['last_seen_ts'], 2)


class BuildPayloadUnconfiguredTest(unittest.TestCase):
    def test_empty_payload_when_watcher_not_configured(self):
        self.assertFalse(graphsignal.watcher.is_configured())
        payload = build_payload()
        self.assertEqual(payload['context'], {})
        self.assertEqual(payload['metrics'], [])
        self.assertEqual(payload['errors'], [])
        self.assertEqual(payload['resources'], [])
        self.assertEqual(payload['profiler'], {'version': version.__version__})
        self.assertGreater(payload['payload_ns'], 0)
        self.assertIsNone(payload['start_ns'])


if __name__ == '__main__':
    unittest.main()


class SignalsEndpointTest(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.watcher = configure_test_watcher(listen_port=self.port)

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _get(self, path):
        with urllib.request.urlopen(
                f'http://127.0.0.1:{self.port}{path}', timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))

    def test_health(self):
        status, body = self._get('/health')
        self.assertEqual(status, 200)
        self.assertEqual(body, {'status': 'ok'})

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._get('/nope')
        self.assertEqual(cm.exception.code, 404)

    def test_signals_serves_payload(self):
        self.watcher.set_gauge('g1', 20.0, measurement_ts=1020, tags={'t1': '1'})

        status, payload = self._get('/signals')
        self.assertEqual(status, 200)
        self.assertEqual(payload['profiler'], {'version': version.__version__})
        self.assertGreater(payload['payload_ns'], 0)
        self.assertIn('instance.id', payload['context'])
        self.assertIn('errors', payload)
        self.assertIn('resources', payload)

        records = [r for r in payload['metrics'] if r['name'] == 'g1']
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['type'], 'gauge')
        self.assertEqual(record['tags'], {'t1': '1'})
        self.assertEqual(record['updated_ns'], 1020)
        self.assertEqual(record['stats'], {'value': 20.0})

    def test_second_endpoint_on_same_port_disabled_without_exception(self):
        second = SignalsEndpoint(port=self.port, bind_retry_timeout=0)
        second.setup()  # must not raise
        try:
            self.assertFalse(second.is_running())
        finally:
            second.shutdown()

    def test_endpoint_binds_once_the_port_frees_up(self):
        # The previous run's watcher releasing the port a moment after the new
        # one starts must not cost the new run its endpoint.
        second = SignalsEndpoint(port=self.port, bind_retry_interval=0.05,
                                 bind_retry_timeout=5)
        second.setup()
        try:
            self.assertFalse(second.is_running())
            self.watcher.signals_endpoint().shutdown()
            deadline = time.time() + 5
            while not second.is_running() and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(second.is_running())
            with urllib.request.urlopen(
                    f'http://127.0.0.1:{self.port}/health', timeout=5) as resp:
                self.assertEqual(resp.status, 200)
        finally:
            second.shutdown()

    def test_endpoint_gives_up_after_timeout(self):
        second = SignalsEndpoint(port=self.port, bind_retry_interval=0.05,
                                 bind_retry_timeout=0.2)
        second.setup()
        try:
            time.sleep(0.5)
            self.assertFalse(second.is_running())
        finally:
            second.shutdown()


if __name__ == '__main__':
    unittest.main()
