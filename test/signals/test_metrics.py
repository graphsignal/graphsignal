import threading
import time
import unittest
from unittest.mock import patch

from graphsignal.signals import metrics as metrics_module
from graphsignal.signals.metrics import (
    MetricStore, GAUGE, COUNTER, HISTOGRAM, PROFILE,
    METRIC_EXPIRY_NS)


class MetricStoreGaugeTest(unittest.TestCase):
    def setUp(self):
        self.store = MetricStore()
        # Suppress the timestamp-based cleanup so small synthetic
        # measurement_ts values are kept; cleanup has its own test class.
        self.store._last_cleanup_ts = time.time_ns()

    def test_set_gauge_overwrites_datapoint(self):
        self.store.set_gauge('m1', 1.0, measurement_ts=10, tags={'t1': '1'})
        self.store.set_gauge('m1', 2.0, measurement_ts=20, tags={'t1': '1'})

        exported = self.store.export()
        self.assertEqual(len(exported), 1)
        metric = exported[0]
        self.assertEqual(metric.name, 'm1')
        self.assertEqual(metric.type, GAUGE)
        self.assertEqual(metric.tags, {'t1': '1'})
        self.assertEqual(metric.datapoint, {'ts': 20, 'value': 2.0})

    def test_different_tags_create_separate_series(self):
        self.store.set_gauge('m1', 1.0, measurement_ts=10, tags={'t1': 'a'})
        self.store.set_gauge('m1', 2.0, measurement_ts=10, tags={'t1': 'b'})
        self.assertEqual(len(self.store.export()), 2)

    def test_none_name_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_gauge(None, 1.0, measurement_ts=10)

    def test_none_value_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_gauge('m1', None, measurement_ts=10)


class MetricStoreCounterTest(unittest.TestCase):
    def setUp(self):
        self.store = MetricStore()
        # Suppress the timestamp-based cleanup so small synthetic
        # measurement_ts values are kept; cleanup has its own test class.
        self.store._last_cleanup_ts = time.time_ns()

    def test_set_counter_overwrites_datapoint(self):
        self.store.set_counter('c1', 100, measurement_ts=10, tags={'t1': '1'})
        self.store.set_counter('c1', 150, measurement_ts=20, tags={'t1': '1'})

        exported = self.store.export()
        self.assertEqual(len(exported), 1)
        metric = exported[0]
        self.assertEqual(metric.type, COUNTER)
        self.assertEqual(metric.datapoint, {'ts': 20, 'total': 150})

    def test_none_name_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_counter(None, 1, measurement_ts=10)

    def test_none_total_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_counter('c1', None, measurement_ts=10)


class MetricStoreHistogramTest(unittest.TestCase):
    def setUp(self):
        self.store = MetricStore()
        # Suppress the timestamp-based cleanup so small synthetic
        # measurement_ts values are kept; cleanup has its own test class.
        self.store._last_cleanup_ts = time.time_ns()

    def test_set_histogram_overwrites_datapoint(self):
        self.store.set_histogram('h1', bins=[0, 10], counts=[4, 6],
                                 measurement_ts=10, tags={'t1': '1'})
        self.store.set_histogram('h1', bins=[0, 10, 20], counts=[4, 6, 5],
                                 measurement_ts=20, tags={'t1': '1'})

        exported = self.store.export()
        self.assertEqual(len(exported), 1)
        metric = exported[0]
        self.assertEqual(metric.type, HISTOGRAM)
        self.assertEqual(metric.datapoint, {
            'ts': 20, 'bins': [0, 10, 20], 'counts': [4, 6, 5]})

    def test_aggregates_kept_only_as_a_complete_count_and_sum_pair(self):
        self.store.set_histogram('h1', bins=[8], counts=[2],
                                 measurement_ts=100, count=2, sum_val=19,
                                 min_val=9, max_val=10)
        dp = self.store.export()[0].datapoint
        self.assertEqual(dp['count'], 2)
        self.assertEqual(dp['sum'], 19)
        self.assertEqual(dp['min'], 9)
        self.assertEqual(dp['max'], 10)
        # sum without count (or vice versa) is not an aggregate: dropped.
        self.store.set_histogram('h1', bins=[8], counts=[2],
                                 measurement_ts=101, sum_val=19)
        dp = self.store.export()[0].datapoint
        self.assertNotIn('count', dp)
        self.assertNotIn('sum', dp)

    def test_bins_and_counts_copied_from_caller(self):
        bins = [0, 10]
        counts = [1, 2]
        self.store.set_histogram('h1', bins=bins, counts=counts,
                                 measurement_ts=10)
        bins.append(99)
        counts[0] = 999

        dp = self.store.export()[0].datapoint
        self.assertEqual(dp['bins'], [0, 10])
        self.assertEqual(dp['counts'], [1, 2])

    def test_none_name_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_histogram(None, bins=[1], counts=[1],
                                     measurement_ts=10)

    def test_aggregates_without_bins_are_a_whole_histogram(self):
        self.store.set_histogram('h1', measurement_ts=10, count=4,
                                 sum_val=1.25, min_val=0.1, max_val=0.9)
        dp = self.store.export()[0].datapoint
        self.assertEqual(dp, {'ts': 10, 'count': 4, 'sum': 1.25,
                              'min': 0.1, 'max': 0.9})

    def test_fractional_sum_is_not_truncated(self):
        self.store.set_histogram('h1', measurement_ts=10, count=2,
                                 sum_val=0.25)
        self.assertEqual(self.store.export()[0].datapoint['sum'], 0.25)

    def test_neither_bins_nor_aggregates_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_histogram('h1', measurement_ts=10)
        with self.assertRaises(ValueError):
            self.store.set_histogram('h1', bins=[], counts=[],
                                     measurement_ts=10)
        # Half a bin pair is no distribution, and half an aggregate is none.
        with self.assertRaises(ValueError):
            self.store.set_histogram('h1', bins=[1], counts=None,
                                     measurement_ts=10)
        with self.assertRaises(ValueError):
            self.store.set_histogram('h1', measurement_ts=10, count=4)

    def test_mismatched_bins_and_counts_raise(self):
        with self.assertRaises(ValueError):
            self.store.set_histogram('h1', bins=[1, 2], counts=[1],
                                     measurement_ts=10)


class MetricStoreProfileTest(unittest.TestCase):
    def setUp(self):
        self.store = MetricStore()
        # Suppress the timestamp-based cleanup so small synthetic
        # measurement_ts values are kept; cleanup has its own test class.
        self.store._last_cleanup_ts = time.time_ns()

    def test_set_profile_overwrites_datapoint(self):
        self.store.set_profile('p1', frames={'a': 100, 'b': 7}, samples={'a': 2, 'b': 1},
                               measurement_ts=10, tags={'t1': '1'})
        self.store.set_profile('p1', frames={'a': 150, 'b': 9}, samples={'a': 3, 'b': 2},
                               measurement_ts=20, tags={'t1': '1'})

        exported = self.store.export()
        self.assertEqual(len(exported), 1)
        metric = exported[0]
        self.assertEqual(metric.type, PROFILE)
        self.assertEqual(metric.datapoint,
                         {'ts': 20, 'frames': {'a': 150, 'b': 9},
                          'samples': {'a': 3, 'b': 2}})

    def test_frames_copied_from_caller(self):
        frames = {'a': 100}
        self.store.set_profile('p1', frames=frames, measurement_ts=10)
        frames['a'] = 999
        frames['b'] = 1

        dp = self.store.export()[0].datapoint
        self.assertEqual(dp['frames'], {'a': 100})

    def test_frames_capped_to_top_by_value(self):
        with patch.object(metrics_module, 'MAX_PROFILE_FRAMES', 3):
            frames = {f'f{i}': i for i in range(10)}
            self.store.set_profile('p1', frames=frames, measurement_ts=10)

        dp = self.store.export()[0].datapoint
        self.assertEqual(dp['frames'], {'f9': 9, 'f8': 8, 'f7': 7})

    def test_none_name_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_profile(None, frames={'a': 1}, measurement_ts=10)

    def test_none_frames_raises(self):
        with self.assertRaises(ValueError):
            self.store.set_profile('p1', frames=None, measurement_ts=10)


class MetricStoreTypeKeyTest(unittest.TestCase):
    def test_same_name_and_tags_coexist_as_counter_and_histogram(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        store.set_counter('m1', 10, measurement_ts=10, tags={'t1': '1'})
        store.set_histogram('m1', bins=[0.5, 1.0], counts=[4, 6],
                            measurement_ts=10, tags={'t1': '1'})

        exported = {m.type: m for m in store.export()}
        self.assertEqual(set(exported.keys()), {COUNTER, HISTOGRAM})
        self.assertEqual(exported[COUNTER].name, 'm1')
        self.assertEqual(exported[HISTOGRAM].name, 'm1')
        self.assertEqual(exported[COUNTER].datapoint['total'], 10)
        self.assertEqual(exported[HISTOGRAM].datapoint['bins'], [0.5, 1.0])


class MetricStoreTagTruncationTest(unittest.TestCase):
    def test_tag_key_and_value_truncated(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        long_key = 'k' * 100
        long_value = 'v' * 500
        store.set_gauge('m1', 1.0, measurement_ts=10,
                        tags={long_key: long_value})

        metric = store.export()[0]
        self.assertEqual(list(metric.tags.keys()), ['k' * 50])
        self.assertEqual(list(metric.tags.values()), ['v' * 250])

    def test_non_string_tags_stringified(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        store.set_gauge('m1', 1.0, measurement_ts=10, tags={1: 2})
        self.assertEqual(store.export()[0].tags, {'1': '2'})


class MetricStoreCleanupTest(unittest.TestCase):
    def test_cleanup_expires_stale_metrics(self):
        store = MetricStore()
        now = time.time_ns()
        expired_ts = now - METRIC_EXPIRY_NS - 1_000_000_000

        # Metrics whose datapoint ts is older than the expiry window
        # disappear entirely.
        store.set_gauge('stale', 1.0, measurement_ts=expired_ts)
        store.set_counter('stale_counter', 5, measurement_ts=expired_ts)
        # Fresh metrics survive.
        store.set_counter('fresh_counter', 5, measurement_ts=now)

        store._last_cleanup_ts = 0
        store.set_gauge('fresh_gauge', 2.0, measurement_ts=now)

        exported = {m.name: m for m in store.export()}
        self.assertEqual(set(exported.keys()), {'fresh_gauge', 'fresh_counter'})
        self.assertEqual(exported['fresh_gauge'].datapoint,
                         {'ts': now, 'value': 2.0})

    def test_cleanup_throttled_to_once_per_interval(self):
        store = MetricStore()
        now = time.time_ns()
        expired_ts = now - METRIC_EXPIRY_NS - 1_000_000_000

        # A cleanup just ran; sets within the interval must not clean.
        store._last_cleanup_ts = time.time_ns()
        store.set_gauge('stale', 1.0, measurement_ts=expired_ts)
        store.set_gauge('fresh', 2.0, measurement_ts=now)

        exported = {m.name: m for m in store.export()}
        self.assertEqual(set(exported.keys()), {'stale', 'fresh'})


class MetricStoreCapTest(unittest.TestCase):
    def test_new_series_beyond_cap_dropped_silently(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        with patch.object(metrics_module, 'MAX_METRICS', 3):
            for i in range(5):
                store.set_gauge(f'm{i}', 1.0, measurement_ts=10)
            # Existing series still accept new snapshots.
            store.set_gauge('m0', 2.0, measurement_ts=20)

        exported = {m.name: m for m in store.export()}
        self.assertEqual(set(exported.keys()), {'m0', 'm1', 'm2'})
        self.assertEqual(exported['m0'].datapoint, {'ts': 20, 'value': 2.0})


class MetricStoreExportTest(unittest.TestCase):
    def _snapshot(self, exported):
        return sorted(
            (m.name, m.type, tuple(sorted(m.tags.items())),
             tuple(sorted((k, tuple(v) if isinstance(v, list) else
                           (tuple(sorted(v.items())) if isinstance(v, dict)
                            else v))
                          for k, v in m.datapoint.items())))
            for m in exported)

    def test_export_non_destructive(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        store.set_gauge('g1', 1.0, measurement_ts=10, tags={'t': '1'})
        store.set_counter('c1', 5, measurement_ts=10)
        store.set_histogram('h1', bins=[1], counts=[1], measurement_ts=10)
        store.set_profile('p1', frames={'a': 1}, measurement_ts=10)

        first = store.export()
        second = store.export()
        self.assertEqual(self._snapshot(first), self._snapshot(second))

    def test_mutating_export_does_not_affect_store(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        store.set_gauge('g1', 1.0, measurement_ts=10, tags={'t': '1'})
        store.set_histogram('h1', count=1, sum_val=2.0, measurement_ts=10)
        store.set_profile('p1', frames={'a': 1}, measurement_ts=10)

        exported = store.export()
        for metric in exported:
            metric.tags['mutated'] = 'yes'
            metric.datapoint['ts'] = 999
            if 'count' in metric.datapoint:
                metric.datapoint['count'] = 999
            if 'frames' in metric.datapoint:
                metric.datapoint['frames']['a'] = 999

        fresh = {m.name: m for m in store.export()}
        self.assertEqual(fresh['g1'].tags, {'t': '1'})
        self.assertEqual(fresh['g1'].datapoint, {'ts': 10, 'value': 1.0})
        self.assertEqual(fresh['h1'].datapoint['ts'], 10)
        self.assertEqual(fresh['h1'].datapoint['count'], 1)
        self.assertEqual(fresh['p1'].datapoint['frames'], {'a': 1})


class MetricStoreThreadSafetyTest(unittest.TestCase):
    def test_concurrent_set_gauge(self):
        store = MetricStore()

        def set_many():
            for i in range(1000):
                store.set_gauge('m1', float(i), measurement_ts=time.time_ns())

        threads = [threading.Thread(target=set_many) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        exported = store.export()
        self.assertEqual(len(exported), 1)
        metric = exported[0]
        self.assertIsNotNone(metric.datapoint)
        self.assertIn('value', metric.datapoint)


class MetricStoreNameCapTest(unittest.TestCase):
    def test_metric_name_truncated(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        store.set_gauge('n' * 1000, 1.0, measurement_ts=10)
        self.assertEqual(store.export()[0].name, 'n' * 250)

    def test_frame_names_truncated_and_merged(self):
        store = MetricStore()
        store._last_cleanup_ts = time.time_ns()
        prefix = 'f' * 4096
        store.set_profile('p1', frames={prefix + 'a': 2, prefix + 'b': 3, 'short': 1},
                          samples={prefix + 'a': 1, prefix + 'b': 1, 'short': 4},
                          measurement_ts=10)
        datapoint = store.export()[0].datapoint
        self.assertEqual(datapoint['frames'], {prefix: 5, 'short': 1})
        self.assertEqual(datapoint['samples'], {prefix: 2, 'short': 4})



if __name__ == '__main__':
    unittest.main()
