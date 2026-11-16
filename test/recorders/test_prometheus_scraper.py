import unittest
from unittest.mock import patch

from graphsignal.recorders.prometheus_scraper import (
    INITIAL_DETECT_DELAY_SEC, MAX_DETECT_DELAY_SEC, PrometheusScraper,
    build_metrics_endpoint, buckets_to_bins, format_metrics_host,
    iter_metric_groups, looks_like_prometheus, normalize_metrics_path)

_BODY = """# HELP vllm_num_requests_running Number of running requests.
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{model_name="m"} 3.0
"""

_ENDPOINT = 'http://127.0.0.1:8000/metrics'


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _scraper(interval=0.0):
    clock = FakeClock()
    scraper = PrometheusScraper(_ENDPOINT, interval=interval, clock=clock)
    scraper.start()
    return scraper, clock


class EndpointBuildingTest(unittest.TestCase):
    def test_normalize_metrics_path(self):
        self.assertEqual(normalize_metrics_path(None), '/metrics')
        self.assertEqual(normalize_metrics_path('/prometheus/metrics'),
                         '/prometheus/metrics')
        self.assertEqual(normalize_metrics_path('prometheus/metrics'),
                         '/prometheus/metrics')

    def test_build_metrics_endpoint(self):
        self.assertEqual(build_metrics_endpoint(8000),
                         'http://127.0.0.1:8000/metrics')
        self.assertEqual(
            build_metrics_endpoint(8000, metrics_path='/prometheus/metrics',
                                   metrics_host='localhost'),
            'http://localhost:8000/prometheus/metrics')

    def test_format_metrics_host_ipv6(self):
        self.assertEqual(format_metrics_host('::1'), '[::1]')


class BucketsToBinsTest(unittest.TestCase):
    def test_cumulative_buckets_become_noncumulative_bins(self):
        bins, counts = buckets_to_bins([('0.5', 4), ('1.0', 7), ('2.0', 9)])
        self.assertEqual(bins, [0.5, 1.0, 2.0])
        self.assertEqual(counts, [4, 3, 2])

    def test_inf_remainder_folded_into_last_finite_bound(self):
        # The last finite bucket has a zero delta (omitted), so the +Inf
        # remainder lands in a new bin at the largest finite bound.
        bins, counts = buckets_to_bins([('0.5', 4), ('1.0', 4), ('+Inf', 10)])
        self.assertEqual(bins, [0.5, 1.0])
        self.assertEqual(counts, [4, 6])

    def test_inf_remainder_merged_into_emitted_last_bound(self):
        bins, counts = buckets_to_bins([('0.5', 4), ('+Inf', 10)])
        self.assertEqual(bins, [0.5])
        self.assertEqual(counts, [10])

    def test_zero_delta_buckets_omitted(self):
        bins, counts = buckets_to_bins(
            [('0.5', 4), ('1.0', 4), ('2.0', 6), ('+Inf', 6)])
        self.assertEqual(bins, [0.5, 2.0])
        self.assertEqual(counts, [4, 2])

    def test_unusable_buckets_return_none(self):
        self.assertEqual(buckets_to_bins(None), (None, None))
        self.assertEqual(buckets_to_bins([]), (None, None))
        # Only a +Inf bucket: no finite bounds to place counts on.
        self.assertEqual(buckets_to_bins([('+Inf', 5)]), (None, None))
        # Unparsable bounds and non-finite values are skipped.
        self.assertEqual(buckets_to_bins([('nope', 5)]), (None, None))
        self.assertEqual(buckets_to_bins([('0.5', float('nan'))]),
                         (None, None))
        # All-zero cumulative values produce no bins.
        self.assertEqual(buckets_to_bins([('0.5', 0), ('+Inf', 0)]),
                         (None, None))


class LooksLikePrometheusTest(unittest.TestCase):
    def test_detection(self):
        self.assertTrue(looks_like_prometheus(_BODY))
        self.assertTrue(looks_like_prometheus('up 1\n'))
        self.assertFalse(looks_like_prometheus(''))
        self.assertFalse(looks_like_prometheus('<html>nope</html>'))


class IterMetricGroupsTest(unittest.TestCase):
    def test_groups_by_label_set_and_collects_buckets(self):
        body = """# TYPE lat histogram
lat_bucket{le="0.5",model_name="a"} 1
lat_bucket{le="+Inf",model_name="a"} 2
lat_count{model_name="a"} 2
lat_sum{model_name="a"} 1.5
lat_bucket{le="+Inf",model_name="b"} 4
lat_count{model_name="b"} 4
lat_sum{model_name="b"} 3.0
"""
        groups = {g.tags['model_name']: g for g in iter_metric_groups(body)}
        self.assertEqual(set(groups), {'a', 'b'})
        a = groups['a']
        self.assertEqual((a.name, a.type), ('lat', 'histogram'))
        self.assertEqual(a.buckets, [('0.5', 1.0), ('+Inf', 2.0)])
        self.assertEqual(set(a.samples), {'lat_count', 'lat_sum'})
        self.assertEqual(groups['b'].buckets, [('+Inf', 4.0)])

    def test_strips_exposition_labels_and_skips_gt_buckets(self):
        body = """# TYPE g gauge
g{gt="0",le="1",model_name="m"} 2.0
g{pid="7",model_name="m"} 5.0
"""
        groups = list(iter_metric_groups(body))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].tags, {'model_name': 'm'})
        self.assertEqual(groups[0].samples['g'].value, 5.0)
        self.assertIsNone(groups[0].buckets)


class PrometheusScraperTest(unittest.TestCase):
    def test_waits_initial_delay_before_first_fetch(self):
        scraper, clock = _scraper()
        with patch.object(scraper, '_fetch', return_value=_BODY) as fetch_m:
            self.assertIsNone(scraper.scrape())
            fetch_m.assert_not_called()
            clock.now += INITIAL_DETECT_DELAY_SEC
            self.assertEqual(scraper.scrape(), _BODY)
        fetch_m.assert_called_once_with(_ENDPOINT)
        self.assertTrue(scraper.verified)

    def test_verified_endpoint_is_fetched_every_call_by_default(self):
        scraper, clock = _scraper()
        clock.now += INITIAL_DETECT_DELAY_SEC
        with patch.object(scraper, '_fetch', return_value=_BODY) as fetch_m:
            for _ in range(3):
                self.assertEqual(scraper.scrape(), _BODY)
        self.assertEqual(fetch_m.call_count, 3)

    def test_interval_throttles_verified_fetches(self):
        scraper, clock = _scraper(interval=5.0)
        clock.now += INITIAL_DETECT_DELAY_SEC
        with patch.object(scraper, '_fetch', return_value=_BODY) as fetch_m:
            self.assertEqual(scraper.scrape(), _BODY)
            clock.now += 4.0
            self.assertIsNone(scraper.scrape())
            clock.now += 1.0
            self.assertEqual(scraper.scrape(), _BODY)
        self.assertEqual(fetch_m.call_count, 2)

    def test_non_prometheus_body_is_not_returned(self):
        scraper, clock = _scraper()
        clock.now += INITIAL_DETECT_DELAY_SEC
        with patch.object(scraper, '_fetch', return_value='<html>nope</html>'):
            self.assertIsNone(scraper.scrape())
        self.assertFalse(scraper.verified)

    def test_fetch_failure_backs_off_exponentially_without_raising(self):
        scraper, clock = _scraper()
        clock.now += INITIAL_DETECT_DELAY_SEC
        with patch.object(scraper, '_fetch', side_effect=OSError('refused')) as fetch_m:
            delays = []
            for _ in range(7):
                self.assertIsNone(scraper.scrape())  # must not raise
                delay = scraper._next_attempt_ts - clock.now
                delays.append(delay)
                # Too early: no fetch until the delay has passed.
                clock.now += delay - 0.5
                self.assertIsNone(scraper.scrape())
                clock.now += 0.5
        self.assertEqual(fetch_m.call_count, 7)
        self.assertEqual(delays, [4.0, 8.0, 16.0, 32.0] + [MAX_DETECT_DELAY_SEC] * 3)
        self.assertFalse(scraper.verified)

    def test_failure_after_verification_drops_back_to_unverified(self):
        scraper, clock = _scraper()
        clock.now += INITIAL_DETECT_DELAY_SEC
        with patch.object(scraper, '_fetch', return_value=_BODY):
            scraper.scrape()
        self.assertTrue(scraper.verified)
        with patch.object(scraper, '_fetch', side_effect=OSError('reset')):
            self.assertIsNone(scraper.scrape())
        self.assertFalse(scraper.verified)
        # Backoff applies again: the next call is too early to fetch.
        with patch.object(scraper, '_fetch', return_value=_BODY) as fetch_m:
            self.assertIsNone(scraper.scrape())
        fetch_m.assert_not_called()


if __name__ == '__main__':
    unittest.main()
