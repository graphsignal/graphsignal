import unittest
from unittest.mock import patch

from prometheus_client.parser import text_string_to_metric_families

import graphsignal.watcher
from graphsignal import version
from graphsignal.recorders.prometheus_recorder import PrometheusRecorder
from graphsignal.signals.prometheus import (
    render_metrics, sanitize_label_name, sanitize_name)
from test.test_utils import configure_test_watcher, find_metric

KERNEL_SYMBOL = '_ZN4vllm15rms_norm_kernelIN3c108BFloat16EEEvPT_PKS3_fii'


def parse(text):
    """{family name: (type, {(sample name, frozenset(labels)): value})}"""
    families = {}
    for family in text_string_to_metric_families(text):
        samples = {(s.name, frozenset(s.labels.items())): s.value
                   for s in family.samples}
        families[family.name] = (family.type, samples)
    return families


def labels(**kwargs):
    return frozenset(kwargs.items())


class SanitizeTest(unittest.TestCase):
    def test_metric_names(self):
        self.assertEqual(sanitize_name('user.hist'), 'user_hist')
        self.assertEqual(sanitize_name('vllm:num_requests'), 'vllm:num_requests')
        self.assertEqual(sanitize_name('9lives-x'), '_9lives_x')

    def test_label_names(self):
        self.assertEqual(sanitize_label_name('process.pid'), 'process_pid')
        # Colons are valid in metric names only.
        self.assertEqual(sanitize_label_name('a:b'), 'a_b')
        self.assertEqual(sanitize_label_name('0x'), '_0x')
        self.assertEqual(sanitize_label_name('__name__'), 'tag__name__')


class RenderMetricsTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.instance_id = self.watcher.get_tag('instance.id')
        # Fixed context tags: the host recorder adds its own (host.name,
        # container.id inside a container) whenever it gets to run.
        self._tags = patch.object(self.watcher, 'tags', return_value={
            'instance.id': self.instance_id, 'host.name': 'h1'})
        self._tags.start()

    def tearDown(self):
        self._tags.stop()
        graphsignal.watcher.shutdown()

    def _render(self):
        return parse(render_metrics())

    def test_empty_store_renders_info_gauge_only(self):
        families = self._render()
        self.assertEqual(list(families), ['graphsignal_profiler_info'])
        family_type, samples = families['graphsignal_profiler_info']
        self.assertEqual(family_type, 'gauge')
        self.assertEqual(samples, {
            ('graphsignal_profiler_info',
             labels(host_name='h1', version=version.__version__,
                    instance_id=self.instance_id)): 1.0})

    def test_gauge(self):
        self.watcher.set_gauge('process_cpu_usage_percent', 12.5, 1000,
                               tags={'process.pid': '42'})
        family_type, samples = self._render()['process_cpu_usage_percent']
        self.assertEqual(family_type, 'gauge')
        self.assertEqual(samples, {
            ('process_cpu_usage_percent',
             labels(process_pid='42', host_name='h1')): 12.5})

    def test_instance_id_is_not_a_series_label(self):
        self.watcher.set_gauge('g1', 1, 1000)
        for name, (_, samples) in self._render().items():
            if name == 'graphsignal_profiler_info':
                continue
            for _, sample_labels in samples:
                self.assertNotIn('instance_id', dict(sample_labels))

    def test_metric_tag_beats_context_tag(self):
        self.watcher.set_gauge('g1', 1, 1000, tags={'host.name': 'mine'})
        _, samples = self._render()['g1']
        self.assertEqual(samples, {('g1', labels(host_name='mine')): 1.0})

    def test_counter_gets_total_suffix(self):
        self.watcher.set_counter('vllm:prompt_tokens', 100, 1000)
        self.watcher.set_counter('already_total', 5, 1000)
        text = render_metrics()
        self.assertIn('# TYPE vllm:prompt_tokens_total counter', text)
        self.assertIn('# TYPE already_total counter', text)
        self.assertNotIn('already_total_total', text)
        families = parse(text)
        # The parser files counters under the name without `_total`.
        self.assertEqual(families['vllm:prompt_tokens'], ('counter', {
            ('vllm:prompt_tokens_total', labels(host_name='h1')): 100.0}))
        self.assertEqual(families['already'], ('counter', {
            ('already_total', labels(host_name='h1')): 5.0}))

    def test_histogram_bins_are_le_bounds(self):
        # A bin value is the bin's inclusive upper bound, from either source:
        # the Prometheus recorder's `le` bounds, or the native writers'
        # log-linear bins (serialized as their last value). Counts are per bin.
        self.watcher.set_histogram('vllm:ttft', bins=[0.1, 1.0], counts=[2, 3],
                                   measurement_ts=1000, count=5, sum_val=2.7)
        family_type, samples = self._render()['vllm:ttft']
        self.assertEqual(family_type, 'histogram')
        self.assertEqual(samples, {
            ('vllm:ttft_bucket', labels(host_name='h1', le='0.1')): 2.0,
            ('vllm:ttft_bucket', labels(host_name='h1', le='1.0')): 5.0,
            ('vllm:ttft_bucket', labels(host_name='h1', le='+Inf')): 5.0,
            ('vllm:ttft_count', labels(host_name='h1')): 5.0,
            ('vllm:ttft_sum', labels(host_name='h1')): 2.7,
        })

    def test_native_histogram_buckets(self):
        # Native bins 0..0, 8..9 and 12..13, as the writers serialize them.
        self.watcher.set_histogram('user.hist', bins=[0, 9, 13], counts=[1, 2, 3],
                                   measurement_ts=1000, count=6, sum_val=60)
        _, samples = self._render()['user_hist']
        buckets = {dict(key[1])['le']: value for key, value in samples.items()
                   if key[0] == 'user_hist_bucket'}
        self.assertEqual(buckets, {'0': 1.0, '9': 3.0, '13': 6.0, '+Inf': 6.0})

    def test_bucket_only_histogram_has_no_sum(self):
        self.watcher.set_histogram('h1', bins=[5, 20], counts=[1, 2],
                                   measurement_ts=1000)
        _, samples = self._render()['h1']
        self.assertEqual(samples, {
            ('h1_bucket', labels(host_name='h1', le='5')): 1.0,
            ('h1_bucket', labels(host_name='h1', le='20')): 3.0,
            ('h1_bucket', labels(host_name='h1', le='+Inf')): 3.0,
            ('h1_count', labels(host_name='h1')): 3.0,
        })

    def test_totals_only_histogram(self):
        # A Prometheus summary scraped from the engine: no buckets.
        self.watcher.set_histogram('s1', measurement_ts=1000, count=4, sum_val=9.5)
        family_type, samples = self._render()['s1']
        self.assertEqual(family_type, 'histogram')
        self.assertEqual(samples, {
            ('s1_bucket', labels(host_name='h1', le='+Inf')): 4.0,
            ('s1_count', labels(host_name='h1')): 4.0,
            ('s1_sum', labels(host_name='h1')): 9.5,
        })

    def test_histogram_min_max_as_gauges(self):
        self.watcher.set_histogram('h1', bins=[9], counts=[2], measurement_ts=1000,
                                   count=2, sum_val=17, min_val=8, max_val=9)
        families = self._render()
        self.assertEqual(families['h1_min'], (
            'gauge', {('h1_min', labels(host_name='h1')): 8.0}))
        self.assertEqual(families['h1_max'], (
            'gauge', {('h1_max', labels(host_name='h1')): 9.0}))

    def test_kernel_profile(self):
        self.watcher.set_profile(
            'cuda_kernels_nanoseconds',
            frames={KERNEL_SYMBOL: 1000, 'mystery': 10},
            samples={KERNEL_SYMBOL: 4},
            measurement_ts=1000, tags={'process.pid': '42'})
        families = self._render()
        self.assertEqual(families['cuda_kernels_nanoseconds'], ('counter', {
            ('cuda_kernels_nanoseconds_total',
             labels(process_pid='42', host_name='h1',
                    kernel='vllm::rms_norm_kernel', symbol=KERNEL_SYMBOL,
                    category='norm', library='vllm')): 1000.0,
            # An unknown library is left off rather than sent empty.
            ('cuda_kernels_nanoseconds_total',
             labels(process_pid='42', host_name='h1', kernel='mystery',
                    symbol='mystery', category='other')): 10.0,
        }))
        _, samples = families['cuda_kernels_nanoseconds_samples']
        self.assertEqual(sorted(samples.values()), [0.0, 4.0])

    def test_graph_profile(self):
        self.watcher.set_profile('cuda_graphs_nanoseconds',
                                 frames={'0123456789abcdef': 50},
                                 samples={'0123456789abcdef': 2},
                                 measurement_ts=1000)
        families = self._render()
        self.assertEqual(families['cuda_graphs_nanoseconds'], ('counter', {
            ('cuda_graphs_nanoseconds_total',
             labels(host_name='h1', graph='0123456789abcdef')): 50.0}))
        self.assertEqual(families['cuda_graphs_nanoseconds_samples'], ('counter', {
            ('cuda_graphs_nanoseconds_samples_total',
             labels(host_name='h1', graph='0123456789abcdef')): 2.0}))

    def test_plain_profile(self):
        self.watcher.set_profile('cuda_memcpy_nanoseconds',
                                 frames={'host_to_device': 70},
                                 samples={'host_to_device': 7},
                                 measurement_ts=1000)
        _, samples = self._render()['cuda_memcpy_nanoseconds']
        self.assertEqual(samples, {
            ('cuda_memcpy_nanoseconds_total',
             labels(host_name='h1', frame='host_to_device')): 70.0})

    def test_tag_colliding_with_own_label_is_exported(self):
        self.watcher.set_profile('cuda_kernels_nanoseconds', frames={'k': 1},
                                 measurement_ts=1000, tags={'kernel': 'tagged'})
        self.watcher.set_histogram('h1', bins=[1], counts=[1], measurement_ts=1000,
                                   tags={'le': 'x'})
        families = self._render()
        (_, kernel_labels), = families['cuda_kernels_nanoseconds'][1]
        self.assertEqual(dict(kernel_labels)['kernel'], 'k')
        self.assertEqual(dict(kernel_labels)['exported_kernel'], 'tagged')
        count_key = ('h1_count', labels(exported_le='x', host_name='h1'))
        self.assertIn(count_key, families['h1'][1])

    def test_tags_sanitizing_to_one_label_keep_the_first(self):
        self.watcher.set_gauge('g1', 1, 1000, tags={'a.b': 'first', 'a_b': 'second'})
        _, samples = self._render()['g1']
        self.assertEqual(samples, {('g1', labels(a_b='first', host_name='h1')): 1.0})

    def test_label_values_escaped(self):
        value = 'back\\slash "quoted"\nnewline'
        self.watcher.set_gauge('g1', 1, 1000, tags={'t': value})
        _, samples = self._render()['g1']
        self.assertEqual(samples, {('g1', labels(t=value, host_name='h1')): 1.0})

    def test_family_name_collisions_keep_the_first(self):
        # counter `foo` renders `foo_total`; a gauge `foo_total` would repeat
        # the family. A gauge `h_count` would repeat a histogram's series.
        self.watcher.set_counter('foo', 1, 1000)
        self.watcher.set_gauge('foo_total', 2, 1000)
        self.watcher.set_histogram('h', bins=[1], counts=[1], measurement_ts=1000)
        self.watcher.set_gauge('h_count', 3, 1000)
        text = render_metrics()
        families = parse(text)
        self.assertEqual(families['foo'][0], 'counter')
        self.assertEqual(families['h'][0], 'histogram')
        self.assertNotIn('h_count', families)
        self.assertEqual(text.count('# TYPE foo_total '), 1)

    def test_unrenderable_metric_is_skipped(self):
        self.watcher.set_gauge('bad', 'not a number', 1000)
        self.watcher.set_gauge('good', 1, 1000)
        families = self._render()
        self.assertNotIn('bad', families)
        self.assertIn('good', families)

    def test_special_float_values(self):
        self.watcher.set_gauge('nan', float('nan'), 1000)
        self.watcher.set_gauge('inf', float('inf'), 1000)
        text = render_metrics()
        self.assertIn('nan{host_name="h1"} NaN', text)
        self.assertIn('inf{host_name="h1"} +Inf', text)

    def test_self_scrape_reproduces_values(self):
        # The Prometheus recorder reading this endpoint gets back what the
        # store holds, under Prometheus names: sanitized tag keys, context
        # tags as labels, counters without `_total`.
        self.watcher.set_gauge('g1', 12.5, 1000, tags={'process.pid': '42'})
        self.watcher.set_counter('c1', 100, 1000)
        self.watcher.set_histogram('h1', bins=[0.5, 2.0], counts=[1, 3],
                                   measurement_ts=1000, count=4, sum_val=6.5)
        text = render_metrics()
        graphsignal.watcher.shutdown()

        second = configure_test_watcher()
        PrometheusRecorder()._parse_and_emit(text)
        exported = second.metric_store().export()

        gauge = find_metric(exported, 'g1', {'process_pid': '42', 'host_name': 'h1'})
        self.assertEqual(gauge.datapoint['value'], 12.5)
        counter = find_metric(exported, 'c1', {'host_name': 'h1'}, metric_type='counter')
        self.assertEqual(counter.datapoint['total'], 100)
        histogram = find_metric(exported, 'h1', {'host_name': 'h1'},
                                metric_type='histogram')
        self.assertEqual(histogram.datapoint['bins'], [0.5, 2.0])
        self.assertEqual(histogram.datapoint['counts'], [1, 3])
        self.assertEqual(histogram.datapoint['count'], 4)
        self.assertEqual(histogram.datapoint['sum'], 6.5)
        info = find_metric(exported, 'graphsignal_profiler_info')
        self.assertEqual(info.tags['instance_id'], self.instance_id)


class RenderMetricsUnconfiguredTest(unittest.TestCase):
    def test_empty_when_watcher_not_configured(self):
        self.assertFalse(graphsignal.watcher.is_configured())
        self.assertEqual(render_metrics(), '')


if __name__ == '__main__':
    unittest.main()
