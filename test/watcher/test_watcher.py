import os
import socket
import unittest
import urllib.request

import graphsignal.watcher
from graphsignal.watcher.watcher import Watcher
from graphsignal.collector.collector import Collector
from test.test_utils import (
    clear_graphsignal_env, configure_test_watcher, free_port)


class WatcherConfigureTest(unittest.TestCase):
    def setUp(self):
        clear_graphsignal_env()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _configure(self, **kwargs):
        return configure_test_watcher(**kwargs)

    def test_configure_without_api_key(self):
        watcher = self._configure()

        self.assertTrue(graphsignal.watcher.is_configured())
        self.assertIsInstance(watcher, Watcher)
        self.assertIsNone(watcher.collector())
        self.assertEqual(watcher.target_pid(), os.getpid())

    def test_configure_with_api_key_enables_collector(self):
        watcher = self._configure(api_key='k')
        self.assertIsInstance(watcher.collector(), Collector)

    def test_instance_id_tag_set(self):
        watcher = self._configure()
        tags = watcher.tags()
        self.assertIn('instance.id', tags)
        self.assertEqual(len(tags['instance.id']), 12)

    def test_instance_id_tag_not_overwritten(self):
        watcher = self._configure(tags={'instance.id': 'my-instance-1'})
        self.assertEqual(watcher.get_tag('instance.id'), 'my-instance-1')

    def test_store_accessors(self):
        watcher = self._configure()
        self.assertIsNotNone(watcher.metric_store())
        self.assertIsNotNone(watcher.log_store())
        self.assertIsNotNone(watcher.resource_store())
        self.assertIsNotNone(watcher.signals_endpoint())

    def test_signals_endpoint_running_on_configured_port(self):
        port = free_port()
        watcher = self._configure(listen_port=port)

        endpoint = watcher.signals_endpoint()
        self.assertTrue(endpoint.is_running())
        self.assertEqual(endpoint.port(), port)
        with urllib.request.urlopen(
                f'http://127.0.0.1:{port}/health', timeout=5) as resp:
            self.assertEqual(resp.status, 200)

    def test_watcher_raises_when_not_configured(self):
        self.assertFalse(graphsignal.watcher.is_configured())
        with self.assertRaises(RuntimeError):
            graphsignal.watcher.watcher()

    def test_shutdown_clears_singleton(self):
        self._configure()
        graphsignal.watcher.shutdown()

        self.assertFalse(graphsignal.watcher.is_configured())
        with self.assertRaises(RuntimeError):
            graphsignal.watcher.watcher()

    def test_second_configure_warns_and_keeps_first(self):
        first = self._configure(tags={'t1': '1'})
        with self.assertLogs('graphsignal', level='WARNING') as cm:
            graphsignal.watcher.configure(
                listen_port=free_port(), tags={'t1': '2'})

        self.assertTrue(any('already configured' in msg for msg in cm.output))
        self.assertIs(graphsignal.watcher.watcher(), first)
        self.assertEqual(graphsignal.watcher.watcher().get_tag('t1'), '1')

    def test_tag_operations(self):
        watcher = self._configure()
        watcher.set_tag('k1', 'v1')
        self.assertEqual(watcher.get_tag('k1'), 'v1')
        self.assertEqual(watcher.tags().get('k1'), 'v1')
        watcher.remove_tag('k1')
        self.assertIsNone(watcher.get_tag('k1'))

    def test_metric_passthroughs(self):
        watcher = self._configure()
        watcher.set_gauge('g1', 1.5, measurement_ts=10)
        watcher.set_counter('c1', 3, measurement_ts=10)
        watcher.set_histogram('s1', count=2, sum_val=4.0, measurement_ts=10)
        watcher.set_histogram('h1', bins=[1, 3], counts=[1, 1],
                              measurement_ts=10, count=2, sum_val=4.0,
                              min_val=1, max_val=3)
        watcher.set_profile('p1', frames={'f1': 5.0}, samples={'f1': 1}, measurement_ts=10)

        exported = {m.name: m for m in watcher.metric_store().export()}
        self.assertEqual(exported['g1'].datapoint, {'ts': 10, 'value': 1.5})
        self.assertEqual(exported['c1'].datapoint, {'ts': 10, 'total': 3})
        self.assertEqual(exported['s1'].datapoint,
                         {'ts': 10, 'count': 2, 'sum': 4.0})
        self.assertEqual(exported['h1'].datapoint,
                         {'ts': 10, 'bins': [1, 3], 'counts': [1, 1],
                          'count': 2, 'sum': 4.0, 'min': 1, 'max': 3})
        self.assertEqual(exported['p1'].datapoint,
                         {'ts': 10, 'frames': {'f1': 5}, 'samples': {'f1': 1}})

    def test_log_and_resource_passthroughs(self):
        watcher = self._configure()
        watcher.log_message('hello', level='error', tags={'t1': '1'})
        watcher.update_resource('process', tags={'process.pid': '1'},
                                attributes={'a': '1'})

        log_entries = watcher.log_store().export()
        self.assertTrue(any(e['message'] == 'hello' and e['level'] == 'error'
                            for e in log_entries))
        resources = watcher.resource_store().export()
        self.assertTrue(any(r['kind'] == 'process' for r in resources))

    def test_listen_port_conflict_fails_setup(self):
        port = free_port()
        first = self._configure(listen_port=port)
        self.assertTrue(first.signals_endpoint().is_running())

        second = Watcher(target_pid=os.getpid(), listen_port=port)
        second._auto_tick = False
        try:
            with self.assertRaises(OSError):
                second.setup()
        finally:
            second.shutdown()

    def test_configure_on_busy_port_raises_and_leaves_nothing_configured(self):
        port = free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(('127.0.0.1', port))
            holder.listen(1)
            with self.assertRaises(OSError):
                self._configure(listen_port=port)
        self.assertFalse(graphsignal.watcher.is_configured())


if __name__ == '__main__':
    unittest.main()
