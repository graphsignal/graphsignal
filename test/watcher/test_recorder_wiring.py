import os
import unittest
from unittest import mock

import graphsignal.watcher
from test.test_utils import configure_test_watcher


class RecorderWiringTest(unittest.TestCase):
    """on_target_created wires every recorder on its own: a recorder whose
    module cannot be imported, or whose constructor raises, must cost that
    one recorder and leave the rest in place."""

    def setUp(self):
        self.watcher = configure_test_watcher()
        # The pid monitor fires on_target_created on its own once it sees the
        # (own-process) target alive; stop it so the only wiring observed is
        # the explicit call under test, and start from no recorders.
        self.watcher._pid_monitor.shutdown()
        with self.watcher._recorders_lock:
            self.watcher._global_recorders.clear()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _recorder_names(self):
        return {type(r).__name__ for r in self.watcher.recorders()}

    def test_all_recorders_wired(self):
        self.watcher.on_target_created(['python', 'app.py'])
        names = self._recorder_names()
        self.assertTrue({'HostRecorder', 'NVMLRecorder', 'AmdSmiRecorder',
                         'NicRecorder', 'ProcessRecorder', 'ShmRecorder'} <= names, names)

    def test_engine_recorder_wired_by_health_port_alone(self):
        self.watcher._metrics_port = None
        self.watcher._health_port = 8000
        self.watcher._health_path = '/health'
        self.watcher.on_target_created(['python', 'app.py'])
        engine = [r for r in self.watcher.recorders()
                  if type(r).__name__ == 'EngineRecorder']
        self.assertEqual(len(engine), 1)
        self.assertIsNone(engine[0]._scraper)
        self.assertEqual(engine[0]._health_url, 'http://127.0.0.1:8000/health')

    def test_dcgm_recorder_wired_only_with_url(self):
        self.watcher.on_target_created(['python', 'app.py'])
        self.assertNotIn('DcgmRecorder', self._recorder_names())
        with self.watcher._recorders_lock:
            self.watcher._global_recorders.clear()
        self.watcher._dcgm_exporter_url = 'http://10.0.0.7:9400/metrics'
        self.watcher.on_target_created(['python', 'app.py'])
        dcgm = [r for r in self.watcher.recorders()
                if type(r).__name__ == 'DcgmRecorder']
        self.assertEqual(len(dcgm), 1)
        self.assertEqual(dcgm[0]._scraper.endpoint, 'http://10.0.0.7:9400/metrics')

    def test_no_engine_recorder_without_ports(self):
        self.watcher._metrics_port = None
        self.watcher._health_port = None
        self.watcher.on_target_created(['python', 'app.py'])
        self.assertNotIn('EngineRecorder', self._recorder_names())

    def test_failing_import_does_not_block_the_others(self):
        import importlib
        real_import = importlib.import_module

        def failing_import(name, *args, **kwargs):
            if name == 'graphsignal.recorders.nvml_recorder':
                raise ModuleNotFoundError("No module named 'pynvml'")
            return real_import(name, *args, **kwargs)

        with mock.patch('importlib.import_module', side_effect=failing_import):
            with self.assertLogs('graphsignal', level='ERROR'):
                self.watcher.on_target_created(['python', 'app.py'])
        names = self._recorder_names()
        self.assertNotIn('NVMLRecorder', names)
        self.assertTrue({'HostRecorder', 'AmdSmiRecorder', 'ProcessRecorder',
                         'ShmRecorder'} <= names, names)

    def test_failing_constructor_does_not_block_the_others(self):
        with mock.patch('graphsignal.recorders.amdsmi_recorder.AmdSmiRecorder.__init__',
                        side_effect=RuntimeError('boom')):
            with self.assertLogs('graphsignal', level='ERROR'):
                self.watcher.on_target_created(['python', 'app.py'])
        names = self._recorder_names()
        self.assertNotIn('AmdSmiRecorder', names)
        self.assertTrue({'HostRecorder', 'NVMLRecorder', 'ProcessRecorder',
                         'ShmRecorder'} <= names, names)


if __name__ == '__main__':
    unittest.main()
