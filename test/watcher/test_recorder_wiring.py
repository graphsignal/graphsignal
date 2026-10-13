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
                         'ProcessRecorder', 'ShmRecorder'} <= names, names)

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
