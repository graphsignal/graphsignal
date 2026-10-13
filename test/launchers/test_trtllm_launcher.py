import unittest

from graphsignal.launchers import trtllm_launcher as trtllm_mod
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher

from test.launchers._helpers import LaunchFixture


class TrtllmMatchTest(unittest.TestCase):
    def test_matches(self):
        self.assertTrue(TrtllmLauncher(['trtllm', 'serve']).match())
        self.assertTrue(TrtllmLauncher(['trtllm-serve', '--model', 'm']).match())
        self.assertTrue(TrtllmLauncher(['trtllm-llmapi-launch']).match())
        self.assertTrue(TrtllmLauncher(['/usr/bin/trtllm-serve']).match())

    def test_does_not_match(self):
        self.assertFalse(TrtllmLauncher(['python', 'app.py']).match())
        self.assertFalse(TrtllmLauncher(['trtllm-other']).match())


class TrtllmLaunchTest(unittest.TestCase):
    def test_argv_unchanged_with_prometheus_path(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', '--model', 'm', '--port', '8000'],
            listen_port=18302)
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()

        fx.cupti_env_m.assert_called_once_with(cuda_graph_trace=None)
        fx.rocm_env_m.assert_called_once_with(cuda_graph_trace=None, rocm_kernel_trace=None)
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', '--model', 'm', '--port', '8000'],
            metrics_port=8000, metrics_path='/prometheus/metrics',
            metrics_host='localhost', listen_host=None, listen_port=18302)

    def test_default_metrics_port_and_host(self):
        launcher = TrtllmLauncher(['trtllm-serve', '--model', 'm'])
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', '--model', 'm'],
            metrics_port=8000, metrics_path='/prometheus/metrics',
            metrics_host='localhost', listen_host=None, listen_port=None)

    def test_metrics_port_from_engine_args(self):
        launcher = TrtllmLauncher(['trtllm-serve', 'm', '--port', '8001'])
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'm', '--port', '8001'],
            metrics_port=8001, metrics_path='/prometheus/metrics',
            metrics_host='localhost', listen_host=None, listen_port=None)

    def test_metrics_host_from_engine_args(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', 'm', '--host', '0.0.0.0', '--port', '8001'])
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'm', '--host', '0.0.0.0', '--port', '8001'],
            metrics_port=8001, metrics_path='/prometheus/metrics',
            metrics_host='0.0.0.0', listen_host=None, listen_port=None)

    def test_explicit_metrics_port_overrides_engine_args(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', 'm', '--port', '8001'], metrics_port=9999)
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'm', '--port', '8001'],
            metrics_port=9999, metrics_path='/prometheus/metrics',
            metrics_host='localhost', listen_host=None, listen_port=None)

    def test_grpc_mode_warns(self):
        launcher = TrtllmLauncher(['trtllm-serve', 'm', '--grpc'])
        with LaunchFixture(trtllm_mod) as fx:
            with self.assertLogs('graphsignal', level='WARNING') as cm:
                launcher.launch()
        self.assertTrue(any('gRPC mode' in msg for msg in cm.output))
        # argv still passes through unchanged.
        self.assertEqual(fx.launched_argv, ['/abs/exec', 'm', '--grpc'])


if __name__ == '__main__':
    unittest.main()
