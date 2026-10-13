import unittest

from graphsignal.launchers import vllm_launcher as vllm_mod
from graphsignal.launchers.vllm_launcher import VllmLauncher, _inject_vllm_args

from test.launchers._helpers import LaunchFixture


class VllmMatchTest(unittest.TestCase):
    def test_matches_vllm_command(self):
        self.assertTrue(VllmLauncher(['vllm', 'serve', 'model']).match())
        self.assertTrue(VllmLauncher(['/abs/path/vllm', 'serve']).match())

    def test_does_not_match_other_commands(self):
        self.assertFalse(VllmLauncher(['python', 'app.py']).match())
        self.assertFalse(VllmLauncher(['vllm-something', 'serve']).match())
        self.assertFalse(VllmLauncher([]).match())


class VllmArgInjectionTest(unittest.TestCase):
    def test_strips_disable_log_stats_only(self):
        out = _inject_vllm_args(
            ['vllm', 'serve', 'm', '--disable-log-stats', '--dtype', 'auto'])
        self.assertEqual(out, ['vllm', 'serve', 'm', '--dtype', 'auto'])

    def test_argv_without_flag_passes_byte_for_byte(self):
        args = ['vllm', 'serve', 'm', '--port', '8001', '--dtype', 'auto']
        self.assertEqual(_inject_vllm_args(args), args)


class VllmLaunchTest(unittest.TestCase):
    def test_launch_sets_up_env_and_supervises(self):
        launcher = VllmLauncher(['vllm', 'serve', 'm'], listen_port=18300)
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()

        fx.cupti_env_m.assert_called_once_with(cuda_graph_trace=None)
        fx.rocm_env_m.assert_called_once_with(cuda_graph_trace=None, rocm_kernel_trace=None)
        # No --port in argv → falls back to vLLM's default serving port (8000).
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm'],
            metrics_port=8000, listen_host=None, listen_port=18300)

    def test_launch_passes_argv_unchanged_except_disable_log_stats(self):
        launcher = VllmLauncher(
            ['vllm', 'serve', 'm', '--disable-log-stats', '--dtype', 'auto'])
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()

        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm', '--dtype', 'auto'],
            metrics_port=8000, listen_host=None, listen_port=None)
        self.assertNotIn('--disable-log-stats', fx.launched_argv)
        self.assertNotIn('--otlp-traces-endpoint', fx.launched_argv)

    def test_launch_metrics_port_from_engine_args(self):
        launcher = VllmLauncher(['vllm', 'serve', 'm', '--port', '8001'])
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm', '--port', '8001'],
            metrics_port=8001, listen_host=None, listen_port=None)

    def test_launch_explicit_metrics_port_overrides_engine_args(self):
        launcher = VllmLauncher(
            ['vllm', 'serve', 'm', '--port', '8001'], metrics_port=9999)
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm', '--port', '8001'],
            metrics_port=9999, listen_host=None, listen_port=None)

    def test_launch_raises_when_executable_missing(self):
        launcher = VllmLauncher(['vllm-typo'])
        with LaunchFixture(vllm_mod) as fx:
            fx.resolve_m.return_value = None
            with self.assertRaises(FileNotFoundError):
                launcher.launch()
        fx.launch_supervised_m.assert_not_called()


if __name__ == '__main__':
    unittest.main()
