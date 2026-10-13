import unittest

from graphsignal.launchers import sglang_launcher as sglang_mod
from graphsignal.launchers.sglang_launcher import SglangLauncher, _inject_sglang_args

from test.launchers._helpers import LaunchFixture


class SglangMatchTest(unittest.TestCase):
    def test_matches_sglang_executable(self):
        self.assertTrue(SglangLauncher(['sglang', 'serve']).match())
        self.assertTrue(SglangLauncher(['sglang.launch_server', '--model', 'foo']).match())
        self.assertTrue(SglangLauncher(['/usr/bin/sglang']).match())

    def test_matches_python_m_form(self):
        self.assertTrue(
            SglangLauncher(['python', '-m', 'sglang.launch_server', '--model', 'foo']).match())
        self.assertTrue(
            SglangLauncher(['python3.11', '-m', 'sglang.launch_server']).match())
        self.assertTrue(
            SglangLauncher(['/usr/bin/python', '-m', 'sglang.launch_server']).match())

    def test_does_not_match(self):
        self.assertFalse(SglangLauncher(['python', 'app.py']).match())
        self.assertFalse(SglangLauncher(['python', '-m', 'something_else']).match())
        self.assertFalse(SglangLauncher(['sglang-other', 'serve']).match())
        self.assertFalse(SglangLauncher([]).match())


class SglangArgInjectionTest(unittest.TestCase):
    def test_appends_enable_metrics_when_absent(self):
        out = _inject_sglang_args(['sglang', 'serve', '--model-path', 'm'])
        self.assertEqual(out,
                         ['sglang', 'serve', '--model-path', 'm', '--enable-metrics'])

    def test_appends_nothing_else(self):
        out = _inject_sglang_args(['sglang', 'serve'])
        self.assertEqual(out, ['sglang', 'serve', '--enable-metrics'])
        self.assertNotIn('--enable-trace', out)
        self.assertNotIn('--otlp-traces-endpoint', out)

    def test_preserves_existing_enable_metrics(self):
        args = ['sglang', 'serve', '--enable-metrics']
        out = _inject_sglang_args(args)
        self.assertEqual(out, args)
        self.assertEqual(out.count('--enable-metrics'), 1)


class SglangLaunchTest(unittest.TestCase):
    def test_launch_appends_metrics_flag_and_supervises(self):
        launcher = SglangLauncher(['sglang', 'serve'], listen_port=18301)
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()

        fx.cupti_env_m.assert_called_once_with(cuda_graph_trace=None)
        fx.rocm_env_m.assert_called_once_with(cuda_graph_trace=None, rocm_kernel_trace=None)
        # No --port in argv → falls back to SGLang's default serving port (30000).
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', '--enable-metrics'],
            metrics_port=30000, listen_host=None, listen_port=18301)
        self.assertNotIn('--enable-trace', fx.launched_argv)
        self.assertNotIn('--otlp-traces-endpoint', fx.launched_argv)

    def test_launch_metrics_port_from_engine_args(self):
        launcher = SglangLauncher(['sglang', 'serve', '--port', '30001'])
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', '--port', '30001', '--enable-metrics'],
            metrics_port=30001, listen_host=None, listen_port=None)

    def test_launch_explicit_metrics_port_overrides_engine_args(self):
        launcher = SglangLauncher(
            ['sglang', 'serve', '--port', '30001'], metrics_port=9999)
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', '--port', '30001', '--enable-metrics'],
            metrics_port=9999, listen_host=None, listen_port=None)

    def test_launch_raises_when_executable_missing(self):
        launcher = SglangLauncher(['sglang'])
        with LaunchFixture(sglang_mod) as fx:
            fx.resolve_m.return_value = None
            with self.assertRaises(FileNotFoundError):
                launcher.launch()
        fx.launch_supervised_m.assert_not_called()


if __name__ == '__main__':
    unittest.main()
