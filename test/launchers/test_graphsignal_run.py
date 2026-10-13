import os
import sys
import unittest
from unittest.mock import patch

from graphsignal.commands import graphsignal_run
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler
from graphsignal.launchers.vllm_launcher import VllmLauncher
from graphsignal.launchers.sglang_launcher import SglangLauncher
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher
from graphsignal.launchers.fallback_launcher import FallbackLauncher


class GraphsignalRunDispatchTest(unittest.TestCase):
    """`graphsignal-run` walks the launcher list and dispatches to the first
    one whose match() returns True."""

    def _run_with_argv(self, argv):
        with patch.object(VllmLauncher, 'launch') as v, \
             patch.object(SglangLauncher, 'launch') as s, \
             patch.object(TrtllmLauncher, 'launch') as t, \
             patch.object(FallbackLauncher, 'launch') as f, \
             patch.object(sys, 'argv', argv):
            try:
                graphsignal_run.main()
            except SystemExit:
                pass
        return v, s, t, f

    def test_no_args_exits(self):
        with patch.object(sys, 'argv', ['graphsignal-run']):
            with self.assertRaises(SystemExit) as cm:
                graphsignal_run.main()
        self.assertEqual(cm.exception.code, 1)

    def test_version_flag_prints_and_exits_zero(self):
        # `--version` counts only as the FIRST argument: it is how an install
        # is verified. Later in argv it belongs to the workload.
        from graphsignal.version import __version__
        with patch.object(sys, 'argv', ['graphsignal-run', '--version']), \
             patch('builtins.print') as out:
            with self.assertRaises(SystemExit) as cm:
                graphsignal_run.main()
        self.assertEqual(cm.exception.code, 0)
        out.assert_called_once_with(f'graphsignal-run {__version__}')

    def test_workload_version_flag_is_forwarded(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'python', '--version'])
        f.assert_called_once()

    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ('-h', '--help'):
            with patch.object(sys, 'argv', ['graphsignal-run', flag]), \
                 patch('builtins.print') as out:
                with self.assertRaises(SystemExit) as cm:
                    graphsignal_run.main()
            self.assertEqual(cm.exception.code, 0, flag)
            out.assert_called_once_with(graphsignal_run.USAGE)

    def test_workload_help_flag_is_forwarded(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'python', 'app.py', '-h'])
        f.assert_called_once()
        v2, s2, t2, f2 = self._run_with_argv(['graphsignal-run', 'vllm', 'serve', '--help'])
        v2.assert_called_once()

    def test_vllm_wins_over_fallback(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_executable_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'sglang', 'serve', '--model', 'm'])
        s.assert_called_once()
        v.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_python_module_form_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'python', '-m', 'sglang.launch_server'])
        s.assert_called_once()
        f.assert_not_called()

    def test_trtllm_wins(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'trtllm-serve', '--model', 'm'])
        t.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        f.assert_not_called()

    def test_unrecognised_falls_back(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'python', 'my_app.py'])
        f.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        t.assert_not_called()

    def test_leading_flags_stripped_and_command_still_matches(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', '--metrics-port', '8000',
             '--listen-port', '18400', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()


class ExtractFlagsTest(unittest.TestCase):
    def test_no_flags(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['vllm', 'serve']),
            (None, None, None, None, None, ['vllm', 'serve']))

    def test_extracts_metrics_port_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port', '8000', 'trtllm-serve', 'm']),
            (8000, None, None, None, None, ['trtllm-serve', 'm']))

    def test_extracts_metrics_port_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port=8000', 'trtllm-serve', 'm']),
            (8000, None, None, None, None, ['trtllm-serve', 'm']))

    def test_extracts_listen_port_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port', '18400', 'vllm', 'serve']),
            (None, None, 18400, None, None, ['vllm', 'serve']))

    def test_extracts_listen_port_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port=18400', 'vllm', 'serve']),
            (None, None, 18400, None, None, ['vllm', 'serve']))

    def test_extracts_listen_host_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-host', '0.0.0.0', 'vllm', 'serve']),
            (None, '0.0.0.0', None, None, None, ['vllm', 'serve']))

    def test_extracts_listen_host_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-host=10.0.0.5', 'vllm', 'serve']),
            (None, '10.0.0.5', None, None, None, ['vllm', 'serve']))

    def test_listen_host_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--listen-host'])

    def test_extracts_both_leading_flags_in_any_order(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port', '18400', '--metrics-port', '9001',
                 'vllm', 'serve']),
            (9001, None, 18400, None, None, ['vllm', 'serve']))

    def test_unknown_leading_token_starts_workload_command(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['python', '--metrics-port', '8000']),
            (None, None, None, None, None, ['python', '--metrics-port', '8000']))

    def test_flags_after_command_left_for_workload(self):
        # Only leading flags are parsed; identically named workload flags
        # later in argv are forwarded untouched.
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--metrics-port', '8000']),
            (None, None, None, None, None, ['vllm', 'serve', '--metrics-port', '8000']))
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--listen-port', '18400']),
            (None, None, None, None, None, ['vllm', 'serve', '--listen-port', '18400']))

    def test_invalid_metrics_port_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port', 'abc', 'vllm', 'serve'])

    def test_metrics_port_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--metrics-port'])

    def test_invalid_listen_port_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port', 'abc', 'vllm', 'serve'])

    def test_listen_port_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--listen-port'])


class MainFlagForwardingTest(unittest.TestCase):
    def test_main_passes_flags_to_launchers(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, metrics_port=None, listen_host=None,
                         listen_port=None, cuda_graph_trace=None,
                         rocm_kernel_trace=None):
            created.append((list(args), metrics_port, listen_host, listen_port,
                            cuda_graph_trace, rocm_kernel_trace))
            return orig_init(self, args, metrics_port=metrics_port,
                             listen_host=listen_host, listen_port=listen_port,
                             cuda_graph_trace=cuda_graph_trace,
                             rocm_kernel_trace=rocm_kernel_trace)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(sys, 'argv',
                          ['graphsignal-run', '--metrics-port', '9001',
                           '--listen-host', '0.0.0.0',
                           '--listen-port', '18400', 'vllm', 'serve']):
            graphsignal_run.main()

        self.assertEqual(created, [(['vllm', 'serve'], 9001, '0.0.0.0', 18400, None, None)])

    def test_main_passes_none_flags_when_absent(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, metrics_port=None, listen_host=None,
                         listen_port=None, cuda_graph_trace=None,
                         rocm_kernel_trace=None):
            created.append((metrics_port, listen_host, listen_port,
                            cuda_graph_trace, rocm_kernel_trace))
            return orig_init(self, args, metrics_port=metrics_port,
                             listen_host=listen_host, listen_port=listen_port,
                             cuda_graph_trace=cuda_graph_trace,
                             rocm_kernel_trace=rocm_kernel_trace)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(sys, 'argv', ['graphsignal-run', 'vllm', 'serve']):
            graphsignal_run.main()

        self.assertEqual(created, [(None, None, None, None, None)])

    def test_main_passes_cuda_graph_trace_to_launchers(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, metrics_port=None, listen_host=None,
                         listen_port=None, cuda_graph_trace=None,
                         rocm_kernel_trace=None):
            created.append((list(args), cuda_graph_trace, rocm_kernel_trace))
            return orig_init(self, args, metrics_port=metrics_port,
                             listen_host=listen_host, listen_port=listen_port,
                             cuda_graph_trace=cuda_graph_trace,
                             rocm_kernel_trace=rocm_kernel_trace)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(sys, 'argv',
                          ['graphsignal-run', '--cuda-graph-trace', 'node',
                           'vllm', 'serve']):
            graphsignal_run.main()

        self.assertEqual(created, [(['vllm', 'serve'], 'node', None)])

    def test_main_passes_rocm_kernel_trace_to_launchers(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, metrics_port=None, listen_host=None,
                         listen_port=None, cuda_graph_trace=None,
                         rocm_kernel_trace=None):
            created.append((list(args), cuda_graph_trace, rocm_kernel_trace))
            return orig_init(self, args, metrics_port=metrics_port,
                             listen_host=listen_host, listen_port=listen_port,
                             cuda_graph_trace=cuda_graph_trace,
                             rocm_kernel_trace=rocm_kernel_trace)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(sys, 'argv',
                          ['graphsignal-run', '--rocm-kernel-trace', 'all',
                           'vllm', 'serve']):
            graphsignal_run.main()

        self.assertEqual(created, [(['vllm', 'serve'], None, 'all')])


class CudaGraphTraceFlagTest(unittest.TestCase):
    """`--cuda-graph-trace {graph|node}` selects the CUDA graph tracing
    granularity. Consumed by graphsignal-run, never forwarded, and handed to
    the native library through GRAPHSIGNAL_CUDA_GRAPH_TRACE."""

    def test_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace', 'node', 'vllm', 'serve']),
            (None, None, None, 'node', None, ['vllm', 'serve']))

    def test_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace=node', 'vllm', 'serve']),
            (None, None, None, 'node', None, ['vllm', 'serve']))

    def test_graph_value_accepted(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace', 'graph', 'vllm', 'serve']),
            (None, None, None, 'graph', None, ['vllm', 'serve']))

    def test_value_is_case_insensitive_and_trimmed(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace=  NODE ', 'vllm', 'serve']),
            (None, None, None, 'node', None, ['vllm', 'serve']))

    def test_combines_with_other_leading_flags(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port', '18400', '--cuda-graph-trace', 'node',
                 '--metrics-port', '9001', 'vllm', 'serve']),
            (9001, None, 18400, 'node', None, ['vllm', 'serve']))

    def test_invalid_value_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace', 'nodes', 'vllm', 'serve'])
        self.assertEqual(cm.exception.code, 1)

    def test_missing_value_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            graphsignal_run._extract_graphsignal_flags(['--cuda-graph-trace'])
        self.assertEqual(cm.exception.code, 1)

    def test_flag_after_command_left_for_workload(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--cuda-graph-trace', 'node']),
            (None, None, None, None, None,
             ['vllm', 'serve', '--cuda-graph-trace', 'node']))


class RocmKernelTraceFlagTest(unittest.TestCase):
    """`--rocm-kernel-trace {none|all}` switches per-kernel tracing on ROCm.
    Consumed by graphsignal-run, never forwarded, and handed to the native
    library through GRAPHSIGNAL_ROCM_KERNEL_TRACE."""

    def test_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--rocm-kernel-trace', 'all', 'vllm', 'serve']),
            (None, None, None, None, 'all', ['vllm', 'serve']))

    def test_equals_form_and_none(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--rocm-kernel-trace=none', 'vllm', 'serve']),
            (None, None, None, None, 'none', ['vllm', 'serve']))

    def test_value_is_case_insensitive_and_trimmed(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--rocm-kernel-trace=  ALL ', 'vllm', 'serve']),
            (None, None, None, None, 'all', ['vllm', 'serve']))

    def test_combines_with_other_leading_flags(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace', 'node', '--rocm-kernel-trace', 'all',
                 '--listen-port', '18400', 'vllm', 'serve']),
            (None, None, 18400, 'node', 'all', ['vllm', 'serve']))

    def test_invalid_value_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            graphsignal_run._extract_graphsignal_flags(
                ['--rocm-kernel-trace', 'on', 'vllm', 'serve'])
        self.assertEqual(cm.exception.code, 1)

    def test_missing_value_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            graphsignal_run._extract_graphsignal_flags(['--rocm-kernel-trace'])
        self.assertEqual(cm.exception.code, 1)

    def test_flag_after_command_left_for_workload(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--rocm-kernel-trace', 'all']),
            (None, None, None, None, None,
             ['vllm', 'serve', '--rocm-kernel-trace', 'all']))


class CudaGraphTraceEnvVarTest(unittest.TestCase):
    """CuptiProfiler.setup_env_vars publishes the mode to the child's
    environment; the flag beats an inherited value, and no flag leaves an
    inherited value alone."""

    ENV_VAR = 'GRAPHSIGNAL_CUDA_GRAPH_TRACE'

    def setUp(self):
        self._saved = os.environ.get(self.ENV_VAR)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(self.ENV_VAR, None)
        else:
            os.environ[self.ENV_VAR] = self._saved

    def test_sets_env_var(self):
        os.environ.pop(self.ENV_VAR, None)
        CuptiProfiler.setup_env_vars(cuda_graph_trace='node')
        self.assertEqual(os.environ[self.ENV_VAR], 'node')

    def test_flag_overrides_inherited_value(self):
        os.environ[self.ENV_VAR] = 'graph'
        CuptiProfiler.setup_env_vars(cuda_graph_trace='node')
        self.assertEqual(os.environ[self.ENV_VAR], 'node')

    def test_no_flag_leaves_inherited_value(self):
        os.environ[self.ENV_VAR] = 'node'
        CuptiProfiler.setup_env_vars()
        self.assertEqual(os.environ[self.ENV_VAR], 'node')

    def test_no_flag_and_no_env_var_sets_nothing(self):
        os.environ.pop(self.ENV_VAR, None)
        CuptiProfiler.setup_env_vars()
        self.assertNotIn(self.ENV_VAR, os.environ)

    def test_rocm_accepts_and_ignores_the_mode(self):
        # rocprofiler-sdk reports dispatches individually, so the mode is a
        # no-op there — but it must never raise or set the variable.
        os.environ.pop(self.ENV_VAR, None)
        RocmProfiler.setup_env_vars(cuda_graph_trace='node')
        self.assertNotIn(self.ENV_VAR, os.environ)


if __name__ == '__main__':
    unittest.main()
