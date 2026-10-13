import logging

from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    resolve_executable as _resolve, resolve_metrics_port)
from graphsignal.launchers.supervisor import launch_supervised
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')

# vLLM serves Prometheus /metrics on its HTTP server (--port, default 8000).
DEFAULT_SERVE_PORT = 8000


class VllmLauncher(BaseLauncher):
    def match(self) -> bool:
        return self.executable_name() == 'vllm'

    def launch(self) -> None:
        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace,
                                    rocm_kernel_trace=self.rocm_kernel_trace)

        new_args = _inject_vllm_args(self.args)

        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=DEFAULT_SERVE_PORT)

        executable = _resolve(new_args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {new_args[0]}')

        logger.debug('VllmLauncher launch: %s %s', executable, new_args)
        launch_supervised([executable] + new_args[1:],
                          metrics_port=metrics_port,
                          listen_host=self.listen_host,
                          listen_port=self.listen_port)


def _inject_vllm_args(args):
    args = list(args)

    # vLLM exposes Prometheus on its HTTP server by default; ensure log stats stay on.
    if _has_flag(args, '--disable-log-stats'):
        args = [a for a in args if a != '--disable-log-stats']

    return args


def _has_flag(args, flag) -> bool:
    for a in args:
        if a == flag or a.startswith(flag + '='):
            return True
    return False
