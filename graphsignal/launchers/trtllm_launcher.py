import logging

from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    resolve_executable as _resolve, resolve_metrics_host, resolve_metrics_port)
from graphsignal.launchers.supervisor import launch_supervised
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')

_TRTLLM_NAMES = {'trtllm', 'trtllm-serve', 'trtllm-llmapi-launch'}

# trtllm-serve exposes Prometheus metrics on its HTTP server (--port, default 8000).
DEFAULT_SERVE_PORT = 8000
DEFAULT_SERVE_HOST = 'localhost'
# Separate from `/metrics`, which returns JSON iteration stats (IterationStats).
DEFAULT_METRICS_PATH = '/prometheus/metrics'


class TrtllmLauncher(BaseLauncher):
    def match(self) -> bool:
        return self.executable_name() in _TRTLLM_NAMES

    def launch(self) -> None:
        if _has_flag(self.args, '--grpc'):
            logger.warning(
                'TRT-LLM gRPC mode has no HTTP /prometheus/metrics endpoint; '
                'engine Prometheus metrics will not be scraped')

        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace,
                                    rocm_kernel_trace=self.rocm_kernel_trace)

        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=DEFAULT_SERVE_PORT)
        metrics_host = resolve_metrics_host(
            None, self.args, default=DEFAULT_SERVE_HOST)

        executable = _resolve(self.args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {self.args[0]}')

        logger.debug('TrtllmLauncher launch: %s %s', executable, self.args)
        launch_supervised([executable] + list(self.args[1:]),
                          metrics_port=metrics_port,
                          metrics_path=DEFAULT_METRICS_PATH,
                          metrics_host=metrics_host,
                          listen_host=self.listen_host,
                          listen_port=self.listen_port)


def _has_flag(args, flag) -> bool:
    for a in args:
        if a == flag or a.startswith(flag + '='):
            return True
    return False
