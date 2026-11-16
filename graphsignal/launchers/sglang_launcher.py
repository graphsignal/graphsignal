import logging
import os

from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    extract_port, resolve_executable as _resolve, resolve_metrics_port,
    resolve_probe_host)
from graphsignal.launchers.supervisor import launch_supervised
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')

_SGLANG_NAMES = {'sglang', 'sglang.launch_server'}

# SGLang serves Prometheus /metrics on its HTTP server (--port, default 30000).
DEFAULT_SERVE_PORT = 30000
# Not /health: in SGLang /health (= /health_generate) submits a 1-token
# generation per call by default. /ready reads flags only, and turns 200 once
# SGLang's own startup warmup succeeds (503 while paused, e.g. a weight sync).
# Releases without /ready (0.5.10 has none) answer 404; the probe then falls
# back to /ping, which only says the HTTP server is up.
HEALTH_PATH = '/ready,/ping'


class SglangLauncher(BaseLauncher):
    def match(self) -> bool:
        name = self.executable_name()
        if name in _SGLANG_NAMES:
            return True
        # `python -m sglang.launch_server …`
        if len(self.args) >= 3 and os.path.basename(self.args[0]).startswith('python') \
                and self.args[1] == '-m' and self.args[2] in _SGLANG_NAMES:
            return True
        return False

    def launch(self) -> None:
        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace,
                                    rocm_kernel_trace=self.rocm_kernel_trace)

        new_args = _inject_sglang_args(self.args)

        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=DEFAULT_SERVE_PORT)

        executable = _resolve(new_args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {new_args[0]}')

        logger.debug('SglangLauncher launch: %s %s', executable, new_args)
        launch_supervised([executable] + new_args[1:],
                          metrics_port=metrics_port,
                          # The serving port, never --metrics-port.
                          health_port=extract_port(self.args, default=DEFAULT_SERVE_PORT),
                          health_path=HEALTH_PATH,
                          health_host=resolve_probe_host(self.args),
                          dcgm_exporter_url=self.dcgm_exporter_url,
                          listen_host=self.listen_host,
                          listen_port=self.listen_port)


def _inject_sglang_args(args):
    args = list(args)

    # SGLang's Prometheus endpoint is off by default; the profiler scrapes it.
    if not _has_flag(args, '--enable-metrics'):
        args.append('--enable-metrics')

    return args


def _has_flag(args, flag) -> bool:
    for a in args:
        if a == flag or a.startswith(flag + '='):
            return True
    return False
