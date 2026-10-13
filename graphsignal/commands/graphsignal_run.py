import logging
import os
import sys

from graphsignal.launchers.vllm_launcher import VllmLauncher
from graphsignal.launchers.sglang_launcher import SglangLauncher
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher
from graphsignal.launchers.fallback_launcher import FallbackLauncher

log = logging.getLogger(__name__)


def _setup_logging():
    """Emit the `graphsignal` logger to stderr so launcher diagnostics (which
    launcher matched, the watcher command) are visible in the workload's log
    instead of being silently dropped. The launcher runs in the workload
    process before spawning, so without this its debug output goes nowhere.
    Verbosity follows GRAPHSIGNAL_DEBUG."""
    debug = os.getenv('GRAPHSIGNAL_DEBUG', '').strip().lower() in ('1', 'true', 'yes')
    gs_logger = logging.getLogger('graphsignal')
    gs_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    if not any(isinstance(h, logging.StreamHandler) for h in gs_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s graphsignal-run: %(message)s'))
        gs_logger.addHandler(handler)

USAGE = """
Run a target application with the Graphsignal profiler.

Options (must precede the command):
  -h, --help      Show this help and exit.
  --version       Print the profiler version and exit.
  --metrics-port PORT
                  Port to scrape the engine's Prometheus /metrics endpoint on.
                  Overrides the port derived from the engine's --port/default.
  --listen-host HOST
                  Host to bind the watcher's /signals HTTP endpoint to
                  (default: 127.0.0.1; any other value exposes the endpoint
                  to that network).
  --listen-port PORT
                  Port for the watcher's local /signals HTTP endpoint
                  (default: 18259).
  --cuda-graph-trace {graph|node}
                  Granularity for CUDA graph launches (default: graph).
                  `graph` records one timing per graph replay in
                  cuda_graphs_nanoseconds. `node` records the kernels inside
                  the graph individually in cuda_kernels_nanoseconds, at
                  higher CUPTI cost. Also settable via
                  GRAPHSIGNAL_CUDA_GRAPH_TRACE; the flag wins.
  --rocm-kernel-trace {none|all}
                  Per-kernel tracing on ROCm (default: none). `all` times
                  every kernel dispatch into rocm_kernels_nanoseconds;
                  rocprofiler-sdk charges tens of microseconds per dispatch
                  for it, so use it for an investigation, not in production.
                  Also settable via GRAPHSIGNAL_ROCM_KERNEL_TRACE; the flag
                  wins.

Example:
  graphsignal-run vllm serve facebook/opt-125m --port 8001
  graphsignal-run sglang serve --model-path <model>
  graphsignal-run --metrics-port 8000 trtllm-serve <model> --port 8000
  graphsignal-run python myapp.py
  graphsignal-run app.py
"""


CUDA_GRAPH_TRACE_MODES = ('graph', 'node')
ROCM_KERNEL_TRACE_MODES = ('none', 'all')


def _extract_graphsignal_flags(argv):
    """Pull graphsignal-run's own flags out of argv before the workload command.

    `--metrics-port` specifies the Prometheus scrape port.
    `--listen-host`/`--listen-port` specify the /signals endpoint bind address.
    `--cuda-graph-trace` selects the CUDA graph tracing granularity.
    `--rocm-kernel-trace` switches per-kernel tracing on ROCm on or off.
    All are consumed here and never forwarded to the workload. Only leading
    flags (before the workload command) are parsed, so identically named
    workload flags later in argv are left untouched.
    """
    metrics_port = None
    listen_host = None
    listen_port = None
    cuda_graph_trace = None
    rocm_kernel_trace = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--metrics-port':
            if i + 1 >= len(argv):
                print("graphsignal-run: --metrics-port requires a value\n")
                sys.exit(1)
            metrics_port = _parse_port('--metrics-port', argv[i + 1])
            i += 2
            continue
        if arg.startswith('--metrics-port='):
            metrics_port = _parse_port('--metrics-port', arg.split('=', 1)[1])
            i += 1
            continue
        if arg == '--listen-host':
            if i + 1 >= len(argv):
                print("graphsignal-run: --listen-host requires a value\n")
                sys.exit(1)
            listen_host = argv[i + 1]
            i += 2
            continue
        if arg.startswith('--listen-host='):
            listen_host = arg.split('=', 1)[1]
            i += 1
            continue
        if arg == '--listen-port':
            if i + 1 >= len(argv):
                print("graphsignal-run: --listen-port requires a value\n")
                sys.exit(1)
            listen_port = _parse_port('--listen-port', argv[i + 1])
            i += 2
            continue
        if arg.startswith('--listen-port='):
            listen_port = _parse_port('--listen-port', arg.split('=', 1)[1])
            i += 1
            continue
        if arg == '--cuda-graph-trace':
            if i + 1 >= len(argv):
                print("graphsignal-run: --cuda-graph-trace requires a value\n")
                sys.exit(1)
            cuda_graph_trace = _parse_cuda_graph_trace(argv[i + 1])
            i += 2
            continue
        if arg.startswith('--cuda-graph-trace='):
            cuda_graph_trace = _parse_cuda_graph_trace(arg.split('=', 1)[1])
            i += 1
            continue
        if arg == '--rocm-kernel-trace':
            if i + 1 >= len(argv):
                print("graphsignal-run: --rocm-kernel-trace requires a value\n")
                sys.exit(1)
            rocm_kernel_trace = _parse_rocm_kernel_trace(argv[i + 1])
            i += 2
            continue
        if arg.startswith('--rocm-kernel-trace='):
            rocm_kernel_trace = _parse_rocm_kernel_trace(arg.split('=', 1)[1])
            i += 1
            continue
        break
    return (metrics_port, listen_host, listen_port, cuda_graph_trace,
            rocm_kernel_trace, argv[i:])


def _parse_port(flag, value):
    try:
        return int(value)
    except (TypeError, ValueError):
        print("graphsignal-run: invalid %s value: %s\n" % (flag, value))
        sys.exit(1)


def _parse_cuda_graph_trace(value):
    mode = (value or '').strip().lower()
    if mode not in CUDA_GRAPH_TRACE_MODES:
        print("graphsignal-run: invalid --cuda-graph-trace value: %s (expected %s)\n"
              % (value, '|'.join(CUDA_GRAPH_TRACE_MODES)))
        sys.exit(1)
    return mode


def _parse_rocm_kernel_trace(value):
    mode = (value or '').strip().lower()
    if mode not in ROCM_KERNEL_TRACE_MODES:
        print("graphsignal-run: invalid --rocm-kernel-trace value: %s (expected %s)\n"
              % (value, '|'.join(ROCM_KERNEL_TRACE_MODES)))
        sys.exit(1)
    return mode


def main():
    # Answered before the no-command check: `--version` IS the whole command.
    # This is how an install is verified — the skill and docs point here — so
    # it must work with nothing else on the line and touch nothing else.
    if '--version' in sys.argv[1:2] or '-V' in sys.argv[1:2]:
        from graphsignal.version import __version__
        print(f'graphsignal-run {__version__}')
        sys.exit(0)
    # Same rule for help: only as the first argument. Later in argv, `-h`
    # belongs to the workload and passes through untouched.
    if '--help' in sys.argv[1:2] or '-h' in sys.argv[1:2]:
        print(USAGE)
        sys.exit(0)

    if len(sys.argv) < 2:
        print("graphsignal-run: no command specified\n")
        print(USAGE)
        sys.exit(1)

    _setup_logging()

    (metrics_port, listen_host, listen_port, cuda_graph_trace,
     rocm_kernel_trace, target_args) = _extract_graphsignal_flags(sys.argv[1:])
    log.debug('graphsignal-run target args: %s (metrics_port=%s, listen_host=%s, '
              'listen_port=%s, cuda_graph_trace=%s, rocm_kernel_trace=%s)',
              target_args, metrics_port, listen_host, listen_port, cuda_graph_trace,
              rocm_kernel_trace)

    launcher_kwargs = dict(
        metrics_port=metrics_port, listen_host=listen_host, listen_port=listen_port,
        cuda_graph_trace=cuda_graph_trace, rocm_kernel_trace=rocm_kernel_trace)
    launchers = [
        VllmLauncher(target_args, **launcher_kwargs),
        SglangLauncher(target_args, **launcher_kwargs),
        TrtllmLauncher(target_args, **launcher_kwargs),
        FallbackLauncher(target_args, **launcher_kwargs),
    ]

    for launcher in launchers:
        try:
            matched = launcher.match()
        except Exception as exc:
            log.error('Error matching launcher %s: %s', type(launcher).__name__, exc, exc_info=True)
            continue
        if matched:
            log.debug('Selected launcher: %s', type(launcher).__name__)
            launcher.launch()
            return

    print("graphsignal-run: no launcher matched")
    sys.exit(1)


if __name__ == '__main__':
    main()
