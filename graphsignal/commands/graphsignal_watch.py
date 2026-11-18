import argparse
import logging
import signal as signal_module
import sys

import graphsignal.watcher as gwatcher

log = logging.getLogger(__name__)

USAGE = """
Watch a target process (and its descendants) and profile it.

Usage:
  graphsignal-watch --pid PID
                    [--metrics-port PORT] [--metrics-path PATH]
                    [--metrics-host HOST]
                    [--health-port PORT] [--health-path PATH]
                    [--health-host HOST]
                    [--engine NAME]
                    [--dcgm-exporter URL]
                    [--listen-host HOST] [--listen-port PORT]
                    [--listen-fd FD]
"""


def main():
    parser = argparse.ArgumentParser(
        prog='graphsignal-watch',
        description='Watch a target process with the Graphsignal profiler',
        usage=USAGE.strip(),
    )
    parser.add_argument('--pid', type=int, required=True,
                        help='Target process PID to watch')
    parser.add_argument('--metrics-port', type=int, default=None,
                        help='Port to scrape the Prometheus metrics endpoint on')
    parser.add_argument('--metrics-path', type=str, default=None,
                        help='HTTP path for the Prometheus metrics endpoint '
                             '(default: /metrics)')
    parser.add_argument('--metrics-host', type=str, default=None,
                        help='HTTP host for the Prometheus metrics endpoint '
                             '(default: 127.0.0.1)')
    parser.add_argument('--health-port', type=int, default=None,
                        help="Engine serving port to probe for health")
    parser.add_argument('--health-path', type=str, default=None,
                        help='HTTP path of the health probe (default: /health)')
    parser.add_argument('--health-host', type=str, default=None,
                        help='HTTP host of the health probe (default: 127.0.0.1)')
    parser.add_argument('--engine', type=str, default=None,
                        help='Inference engine (vllm, sglang): its version and '
                             'configuration are read from the health probe\'s '
                             'host and port and recorded as the engine resource')
    parser.add_argument('--dcgm-exporter', type=str, default=None,
                        help='dcgm-exporter metrics URL to import DCGM '
                             'profiling metrics from')
    parser.add_argument('--listen-host', type=str, default=None,
                        help='Host to bind the /signals HTTP endpoint to '
                             '(default: 127.0.0.1; any other value exposes '
                             'the endpoint to that network)')
    parser.add_argument('--listen-port', type=int, default=None,
                        help='Port for the local /signals HTTP endpoint '
                             '(default: 18259)')
    parser.add_argument('--listen-fd', type=int, default=None,
                        help='File descriptor of an already bound socket to '
                             'serve the /signals HTTP endpoint on, instead of '
                             'binding --listen-host/--listen-port')
    args = parser.parse_args()

    try:
        gwatcher.configure(
            target_pid=args.pid,
            metrics_port=args.metrics_port,
            metrics_path=args.metrics_path,
            metrics_host=args.metrics_host,
            health_port=args.health_port,
            health_path=args.health_path,
            health_host=args.health_host,
            engine=args.engine,
            dcgm_exporter_url=args.dcgm_exporter,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            listen_fd=args.listen_fd,
        )
    except Exception as exc:
        log.error('graphsignal-watch: profiler failed to configure: %s', exc, exc_info=True)
        sys.exit(1)

    watcher = gwatcher.watcher()
    terminated = watcher.target_terminated_event()

    def _signal_handler(signum, frame):
        log.debug('graphsignal-watch: received signal %s', signum)
        terminated.set()

    for sig in (signal_module.SIGINT, signal_module.SIGTERM):
        try:
            signal_module.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    # Block until the target terminates (or we receive a signal).
    terminated.wait()

    gwatcher.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
