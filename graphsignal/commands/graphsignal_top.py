"""`graphsignal-top`: a live terminal view of a workload run under graphsignal-run.

The view itself lives in graphsignal/top; this is its command-line entry point,
next to graphsignal-run and graphsignal-watch."""

import shutil
import sys
import time

from graphsignal.signals.routes import DEFAULT_LISTEN_HOST, DEFAULT_LISTEN_PORT

USAGE = """
Live terminal view of a workload profiled by graphsignal-run.

Usage:
  graphsignal-top [options]

Options:
  --host HOST       Host of the profiler's /signals endpoint
                    (default: 127.0.0.1; the host graphsignal-run was given
                    with --listen-host when watching a remote machine).
  --port PORT       Port of the /signals endpoint (default: 18259).
  --interval SEC    Seconds between polls (default: 1).
  --window SEC      Seconds over which rates and the kernel ranking are
                    computed (default: 5).
  --once            Print one frame to stdout and exit, instead of the
                    interactive view. Polls twice, --interval apart, so the
                    frame carries real rates.
  --no-color        Disable colors (NO_COLOR in the environment does too).
  -V, --version     Print the profiler version and exit.
  -h, --help        Show this help and exit.

Keys:
  k g m s     kernels / graphs / memcpy / sync tabs (tab cycles)
  up/down     select a row; enter shows its details (full symbol,
              totals since start, per-process split)
  p           pause          e  recent errors
  ?           help           q  quit

Example:
  graphsignal-run vllm serve <model> --port 8001     # in one terminal
  graphsignal-top                                   # in another
  graphsignal-top --host gpu-box --port 18259       # a remote workload
"""


class UsageError(Exception):
    pass


def parse_args(argv):
    """Hand-rolled like graphsignal-run's, so the two commands feel alike."""
    opts = {
        'host': DEFAULT_LISTEN_HOST,
        'port': DEFAULT_LISTEN_PORT,
        'interval': 1.0,
        'window': 5.0,
        'once': False,
        'color': True,
        'version': False,
        'help': False,
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ('-h', '--help'):
            opts['help'] = True
        elif arg in ('-V', '--version'):
            opts['version'] = True
        elif arg == '--once':
            opts['once'] = True
        elif arg == '--no-color':
            opts['color'] = False
        elif arg in ('--host', '--port', '--interval', '--window'):
            if i + 1 >= len(argv):
                raise UsageError(f'{arg} requires a value')
            value = argv[i + 1]
            i += 1
            if arg == '--host':
                if not value:
                    raise UsageError('--host requires a value')
                opts['host'] = value
            elif arg == '--port':
                opts['port'] = _parse_int(arg, value, 1, 65535)
            elif arg == '--interval':
                opts['interval'] = _parse_float(arg, value, 0.01)
            else:
                opts['window'] = _parse_float(arg, value, 0.2)
        elif arg.startswith('--host=') or arg.startswith('--port=') \
                or arg.startswith('--interval=') or arg.startswith('--window='):
            flag, value = arg.split('=', 1)
            argv = argv[:i] + [flag, value] + argv[i + 1:]
            continue
        else:
            raise UsageError(f'unknown argument: {arg}')
        i += 1
    return opts


def _parse_int(flag, value, lo, hi):
    try:
        number = int(value)
    except ValueError:
        raise UsageError(f'{flag} expects an integer, got {value!r}')
    if not lo <= number <= hi:
        raise UsageError(f'{flag} must be between {lo} and {hi}')
    return number


def _parse_float(flag, value, lo):
    try:
        number = float(value)
    except ValueError:
        raise UsageError(f'{flag} expects a number, got {value!r}')
    if number < lo:
        raise UsageError(f'{flag} must be at least {lo}')
    return number


def render_once(client, state, interval, out=None):
    """Two polls, one frame on stdout. Returns the exit code."""
    from graphsignal.top.client import SignalsUnavailable
    from graphsignal.top.render import UiState, frame_text, render_frame

    out = out or sys.stdout
    for attempt in range(2):
        try:
            state.update(client.fetch(), received_ts=time.time())
        except SignalsUnavailable as err:
            print(f'graphsignal-top: {client.url}: {err}', file=sys.stderr)
            return 1
        if attempt == 0:
            time.sleep(interval)
    size = shutil.get_terminal_size(fallback=(120, 40))
    ui = UiState(once=True)
    lines = render_frame(state, ui, size.columns, size.lines)
    out.write(frame_text(lines) + '\n')
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        opts = parse_args(argv)
    except UsageError as err:
        print(f'graphsignal-top: {err}', file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(2)

    if opts['help']:
        print(USAGE)
        sys.exit(0)
    if opts['version']:
        from graphsignal.version import __version__
        print(f'graphsignal-top {__version__}')
        sys.exit(0)

    from graphsignal.top.client import SignalsClient
    from graphsignal.top.state import TopState

    client = SignalsClient(opts['host'], opts['port'])
    state = TopState(window=opts['window'], endpoint=client.endpoint)

    if opts['once']:
        sys.exit(render_once(client, state, opts['interval']))

    if not sys.stdout.isatty():
        print('graphsignal-top: stdout is not a terminal; use --once for a printable frame',
              file=sys.stderr)
        sys.exit(2)
    try:
        import curses  # noqa: F401
    except ImportError:
        print('graphsignal-top: the interactive view needs the curses module; use --once',
              file=sys.stderr)
        sys.exit(2)

    from graphsignal.top import theme, ui
    sys.exit(ui.run(client, state, interval=opts['interval'],
                    use_color=theme.color_enabled(opts['color'])))


if __name__ == '__main__':
    main()
