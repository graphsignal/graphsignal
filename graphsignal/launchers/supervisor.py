import ctypes
import json
import logging
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import List, NoReturn, Optional

from graphsignal.launchers.command_utils import bind_listen_socket, start_watcher
from graphsignal.signals.routes import DEFAULT_LISTEN_HOST, DEFAULT_LISTEN_PORT

logger = logging.getLogger('graphsignal')

# Base dir for console capture; capture is disabled when it doesn't exist
# (e.g. macOS). Tests patch this to a temp dir.
_SHM_BASE = '/dev/shm'
_LOG_DIR_PREFIX = 'graphsignal_log_'

SEGMENT_SIZE_LIMIT = 1024 * 1024
MAX_SEGMENTS = 2

# Catchable signals proxied to the target's process group. The target runs in
# its own session, so terminal/orchestrator signals only ever reach the
# supervisor — without forwarding, graceful shutdown (drain on SIGTERM) breaks.
_FORWARD_SIGNALS = tuple(
    getattr(signal, name) for name in
    ('SIGINT', 'SIGTERM', 'SIGHUP', 'SIGQUIT', 'SIGUSR1', 'SIGUSR2', 'SIGWINCH')
    if hasattr(signal, name))


def log_dir_for_pid(pid: int) -> str:
    return os.path.join(_SHM_BASE, f'{_LOG_DIR_PREFIX}{pid}')


def _create_log_dir(log_dir: str) -> None:
    try:
        os.mkdir(log_dir, 0o700)
    except FileExistsError:
        pass
    st = os.lstat(log_dir)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
        raise OSError(f'{log_dir} is not a directory owned by the current user')
    if st.st_mode & 0o077:
        os.chmod(log_dir, 0o700)


def _capture_enabled() -> bool:
    return os.path.isdir(_SHM_BASE)


def _set_pdeathsig():
    """Runs in the forked child before exec (Linux only). PR_SET_PDEATHSIG
    makes the target receive SIGTERM when the supervisor dies, so an
    uncatchable `kill -9` of the supervisor can't orphan the target — the
    closest equivalent to execv's single-process semantics."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # 1 = PR_SET_PDEATHSIG
    except Exception:
        pass


class _SegmentWriter:
    """Bounded JSONL segment writer for captured console lines.

    Segments are `log_<seq>.jsonl` in `log_dir`; rotation opens the next seq
    and removes the oldest so at most `max_segments` exist. A failed write
    (e.g. tmpfs full) drops the line and is later recorded in-stream as a
    `{"dropped": N}` marker — the writer never blocks or raises into the
    drain thread.
    """

    def __init__(self, log_dir: str,
                 segment_size_limit: int = SEGMENT_SIZE_LIMIT,
                 max_segments: int = MAX_SEGMENTS):
        self._log_dir = log_dir
        self._segment_size_limit = segment_size_limit
        self._max_segments = max(1, max_segments)
        self._lock = threading.Lock()
        self._seq = 0
        self._file = None
        self._size = 0
        self._dropped = 0
        self._closed = False

    def write(self, stream: str, line: str, ts_ns: Optional[int] = None) -> None:
        self.write_record(
            {'ts': ts_ns if ts_ns else time.time_ns(), 'stream': stream, 'line': line})

    def write_record(self, record: dict) -> None:
        data = (json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8', errors='replace')
        with self._lock:
            if self._closed:
                return
            try:
                if self._file is None or self._size + len(data) > self._segment_size_limit:
                    self._rotate_locked()
                if self._dropped:
                    marker = (json.dumps({'ts': time.time_ns(), 'dropped': self._dropped})
                              + '\n').encode('utf-8')
                    self._file.write(marker)
                    self._size += len(marker)
                    self._dropped = 0
                self._file.write(data)
                self._file.flush()
                self._size += len(data)
            except OSError:
                self._dropped += 1

    def _rotate_locked(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
        cur = self._seq
        fd = os.open(os.path.join(self._log_dir, f'log_{cur}.jsonl'),
                     os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self._file = os.fdopen(fd, 'ab')
        self._size = 0
        self._seq = cur + 1
        stale = cur - self._max_segments
        if stale >= 0:
            try:
                os.remove(os.path.join(self._log_dir, f'log_{stale}.jsonl'))
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._file is None:
                return
            try:
                if self._dropped:
                    marker = (json.dumps({'ts': time.time_ns(), 'dropped': self._dropped})
                              + '\n').encode('utf-8')
                    self._file.write(marker)
                    self._dropped = 0
                self._file.close()
            except OSError:
                pass
            self._file = None


def _drain_stream(pipe, console_fd: int, stream_name: str,
                  writer: Optional[_SegmentWriter]) -> None:
    """Tee one target pipe: raw bytes to the original console fd, decoded
    lines to the segment writer. Must never stop reading while the pipe is
    open — a stalled reader would fill the pipe and block the target's
    write()s."""
    try:
        for raw in iter(pipe.readline, b''):
            try:
                os.write(console_fd, raw)
            except OSError:
                pass
            if writer is not None:
                writer.write(stream_name, raw.decode('utf-8', errors='replace').rstrip('\n'))
    except Exception:
        logger.error('Error draining target %s', stream_name, exc_info=True)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _open_listen_socket(listen_host: Optional[str], listen_port: Optional[int]):
    """Bind the address the watcher will serve /signals on, or exit: the
    profiler does not start a workload it cannot be read from."""
    host = listen_host or os.getenv('GRAPHSIGNAL_LISTEN_HOST') or DEFAULT_LISTEN_HOST
    port = listen_port
    try:
        if port is None:
            port = int(os.getenv('GRAPHSIGNAL_LISTEN_PORT') or DEFAULT_LISTEN_PORT)
        return bind_listen_socket(host, port)
    except (OSError, ValueError, OverflowError) as exc:
        print('graphsignal-run: cannot listen on %s:%s: %s' % (host, port, exc),
              file=sys.stderr)
        sys.exit(1)


def _exit_like(rc: int) -> NoReturn:
    """Terminate the supervisor exactly like the target: same exit code, or
    death by the same signal (so the caller sees WIFSIGNALED, not a plain
    code)."""
    if rc < 0:
        signum = -rc
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
        try:
            os.kill(os.getpid(), signum)
        except OSError:
            pass
        # Reached only if the signal didn't terminate us (blocked/ignored);
        # fall back to the shell's signal-death code convention.
        sys.exit(128 + signum)
    sys.exit(rc)


def launch_supervised(argv: List[str], *,
                      metrics_port: Optional[int] = None,
                      metrics_path: Optional[str] = None,
                      metrics_host: Optional[str] = None,
                      listen_host: Optional[str] = None,
                      listen_port: Optional[int] = None) -> NoReturn:
    """Drop-in replacement for the `start_watcher(os.getpid()) + os.execv`
    launcher tail. Never returns. Kwargs mirror `start_watcher` (minus pid)
    and are passed through with the target child's pid."""
    if not argv:
        print('graphsignal-run: no command specified')
        sys.exit(1)

    listen_socket = _open_listen_socket(listen_host, listen_port)

    # Handlers are installed before spawn (closing the spawn/signal race);
    # the holder is filled in right after Popen returns.
    holder = {}

    def _forward(signum, frame):
        target = holder.get('target')
        if target is None:
            return
        try:
            os.killpg(target.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass

    for sig in _FORWARD_SIGNALS:
        try:
            signal.signal(sig, _forward)
        except (ValueError, OSError):
            pass

    capture = _capture_enabled()

    env = dict(os.environ)
    popen_kwargs = {'env': env, 'start_new_session': True}
    if sys.platform.startswith('linux'):
        popen_kwargs['preexec_fn'] = _set_pdeathsig
    if capture:
        # Piped stdio flips libc to block buffering; keep Python targets
        # line-buffered so the console tee stays live.
        env['PYTHONUNBUFFERED'] = '1'
        popen_kwargs['stdout'] = subprocess.PIPE
        popen_kwargs['stderr'] = subprocess.PIPE

    # Spawned before any threads exist in this process, keeping preexec_fn
    # fork-safe.
    try:
        target = subprocess.Popen(argv, **popen_kwargs)
    except PermissionError:
        print("graphsignal-run: permission error while launching '%s'" % argv[0])
        print("Did you mean `graphsignal-run python %s`?" % argv[0])
        listen_socket.close()
        sys.exit(1)
    except FileNotFoundError:
        print("graphsignal-run: executable not found: %s" % argv[0])
        listen_socket.close()
        sys.exit(1)
    holder['target'] = target

    writer = None
    drain_threads = []
    if capture:
        log_dir = log_dir_for_pid(target.pid)
        try:
            _create_log_dir(log_dir)
            writer = _SegmentWriter(log_dir)
        except OSError:
            logger.error('Failed to create console log dir %s', log_dir, exc_info=True)
        # Drain even without a writer — the pipes exist and must be emptied.
        for pipe, console_fd, name in (
                (target.stdout, 1, 'stdout'), (target.stderr, 2, 'stderr')):
            thread = threading.Thread(
                target=_drain_stream, args=(pipe, console_fd, name, writer), daemon=True)
            thread.start()
            drain_threads.append(thread)

    logger.debug('Supervising target pid=%s: %s', target.pid, argv)
    start_watcher(target.pid,
                  metrics_port=metrics_port,
                  metrics_path=metrics_path,
                  metrics_host=metrics_host,
                  listen_host=listen_host,
                  listen_port=listen_port,
                  listen_socket=listen_socket)
    listen_socket.close()

    rc = target.wait()

    # Pipes hit EOF once the target (and any fd-inheriting descendants) exit.
    for thread in drain_threads:
        thread.join(timeout=2.0)
    if writer is not None:
        # Terminal status record, written last so the reader always has a
        # guaranteed error signal on abnormal exit even if the target printed
        # nothing parseable.
        if rc < 0:
            writer.write_record({'ts': time.time_ns(), 'signal': -rc})
        elif rc != 0:
            writer.write_record({'ts': time.time_ns(), 'exit_code': rc})
        writer.close()

    if rc < 0:
        logger.debug('Target pid=%s terminated by signal %s', target.pid, -rc)
    elif rc != 0:
        logger.debug('Target pid=%s exited with code %s', target.pid, rc)
    _exit_like(rc)
