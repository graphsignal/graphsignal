import logging
import os
import re
import threading

import psutil

logger = logging.getLogger('graphsignal')


REDACTED = '[redacted]'

# Flags whose value is a credential: `--api-key`, `--hf-token`, `--password`, ...
_CREDENTIAL_FLAG_RE = re.compile(
    r'^--?(?:[a-z0-9]+[-_])*'
    r'(?:api[-_]?key|access[-_]?key|secret[-_]?key|token|secret|password|passwd|credentials?)$',
    re.IGNORECASE)


def redact_command_line(cmdline):
    """Join argv into one string, with the values of credential flags replaced."""
    parts = []
    redact_next = False
    for arg in cmdline:
        if redact_next:
            redact_next = False
            if not arg.startswith('-'):
                parts.append(REDACTED)
                continue
        flag, sep, _ = arg.partition('=')
        if _CREDENTIAL_FLAG_RE.match(flag):
            if sep:
                parts.append(f'{flag}={REDACTED}')
                continue
            redact_next = True
        parts.append(arg)
    return ' '.join(parts)


class PidMonitor:
    def __init__(self, target_pid: int, poll_interval: float = 2.0):
        self._target_pid = int(target_pid)
        self._poll_interval = float(poll_interval)
        self._self_pid = os.getpid()
        self._child_pids: set = set()
        self._listeners: list = []
        self._stop_event = threading.Event()
        self._thread = None
        self._target_seen = False
        self._terminated_emitted = False

    def add_listener(self, listener):
        self._listeners.append(listener)

    def child_pids(self):
        return set(self._child_pids)

    def setup(self):
        self._stop_event = threading.Event()
        # Fire on_target_known before polling starts: some observations (e.g.
        # the supervisor's console-log dir) are pid-derived artifacts that must
        # be picked up even when the target dies before it is ever seen alive.
        try:
            self._emit('on_target_known', self._target_pid)
        except Exception as exc:
            logger.error('Error emitting on_target_known: %s', exc, exc_info=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._poll_interval + 1.0)
            self._thread = None

    def _loop(self):
        # First tick runs immediately so target_created fires without waiting one full interval.
        try:
            self._tick()
        except Exception as exc:
            logger.error('Error in pid monitor tick: %s', exc, exc_info=True)

        while not self._stop_event.wait(self._poll_interval):
            try:
                if self._terminated_emitted:
                    return
                self._tick()
            except Exception as exc:
                logger.error('Error in pid monitor tick: %s', exc, exc_info=True)

    def _tick(self):
        if self._terminated_emitted:
            return

        try:
            proc = psutil.Process(self._target_pid)
            alive = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            proc = None
            alive = False
        except psutil.Error:
            return

        if not alive:
            self._emit_terminated()
            return

        if not self._target_seen:
            args = _read_args(proc)
            self._target_seen = True
            self._emit('on_target_created', args)

        try:
            children = [c for c in proc.children(recursive=True) if c.pid != self._self_pid]
        except psutil.NoSuchProcess:
            self._emit_terminated()
            return
        except psutil.Error:
            return

        current = {c.pid: c for c in children}
        current_pids = set(current.keys())

        new_pids = current_pids - self._child_pids
        gone_pids = self._child_pids - current_pids

        for pid in new_pids:
            args = _read_args(current[pid])
            self._child_pids.add(pid)
            self._emit('on_child_created', pid, args)

        for pid in gone_pids:
            self._child_pids.discard(pid)
            self._emit('on_child_terminated', pid)

    def _emit_terminated(self):
        if self._terminated_emitted:
            return
        self._terminated_emitted = True

        for pid in list(self._child_pids):
            self._emit('on_child_terminated', pid)
        self._child_pids.clear()

        self._emit('on_target_terminated')

    def _emit(self, method_name, *args):
        for listener in self._listeners:
            method = getattr(listener, method_name, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception as exc:
                logger.error('Error in pid monitor listener %s: %s', method_name, exc, exc_info=True)


def _read_args(proc) -> str:
    try:
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
        return ''
    if not cmdline:
        return ''
    return redact_command_line(cmdline)
