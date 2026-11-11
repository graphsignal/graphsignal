import json
import logging
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.shm_files import is_owned_dir, read_regular_file
from graphsignal.signals.logs import LogStore

logger = logging.getLogger('graphsignal')

MAX_ENTRIES_PER_DRAIN = 50

# Bytes read from one segment per drain; the rest is picked up on later passes.
MAX_BYTES_PER_DRAIN = 4 * 1024 * 1024

# Must match graphsignal/launchers/supervisor.py; patched in tests.
_SHM_BASE = '/dev/shm'
_LOG_DIR_PREFIX = 'graphsignal_log_'

_SEGMENT_RE = re.compile(r'^log_(\d+)\.jsonl$')

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

_TRACEBACK_MARKER = 'Traceback (most recent call last):'

# vLLM: `<LEVEL> MM-DD HH:MM:SS [file:line] msg` (vllm/logger.py _FORMAT);
# NewLineFormatter re-prefixes every continuation line identically.
_VLLM_RE = re.compile(
    r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+'
    r'\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)? \[[^\]]*:\d+\] ?(.*)$')

# SGLang / generic bracketed timestamp: `[YYYY-MM-DD HH:MM:SS[.mmm][ TP0…]] msg`.
_TS_PREFIX_RE = re.compile(
    r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?[^\]]*\] ?(.*)$')

# TRT-LLM: `[TRT-LLM] [E] …` runtime lines and `[TRT-LLM][ERROR]` bootstrap lines.
_TRTLLM_RE = re.compile(r'^\[TRT-LLM\] ?\[([A-Z]+)\] ?(.*)$')
_TRTLLM_LEVELS = {
    'V': 'debug', 'D': 'debug', 'I': 'info', 'W': 'warning', 'E': 'error',
    'TRACE': 'debug', 'DEBUG': 'debug', 'INFO': 'info',
    'WARNING': 'warning', 'ERROR': 'error', 'FATAL': 'critical',
}

# Leading level tokens: uvicorn's `ERROR:` levelprefix, `[ERROR]`, bare `ERROR`.
_LEVEL_TOKEN_RE = re.compile(
    r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)\b:?\s*(.*)$')
_BRACKET_LEVEL_RE = re.compile(
    r'^\[(DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)\]\s*(.*)$')

# CLI / argparse errors printed straight to stderr, e.g. argparse's
# "<prog>: error: <msg>" ("sglang serve: error: unrecognized arguments: --x")
# and leading "error:"/"fatal:"/"panic:" from many tools. `<prog>: ` is
# optional and may contain spaces. The keyword must be followed by a colon, so
# prose like "error rate high" or "0 errors" does not match. This is the
# dominant invalid-flag / bad-config crash signal, which carries no level name
# or timestamp and so is missed by the format-specific patterns above.
_CLI_ERROR_RE = re.compile(
    r'^(?:[^:\n]{1,80}: )?(?:error|fatal|panic):', re.IGNORECASE)

# SGLang's format has no level name; these message markers identify errors
# (sglang scheduler.py / tokenizer_manager.py).
_SGLANG_ERROR_MARKERS = ('Scheduler hit an exception', 'Dumping requests before crash')

_ERROR_LEVELS = ('error', 'critical', 'fatal')

# Closing line of a Python traceback: `SomeError: msg`, `pkg.mod.SomeError`,
# `KeyboardInterrupt`, … Used to decide whether a non-indented line still
# belongs to an open traceback block.
_TB_CLOSING_RE = re.compile(
    r'^[A-Za-z_][\w\.]*(?:Error|Exception|Warning|Interrupt|Exit|Fault|Timeout)\b')


class _ParsedLine:
    __slots__ = ('record_start', 'level', 'text')

    def __init__(self, record_start, level, text):
        # record_start: the line carries an engine log-record prefix (it opens
        # a new logical record rather than continuing a raw multi-line block).
        self.record_start = record_start
        self.level = level
        self.text = text


def _parse_line(line: str) -> _ParsedLine:
    m = _VLLM_RE.match(line)
    if m:
        return _ParsedLine(True, m.group(1).lower(), m.group(2))

    m = _TRTLLM_RE.match(line)
    if m:
        return _ParsedLine(True, _TRTLLM_LEVELS.get(m.group(1)), m.group(2))

    m = _TS_PREFIX_RE.match(line)
    if m:
        inner = m.group(1)
        lm = _LEVEL_TOKEN_RE.match(inner) or _BRACKET_LEVEL_RE.match(inner)
        if lm:
            return _ParsedLine(True, lm.group(1).lower(), lm.group(2))
        if any(marker in inner for marker in _SGLANG_ERROR_MARKERS):
            return _ParsedLine(True, 'error', inner)
        if inner.startswith(_TRACEBACK_MARKER):
            return _ParsedLine(True, 'error', inner)
        return _ParsedLine(True, None, inner)

    m = _BRACKET_LEVEL_RE.match(line)
    if m:
        return _ParsedLine(True, m.group(1).lower(), m.group(2))

    m = _LEVEL_TOKEN_RE.match(line)
    if m:
        return _ParsedLine(True, m.group(1).lower(), m.group(2))

    if _CLI_ERROR_RE.match(line):
        # Keep the whole line as the message: the "<prog>: error: " prefix is
        # the useful context (which command and what it rejected).
        return _ParsedLine(True, 'error', line)

    return _ParsedLine(False, None, line)


class _StreamExtractor:
    """Stateful per-stream line grouper: turns a line sequence into error
    entries, folding multi-line tracebacks into one entry's `exception`."""

    def __init__(self):
        self._pending = None  # {'ts', 'message', 'exc_lines', 'exc_size', 'in_tb'}

    def feed(self, ts_ns: int, line: str) -> List[dict]:
        out = []
        line = _ANSI_RE.sub('', line.rstrip('\r\n'))
        if not line.strip():
            return out
        parsed = _parse_line(line)
        content = parsed.text if parsed.record_start else line

        if self._pending is not None:
            if self._continues_pending(parsed, content):
                self._append_exc(content)
                if self._pending['in_tb'] and not content[:1].isspace() \
                        and not content.startswith(_TRACEBACK_MARKER):
                    # Non-indented closing exception line ends the block.
                    out.append(self._finalize())
                return out
            out.append(self._finalize())

        if parsed.level in _ERROR_LEVELS:
            self._pending = {
                'ts': ts_ns, 'message': content, 'exc_lines': [],
                'exc_size': 0, 'in_tb': _TRACEBACK_MARKER in content,
            }
        elif not parsed.record_start and line.startswith(_TRACEBACK_MARKER):
            # Bare traceback (e.g. an uncaught crash printed straight to
            # stderr) with no preceding error line.
            self._pending = {
                'ts': ts_ns, 'message': line, 'exc_lines': [line],
                'exc_size': len(line), 'in_tb': True,
            }
        return out

    def flush(self) -> List[dict]:
        if self._pending is None:
            return []
        return [self._finalize()]

    def _continues_pending(self, parsed: _ParsedLine, content: str) -> bool:
        pending = self._pending
        if content.startswith(_TRACEBACK_MARKER):
            pending['in_tb'] = True
            return True
        if not pending['in_tb']:
            return False
        if content[:1].isspace():
            return True
        # Non-indented line inside a traceback: only the closing
        # `SomeError: …` line belongs to the block.
        return bool(_TB_CLOSING_RE.match(content))

    def _append_exc(self, content: str) -> None:
        pending = self._pending
        if pending['exc_size'] >= LogStore.STACK_TRACE_SIZE_LIMIT:
            return
        pending['exc_lines'].append(content)
        pending['exc_size'] += len(content) + 1

    def _finalize(self) -> dict:
        pending, self._pending = self._pending, None
        exception = '\n'.join(pending['exc_lines'])
        return {
            'ts': pending['ts'],
            'message': pending['message'][:LogStore.MESSAGE_SIZE_LIMIT],
            'exception': exception[:LogStore.STACK_TRACE_SIZE_LIMIT],
        }


class LogRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._disabled = True
        self._offsets: Dict[str, int] = {}
        self._extractors: Dict[str, _StreamExtractor] = {}

    def _shm_dir(self) -> str:
        # Written by the graphsignal-run supervisor; own prefix so the
        # CUPTI/ROCm sweepers and rmtrees can never touch it (and vice versa).
        return os.path.join(_SHM_BASE, f'{_LOG_DIR_PREFIX}{self.pid}')

    def setup(self):
        if not sys.platform.startswith('linux'):
            return
        if self.pid is None:
            logger.debug('LogRecorder requires a pid; skipping setup')
            return

        _sweep_stale_shm_dirs()

        self._disabled = False
        logger.debug('LogRecorder started for pid=%s', self.pid)

    def on_tick(self):
        if self._disabled:
            return
        self._drain()

    def finalize(self):
        if self._disabled:
            return
        self._drain()
        self._flush_extractors()

    def shutdown(self):
        if self._disabled:
            return
        self._disabled = True
        self._cleanup_shm_dir()

    def _drain(self):
        shm_dir = self._shm_dir()
        if not is_owned_dir(shm_dir):
            return
        try:
            names = os.listdir(shm_dir)
        except OSError:
            return

        segments = []
        for name in names:
            m = _SEGMENT_RE.match(name)
            if m:
                segments.append((int(m.group(1)), name))
        segments.sort()

        emitted = 0
        dropped = 0
        current_paths = set()
        for _, name in segments:
            path = os.path.join(shm_dir, name)
            current_paths.add(path)
            offset = self._offsets.get(path, 0)
            data = read_regular_file(path, MAX_BYTES_PER_DRAIN, offset)
            if data is None:
                continue
            # Consume complete lines only; a partially written record is
            # picked up on the next pass.
            end = data.rfind(b'\n')
            if end < 0:
                if len(data) >= MAX_BYTES_PER_DRAIN:
                    # One record longer than a whole pass: skip past it.
                    self._offsets[path] = offset + len(data)
                continue
            self._offsets[path] = offset + end + 1

            for raw in data[:end].split(b'\n'):
                try:
                    record = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict):
                    continue
                if 'dropped' in record:
                    logger.debug('Supervisor dropped %s console lines',
                                 record['dropped'])
                    continue
                if 'exit_code' in record or 'signal' in record:
                    # Terminal status record written by the supervisor: a
                    # guaranteed error signal on abnormal exit, independent of
                    # whatever (if anything) the target printed.
                    entry = _exit_status_entry(record)
                    if entry is None:
                        continue
                    if emitted < MAX_ENTRIES_PER_DRAIN:
                        self._record_entry(entry, 'exit')
                        emitted += 1
                    else:
                        dropped += 1
                    continue
                line = record.get('line')
                if line is None:
                    continue
                stream = record.get('stream', '')
                extractor = self._extractors.setdefault(stream, _StreamExtractor())
                for entry in extractor.feed(record.get('ts'), line):
                    if emitted < MAX_ENTRIES_PER_DRAIN:
                        self._record_entry(entry, stream)
                        emitted += 1
                    else:
                        dropped += 1

        # Rotated-away segments no longer need offsets.
        self._offsets = {p: o for p, o in self._offsets.items() if p in current_paths}

        if dropped:
            logger.debug('LogRecorder rate cap: dropped %s error entries this pass',
                         dropped)

    def _flush_extractors(self):
        for stream, extractor in self._extractors.items():
            for entry in extractor.flush():
                self._record_entry(entry, stream)

    def _record_entry(self, entry: dict, stream: str) -> None:
        try:
            if not graphsignal.watcher.is_configured():
                return
            graphsignal.watcher.watcher().log_store().log_message(
                message=entry['message'],
                level='error',
                exception=entry.get('exception', ''),
                timestamp_ns=entry['ts'],
                tags={
                    'process.pid': str(self.pid),
                    'scope.name': 'workload',
                    'stream': stream,
                })
        except Exception:
            logger.error('Failed to record log entry', exc_info=True)

    def _cleanup_shm_dir(self):
        try:
            if os.path.isdir(self._shm_dir()):
                shutil.rmtree(self._shm_dir(), ignore_errors=True)
        except Exception:
            pass


def _exit_status_entry(record: dict) -> Optional[dict]:
    """Build an error entry from a supervisor terminal-status record. Returns
    None for a clean (code 0) exit, which carries no error."""
    ts = record.get('ts')
    signum = record.get('signal')
    if signum:
        return {'ts': ts,
                'message': f'Process terminated by signal {signum}',
                'exception': ''}
    code = record.get('exit_code')
    if code:
        return {'ts': ts,
                'message': f'Process exited with non-zero code {code}',
                'exception': ''}
    return None


def _sweep_stale_shm_dirs():
    """Remove `/dev/shm/graphsignal_log_<pid>` directories whose pid is no
    longer running (leftovers from crashed supervisors). Scoped to the log
    prefix so it never touches the CUPTI/ROCm backends' dirs."""
    base = _SHM_BASE
    prefix = _LOG_DIR_PREFIX
    if not os.path.isdir(base):
        return
    try:
        entries = os.listdir(base)
    except OSError:
        return
    for name in entries:
        if not name.startswith(prefix):
            continue
        try:
            pid = int(name[len(prefix):])
        except ValueError:
            continue
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
            continue  # pid alive — leave its dir alone
        except ProcessLookupError:
            pass  # stale — fall through to rmtree
        except PermissionError:
            continue  # alive under another uid
        except OSError:
            continue
        shutil.rmtree(os.path.join(base, name), ignore_errors=True)
