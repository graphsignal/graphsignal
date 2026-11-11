"""Imports metrics written by in-process libraries for one observed pid.

Writers inside the workload process (the CUPTI/ROCm injection libraries and
any probe writer) rewrite `<base>/graphsignal_<pid>/<lib>.json` atomically
every second with full cumulative state; every read is an immutable snapshot.
Each tick the recorder reads every file in the pid's directory and sets the
latest values into the metric store.
"""

import glob
import json
import logging
import os
import shutil
import time

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.shm_files import is_owned_dir, read_regular_file

logger = logging.getLogger('graphsignal')

_SHM_BASE = '/dev/shm'
_SHM_DIR_PREFIX = 'graphsignal_'

# A metrics file is a full cumulative snapshot; anything larger is not read.
MAX_FILE_BYTES = 32 * 1024 * 1024

# Context keys published by metric writers, mapped to metric tags.
_CONTEXT_TAG_MAP = {
    'rank': 'process.rank',
    'local_rank': 'process.local_rank',
    'world_size': 'process.world_size',
    'local_world_size': 'process.local_world_size',
    'master_addr': 'distributed.master_addr',
    'master_port': 'distributed.master_port',
    'slurm_job_id': 'slurm.job_id',
    'slurm_step_id': 'slurm.step_id',
    'slurm_node_id': 'slurm.node_id',
    'slurm_node_count': 'slurm.node_count',
}


class ShmRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None, shm_base_dir=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._base_dir = shm_base_dir or _SHM_BASE
        self._file_start_ts = {}
        self._seen_log_ts = {}

    def shm_dir(self):
        return os.path.join(self._base_dir, f'{_SHM_DIR_PREFIX}{self.pid}')

    def setup(self):
        if self.pid is None:
            return
        # One sweep per watcher is enough; the root recorder does it.
        if self.pid == self.root_pid:
            self._sweep_stale_dirs()

    def shutdown(self):
        try:
            if os.path.isdir(self.shm_dir()):
                shutil.rmtree(self.shm_dir(), ignore_errors=True)
        except Exception:
            pass

    def on_tick(self):
        if self.pid is None:
            return
        shm_dir = self.shm_dir()
        if not is_owned_dir(shm_dir, self.pid):
            return

        watcher = graphsignal.watcher.watcher()
        for filepath in sorted(glob.glob(os.path.join(shm_dir, '*.json'))):
            raw = read_regular_file(filepath, MAX_FILE_BYTES + 1)
            if raw is None or len(raw) > MAX_FILE_BYTES:
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            try:
                self._import_file(watcher, filepath, data)
            except Exception:
                logger.debug('Error importing shm metrics from %s',
                             filepath, exc_info=True)

    def _sweep_stale_dirs(self):
        """Remove `graphsignal_<pid>` dirs whose pid is no longer alive."""
        try:
            for path in glob.glob(os.path.join(self._base_dir, _SHM_DIR_PREFIX + '*')):
                if not os.path.isdir(path):
                    continue
                name = os.path.basename(path)
                pid_str = name[len(_SHM_DIR_PREFIX):]
                try:
                    pid = int(pid_str)
                except ValueError:
                    # Not a metrics dir (e.g. graphsignal_log_<pid>).
                    continue
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    shutil.rmtree(path, ignore_errors=True)
                    logger.debug('Removed stale shm dir: %s', path)
                except OSError:
                    pass
        except Exception:
            logger.debug('Error sweeping stale shm dirs', exc_info=True)

    def _import_file(self, watcher, filepath, data):
        if not isinstance(data, dict) or data.get('version') != 1:
            logger.debug('Skipping shm metrics file with unsupported schema: %s',
                         filepath)
            return

        start_ts = data.get('start_ts')
        last_start_ts = self._file_start_ts.get(filepath)
        if last_start_ts is not None and start_ts != last_start_ts:
            logger.debug('Metric writer restarted: %s', filepath)
        self._file_start_ts[filepath] = start_ts

        write_ts = data.get('write_ts') or time.time_ns()

        base_tags = {'process.pid': str(self.pid)}
        context = data.get('context')
        if isinstance(context, dict):
            for key, value in context.items():
                tag_key = _CONTEXT_TAG_MAP.get(key)
                if tag_key and value not in (None, ''):
                    base_tags[tag_key] = str(value)

        for entry in data.get('metrics', []):
            if not isinstance(entry, dict):
                continue
            name = entry.get('name')
            metric_type = entry.get('type')
            if not name or not metric_type:
                continue

            tags = dict(base_tags)
            entry_tags = entry.get('tags')
            if isinstance(entry_tags, dict):
                tags.update(entry_tags)

            try:
                if metric_type == 'gauge':
                    watcher.set_gauge(name, entry['value'], write_ts, tags=tags)
                elif metric_type == 'counter':
                    watcher.set_counter(name, entry['value'], write_ts, tags=tags)
                elif metric_type == 'histogram':
                    # Bins and exact aggregates arrive independently; the
                    # writer omits the aggregates for an instrument that has
                    # never been recorded, which set_histogram then rejects —
                    # the same way an all-empty histogram was always dropped.
                    watcher.set_histogram(
                        name,
                        bins=entry.get('bins'),
                        counts=entry.get('counts'),
                        measurement_ts=write_ts,
                        tags=tags,
                        count=entry.get('count'),
                        sum_val=entry.get('sum'),
                        min_val=entry.get('min'),
                        max_val=entry.get('max'))
                elif metric_type == 'profile':
                    # A frame arrives as [cumulative value, sample count].
                    frame_entries = entry['frames'].items()
                    watcher.set_profile(
                        name,
                        frames={f: pair[0] for f, pair in frame_entries},
                        samples={f: pair[1] for f, pair in frame_entries},
                        measurement_ts=write_ts,
                        tags=tags)
            except (KeyError, TypeError, ValueError):
                logger.debug('Skipping malformed metric entry in %s: %s',
                             filepath, name, exc_info=True)

        self._log_entries(filepath, data.get('log'))

    def _log_entries(self, filepath, entries):
        if not entries:
            return
        seen = self._seen_log_ts.setdefault(filepath, 0)
        try:
            max_ts = seen
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                ts = entry.get('ts') or 0
                if ts <= seen:
                    continue
                max_ts = max(max_ts, ts)
                msg = entry.get('msg', '')
                if msg:
                    logger.debug('workload: %s', str(msg).rstrip())
            self._seen_log_ts[filepath] = max_ts
        except Exception:
            pass
