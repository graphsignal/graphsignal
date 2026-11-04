import logging
import os
import sys
import time

import psutil

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

_PROC_ROOT = '/proc'
_CGROUP_ROOT = '/sys/fs/cgroup'
# cgroup v1 mounts the cpu controller under one of these names.
_CGROUP_V1_CPU_MOUNTS = ('cpu,cpuacct', 'cpu')


class ProcessRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None,
                 proc_root=None, cgroup_root=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._process_start_ts = time.time_ns()
        self._psutil_proc = None
        self._cpu_percent_initialized = False
        self._proc_root = proc_root or _PROC_ROOT
        self._cgroup_root = cgroup_root or _CGROUP_ROOT

    def setup(self):
        if self.pid is not None:
            try:
                self._psutil_proc = psutil.Process(self.pid)
                # First call seeds the internal counter; subsequent calls return real values.
                self._psutil_proc.cpu_percent(interval=None)
                self._cpu_percent_initialized = True
                try:
                    create_time = self._psutil_proc.create_time()
                    if create_time:
                        self._process_start_ts = int(create_time * 1e9)
                except psutil.Error:
                    pass
            except psutil.Error:
                self._psutil_proc = None

    def on_tick(self):
        if self._psutil_proc is None:
            return

        watcher = graphsignal.watcher.watcher()
        now_ns = time.time_ns()

        try:
            cpu_percent = self._psutil_proc.cpu_percent(interval=None)
            try:
                cpu_count = psutil.cpu_count() or 1
            except Exception:
                cpu_count = 1
            cpu_percent_normalized = cpu_percent / cpu_count if cpu_count else cpu_percent
            if cpu_percent_normalized > 0:
                watcher.set_gauge(
                    name='process_cpu_usage_percent',
                    value=cpu_percent_normalized,
                    measurement_ts=now_ns,
                    tags={'process.pid': str(self.pid)})
        except psutil.Error:
            pass

        try:
            mem = self._psutil_proc.memory_info()
            if mem.rss > 0:
                watcher.set_gauge(
                    name='process_memory_usage_bytes',
                    value=mem.rss,
                    measurement_ts=now_ns,
                    tags={'process.pid': str(self.pid)})
            if mem.vms > 0:
                watcher.set_gauge(
                    name='process_memory_virtual_bytes',
                    value=mem.vms,
                    measurement_ts=now_ns,
                    tags={'process.pid': str(self.pid)})
        except psutil.Error:
            pass

        try:
            affinity = self._psutil_proc.cpu_affinity()
            if affinity:
                watcher.set_gauge(
                    name='process_cpu_allowed',
                    value=len(affinity),
                    measurement_ts=now_ns,
                    tags={'process.pid': str(self.pid)})
        except (psutil.Error, AttributeError, OSError):
            pass

        # Every process of the workload normally shares one cgroup, so the
        # root recorder reports it once for the tree.
        if self.pid == self.root_pid:
            try:
                self._record_cgroup_cpu(watcher, now_ns)
            except Exception:
                logger.debug('Failed to read cgroup CPU stats for pid %s',
                             self.pid, exc_info=True)

        process_attrs = {}
        if self.args:
            process_attrs['process.command_line'] = self.args

        is_self = self.pid == os.getpid()
        if is_self:
            try:
                if sys.argv and 'process.command_line' not in process_attrs:
                    process_attrs['process.command_line'] = ' '.join(sys.argv)
            except Exception:
                pass
            try:
                if sys.version_info and len(sys.version_info) >= 3:
                    process_attrs['runtime.name'] = 'python'
                    process_attrs['runtime.version'] = (
                        f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
                    )
            except Exception:
                pass

        resource_tags = None
        if self.pid is not None:
            resource_tags = {'process.pid': str(self.pid)}
            # Links every process resource to the watcher's target (root)
            # process — the main server process. The root's own resource has
            # process.pid == process.root_pid, which is how consumers pick
            # the authoritative command line over worker command lines.
            if self.root_pid is not None:
                resource_tags['process.root_pid'] = str(self.root_pid)
        watcher.update_resource(
            'process',
            tags=resource_tags,
            attributes=process_attrs,
            first_seen_ts=self._process_start_ts)

    # --- cgroup CPU limit and throttling ---------------------------------

    def _record_cgroup_cpu(self, watcher, now_ns):
        """Emit the CPU quota and CFS throttling of the process's cgroup.

        A missing or unreadable file emits nothing (null, not 0): a host
        without cgroup accounting is not a host with zero throttling.
        """
        cgroup = self._resolve_cgroup()
        if cgroup is None:
            return
        version, cgroup_dir = cgroup
        tags = {'process.pid': str(self.pid)}

        stat = self._read_kv_file(os.path.join(cgroup_dir, 'cpu.stat'))
        if stat is not None:
            if 'nr_throttled' in stat:
                watcher.set_counter(
                    name='process_cpu_throttled_periods',
                    total=stat['nr_throttled'],
                    measurement_ts=now_ns, tags=tags)
            if version == 2 and 'throttled_usec' in stat:
                watcher.set_counter(
                    name='process_cpu_throttled_nanoseconds',
                    total=stat['throttled_usec'] * 1000,
                    measurement_ts=now_ns, tags=tags)
            elif version == 1 and 'throttled_time' in stat:
                watcher.set_counter(
                    name='process_cpu_throttled_nanoseconds',
                    total=stat['throttled_time'],
                    measurement_ts=now_ns, tags=tags)

        quota = self._read_cpu_quota(version, cgroup_dir)
        if quota is not None:
            watcher.set_gauge(
                name='process_cpu_quota',
                value=quota,
                measurement_ts=now_ns, tags=tags)

    def _resolve_cgroup(self):
        """Return (version, directory) of the process's cpu cgroup, or None."""
        try:
            with open(os.path.join(self._proc_root, str(self.pid), 'cgroup')) as f:
                lines = f.read().splitlines()
        except OSError:
            return None
        v1_path = None
        v2_path = None
        for line in lines:
            parts = line.split(':', 2)
            if len(parts) != 3:
                continue
            hierarchy, controllers, path = parts
            if hierarchy == '0' and controllers == '':
                v2_path = path
            elif 'cpu' in controllers.split(','):
                v1_path = path
        # A hybrid host lists both; the cpu controller is on v1 there.
        if v1_path is not None:
            for mount in _CGROUP_V1_CPU_MOUNTS:
                cgroup_dir = os.path.join(
                    self._cgroup_root, mount, v1_path.lstrip('/'))
                if os.path.isdir(cgroup_dir):
                    return (1, cgroup_dir)
        if v2_path is not None:
            cgroup_dir = os.path.join(self._cgroup_root, v2_path.lstrip('/'))
            if os.path.isdir(cgroup_dir):
                return (2, cgroup_dir)
        return None

    def _read_cpu_quota(self, version, cgroup_dir):
        """The cgroup's CPU quota as a CPU count, or None when unlimited."""
        try:
            if version == 2:
                with open(os.path.join(cgroup_dir, 'cpu.max')) as f:
                    fields = f.read().split()
                if len(fields) != 2 or fields[0] == 'max':
                    return None
                quota_us, period_us = int(fields[0]), int(fields[1])
            else:
                with open(os.path.join(cgroup_dir, 'cpu.cfs_quota_us')) as f:
                    quota_us = int(f.read().strip())
                with open(os.path.join(cgroup_dir, 'cpu.cfs_period_us')) as f:
                    period_us = int(f.read().strip())
                if quota_us < 0:
                    return None
        except (OSError, ValueError):
            return None
        if period_us <= 0:
            return None
        return quota_us / period_us

    @staticmethod
    def _read_kv_file(path):
        """Parse a `key value` per line file (cpu.stat) into ints."""
        try:
            with open(path) as f:
                lines = f.read().splitlines()
        except OSError:
            return None
        values = {}
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                continue
        return values
