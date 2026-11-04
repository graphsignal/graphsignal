"""Per-thread CPU metrics for one observed pid, read from /proc.

The process recorder says how much CPU a process uses; this one says which
threads use it and whether they got the CPU they asked for. Each tick it
walks `<proc_root>/<pid>/task/<tid>/` and reads `comm`, `schedstat` (on-CPU
ns, run-queue wait ns, times scheduled; `CONFIG_SCHED_INFO`), `stat`
(utime/stime fallback when `schedstat` is absent) and `status` (context
switches, which `/proc/<pid>/status` reports for the leader thread only).
Whole-process CPU time and page faults come from `/proc/<pid>/stat`.

Every emitted total is owned by the recorder: per tid it remembers the last
values read and adds the increase to the tid's current frame. A thread that
exits freezes its contribution and a frame never decreases, so the collector
sees ordinary deltas. Frames are thread names: `comm` for threads that named
themselves (`cuda-EvtHandlr`, `ncclProxyProg`, `ZMQbg/IO/0`), aggregated over
threads sharing the name. Threads named like the main thread (CPython below
3.14 names none of its threads; they inherit the main thread's `comm`) are
split by tid as `<comm>/<tid>` for the first `MAX_SPLIT_THREADS` of them,
the rest fold into `<comm>`; the main thread itself is `<comm>/main`.
"""

import logging
import os
import time

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.signals.metrics import MAX_PROFILE_FRAMES

logger = logging.getLogger('graphsignal')

_PROC_ROOT = '/proc'
MAX_SPLIT_THREADS = 32
MAX_FRAMES = MAX_PROFILE_FRAMES - 1
OTHER_FRAME = 'other'
MAIN_THREAD_SUFFIX = 'main'

# /proc/<pid>/stat fields after the `(comm)` token, 0-based.
_STAT_MINFLT = 7
_STAT_MAJFLT = 9
_STAT_UTIME = 11
_STAT_STIME = 12


class ThreadRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None, proc_root=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._proc_root = proc_root or _PROC_ROOT
        self._enabled = False
        try:
            self._ns_per_tick = 1_000_000_000 // os.sysconf('SC_CLK_TCK')
        except (AttributeError, ValueError, OSError):
            self._ns_per_tick = 10_000_000
        # Names the main thread has had: a thread with one of them is unnamed
        # and inherited it at creation (torch renames the main thread to
        # pt_main_thread after the first Python threads were created).
        self._main_names = set()
        # frame -> [cpu_ns, wait_ns, scheduled]
        self._frames = {}
        # tid -> (cpu_ns, wait_ns, scheduled) at the last tick
        self._tid_last = {}
        # tid -> (voluntary, involuntary) at the last tick
        self._tid_ctxt_last = {}
        self._ctxt_totals = [0, 0]
        # tids given a `<comm>/<tid>` frame of their own, in order of arrival
        self._split_tids = set()
        self._has_schedstat = False

    def _proc_dir(self):
        return os.path.join(self._proc_root, str(self.pid))

    def setup(self):
        if self.pid is None:
            return
        # No /proc task tree (macOS, a pid that died before being seen)
        # disables the recorder; nothing else here is platform-specific.
        self._enabled = os.path.isdir(os.path.join(self._proc_dir(), 'task'))

    def on_tick(self):
        if not self._enabled:
            return
        watcher = graphsignal.watcher.watcher()
        now_ns = time.time_ns()
        tags = {'process.pid': str(self.pid)}

        try:
            self._record_process(watcher, now_ns, tags)
            self._record_threads(watcher, now_ns, tags)
        except Exception:
            logger.debug('Failed to read /proc for pid %s', self.pid,
                         exc_info=True)

    # --- whole-process counters -------------------------------------------

    def _record_process(self, watcher, now_ns, tags):
        fields = self._read_stat_fields(os.path.join(self._proc_dir(), 'stat'))
        if fields is None:
            return
        try:
            utime = int(fields[_STAT_UTIME]) * self._ns_per_tick
            stime = int(fields[_STAT_STIME]) * self._ns_per_tick
            minflt = int(fields[_STAT_MINFLT])
            majflt = int(fields[_STAT_MAJFLT])
        except (IndexError, ValueError):
            return
        for mode, total in (('user', utime), ('system', stime)):
            watcher.set_counter(
                name='process_cpu_nanoseconds', total=total,
                measurement_ts=now_ns, tags=dict(tags, mode=mode))
        for kind, total in (('minor', minflt), ('major', majflt)):
            watcher.set_counter(
                name='process_page_faults', total=total,
                measurement_ts=now_ns, tags=dict(tags, kind=kind))

    # --- per-thread profiles ------------------------------------------------

    def _record_threads(self, watcher, now_ns, tags):
        task_dir = os.path.join(self._proc_dir(), 'task')
        try:
            tids = sorted(int(name) for name in os.listdir(task_dir)
                          if name.isdigit())
        except OSError:
            return
        if not tids:
            return

        main_comm = self._read_text(os.path.join(task_dir, str(self.pid), 'comm'))
        if main_comm:
            self._main_names.add(main_comm)

        live = 0
        for tid in tids:
            tid_dir = os.path.join(task_dir, str(tid))
            comm = self._read_text(os.path.join(tid_dir, 'comm'))
            sched = self._read_schedstat(tid_dir)
            if comm is None or sched is None:
                # Exited between listdir and read; its last contribution
                # stays in its frame.
                continue
            live += 1
            frame = self._frame_for(tid, comm)
            self._add_delta(frame, tid, sched)
            self._add_ctxt(tid, tid_dir)

        if live == 0:
            return

        watcher.set_gauge(
            name='process_threads', value=live,
            measurement_ts=now_ns, tags=tags)
        for kind, total in (('voluntary', self._ctxt_totals[0]),
                            ('involuntary', self._ctxt_totals[1])):
            watcher.set_counter(
                name='process_context_switches', total=total,
                measurement_ts=now_ns, tags=dict(tags, kind=kind))

        if not self._frames:
            return
        samples = {name: totals[2] for name, totals in self._frames.items()}
        watcher.set_profile(
            name='process_threads_cpu_nanoseconds',
            frames={name: totals[0] for name, totals in self._frames.items()},
            samples=samples if self._has_schedstat else None,
            measurement_ts=now_ns, tags=tags)
        if self._has_schedstat:
            watcher.set_profile(
                name='process_threads_wait_nanoseconds',
                frames={name: totals[1] for name, totals in self._frames.items()},
                samples=samples,
                measurement_ts=now_ns, tags=tags)

    def _frame_for(self, tid, comm):
        if comm not in self._main_names:
            name = comm
        elif tid == self.pid:
            name = f'{comm}/{MAIN_THREAD_SUFFIX}'
        else:
            if tid not in self._split_tids and len(self._split_tids) < MAX_SPLIT_THREADS:
                self._split_tids.add(tid)
            name = f'{comm}/{tid}' if tid in self._split_tids else comm
        if name in self._frames:
            return name
        if len(self._frames) >= MAX_FRAMES:
            return OTHER_FRAME
        return name

    def _add_delta(self, frame, tid, sched):
        last = self._tid_last.get(tid, (0, 0, 0))
        totals = self._frames.setdefault(frame, [0, 0, 0])
        for i in range(3):
            delta = sched[i] - last[i]
            if delta > 0:
                totals[i] += delta
        self._tid_last[tid] = sched

    def _add_ctxt(self, tid, tid_dir):
        current = self._read_ctxt_switches(tid_dir)
        if current is None:
            return
        last = self._tid_ctxt_last.get(tid, (0, 0))
        for i in range(2):
            delta = current[i] - last[i]
            if delta > 0:
                self._ctxt_totals[i] += delta
        self._tid_ctxt_last[tid] = current

    # --- /proc readers --------------------------------------------------------

    def _read_schedstat(self, tid_dir):
        """(cpu_ns, wait_ns, scheduled) for one task, from schedstat or, on a
        kernel without it, from stat with no wait and no sample count."""
        text = self._read_text(os.path.join(tid_dir, 'schedstat'))
        if text is not None:
            fields = text.split()
            if len(fields) >= 3:
                try:
                    values = (int(fields[0]), int(fields[1]), int(fields[2]))
                    self._has_schedstat = True
                    return values
                except ValueError:
                    pass
        fields = self._read_stat_fields(os.path.join(tid_dir, 'stat'))
        if fields is None:
            return None
        try:
            cpu_ns = (int(fields[_STAT_UTIME]) + int(fields[_STAT_STIME])) * self._ns_per_tick
        except (IndexError, ValueError):
            return None
        return (cpu_ns, 0, 0)

    def _read_ctxt_switches(self, tid_dir):
        text = self._read_text(os.path.join(tid_dir, 'status'))
        if text is None:
            return None
        voluntary = involuntary = None
        for line in text.splitlines():
            if line.startswith('voluntary_ctxt_switches:'):
                voluntary = self._int_after_colon(line)
            elif line.startswith('nonvoluntary_ctxt_switches:'):
                involuntary = self._int_after_colon(line)
        if voluntary is None or involuntary is None:
            return None
        return (voluntary, involuntary)

    @staticmethod
    def _int_after_colon(line):
        try:
            return int(line.split(':', 1)[1].strip())
        except (IndexError, ValueError):
            return None

    def _read_stat_fields(self, path):
        """Fields of a /proc stat file after the `(comm)` token, which may
        itself contain spaces and parentheses."""
        text = self._read_text(path)
        if text is None:
            return None
        _, sep, rest = text.rpartition(')')
        if not sep:
            return None
        return rest.split()

    @staticmethod
    def _read_text(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None
