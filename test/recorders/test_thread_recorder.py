import os
import shutil
import tempfile
import unittest

import graphsignal.watcher
from graphsignal.recorders.thread_recorder import (
    ThreadRecorder, MAX_SPLIT_THREADS, MAX_FRAMES, OTHER_FRAME)
from test.test_utils import configure_test_watcher, find_metric

PID = 4242


class FakeProc:
    """A /proc tree for one pid, built in a temp dir."""

    def __init__(self, root, pid=PID, comm='python3', utime=0, stime=0,
                 minflt=0, majflt=0):
        self.root = root
        self.pid = pid
        self.pid_dir = os.path.join(root, str(pid))
        os.makedirs(os.path.join(self.pid_dir, 'task'))
        self.set_process(comm, utime, stime, minflt, majflt)

    def set_process(self, comm='python3', utime=0, stime=0, minflt=0, majflt=0):
        self._write(os.path.join(self.pid_dir, 'stat'),
                    self._stat_line(self.pid, comm, utime, stime, minflt, majflt))

    @staticmethod
    def _stat_line(tid, comm, utime, stime, minflt=0, majflt=0):
        # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt
        # majflt cmajflt utime stime ...
        return (f'{tid} ({comm}) S 1 {tid} {tid} 0 -1 4194304 {minflt} 0 '
                f'{majflt} 0 {utime} {stime} 0 0 20 0 1 0 100 1000 10\n')

    def set_thread(self, tid, comm, cpu_ns=0, wait_ns=0, scheduled=0,
                   schedstat=True, utime=0, stime=0, voluntary=0,
                   involuntary=0):
        tid_dir = os.path.join(self.pid_dir, 'task', str(tid))
        os.makedirs(tid_dir, exist_ok=True)
        self._write(os.path.join(tid_dir, 'comm'), comm + '\n')
        schedstat_path = os.path.join(tid_dir, 'schedstat')
        if schedstat:
            self._write(schedstat_path, f'{cpu_ns} {wait_ns} {scheduled}\n')
        elif os.path.exists(schedstat_path):
            os.remove(schedstat_path)
        self._write(os.path.join(tid_dir, 'stat'),
                    self._stat_line(tid, comm, utime, stime))
        self._write(os.path.join(tid_dir, 'status'),
                    f'Name:\t{comm}\nState:\tS (sleeping)\n'
                    f'Cpus_allowed_list:\t0-7\n'
                    f'voluntary_ctxt_switches:\t{voluntary}\n'
                    f'nonvoluntary_ctxt_switches:\t{involuntary}\n')

    def remove_thread(self, tid):
        shutil.rmtree(os.path.join(self.pid_dir, 'task', str(tid)))

    @staticmethod
    def _write(path, text):
        with open(path, 'w') as f:
            f.write(text)


class ThreadRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.root = tempfile.mkdtemp()
        self.proc = FakeProc(self.root)
        self.recorder = ThreadRecorder(root_pid=PID, pid=PID,
                                       args='python3 app.py',
                                       proc_root=self.root)
        self.tags = {'process.pid': str(PID)}

    def tearDown(self):
        graphsignal.watcher.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def _export(self):
        return self.watcher.metric_store().export()

    def _frames(self, name):
        metric = find_metric(self._export(), name, self.tags)
        if metric is None:
            return None
        return metric.datapoint['frames']

    def _samples(self, name):
        return find_metric(self._export(), name, self.tags).datapoint['samples']

    def _counter(self, name, **tags):
        metric = find_metric(self._export(), name, dict(self.tags, **tags))
        self.assertIsNotNone(metric, name)
        return metric.datapoint['total']

    def test_profiles_counters_and_gauge_from_schedstat(self):
        self.proc.set_process(utime=300, stime=100, minflt=50, majflt=2)
        self.proc.set_thread(PID, 'python3', cpu_ns=5000, wait_ns=100,
                             scheduled=10, voluntary=7, involuntary=1)
        self.proc.set_thread(4300, 'cuda-EvtHandlr', cpu_ns=2000, wait_ns=50,
                             scheduled=4, voluntary=3, involuntary=2)
        self.recorder.setup()
        self.recorder.on_tick()

        cpu = self._frames('process_threads_cpu_nanoseconds')
        self.assertEqual(cpu, {'python3/main': 5000, 'cuda-EvtHandlr': 2000})
        self.assertEqual(self._samples('process_threads_cpu_nanoseconds'),
                         {'python3/main': 10, 'cuda-EvtHandlr': 4})
        wait = self._frames('process_threads_wait_nanoseconds')
        self.assertEqual(wait, {'python3/main': 100, 'cuda-EvtHandlr': 50})
        self.assertEqual(self._samples('process_threads_wait_nanoseconds'),
                         {'python3/main': 10, 'cuda-EvtHandlr': 4})

        ns_per_tick = self.recorder._ns_per_tick
        self.assertEqual(self._counter('process_cpu_nanoseconds', mode='user'),
                         300 * ns_per_tick)
        self.assertEqual(self._counter('process_cpu_nanoseconds', mode='system'),
                         100 * ns_per_tick)
        self.assertEqual(self._counter('process_page_faults', kind='minor'), 50)
        self.assertEqual(self._counter('process_page_faults', kind='major'), 2)
        self.assertEqual(self._counter('process_context_switches', kind='voluntary'), 10)
        self.assertEqual(self._counter('process_context_switches', kind='involuntary'), 3)
        threads = find_metric(self._export(), 'process_threads', self.tags)
        self.assertEqual(threads.type, 'gauge')
        self.assertEqual(threads.datapoint['value'], 2)

    def test_stat_fallback_without_schedstat(self):
        self.proc.set_thread(PID, 'python3', schedstat=False, utime=20, stime=5)
        self.proc.set_thread(4300, 'worker', schedstat=False, utime=4, stime=0)
        self.recorder.setup()
        self.recorder.on_tick()

        ns_per_tick = self.recorder._ns_per_tick
        self.assertEqual(self._frames('process_threads_cpu_nanoseconds'),
                         {'python3/main': 25 * ns_per_tick,
                          'worker': 4 * ns_per_tick})
        # No scheduling info: no sample counts, and no wait profile at all.
        self.assertEqual(self._samples('process_threads_cpu_nanoseconds'), {})
        self.assertIsNone(self._frames('process_threads_wait_nanoseconds'))

    def test_same_name_threads_aggregate(self):
        self.proc.set_thread(PID, 'python3', cpu_ns=1)
        self.proc.set_thread(4301, 'ZMQbg/IO/0', cpu_ns=100, scheduled=1)
        self.proc.set_thread(4302, 'ZMQbg/IO/0', cpu_ns=250, scheduled=2)
        self.recorder.setup()
        self.recorder.on_tick()

        self.assertEqual(self._frames('process_threads_cpu_nanoseconds')['ZMQbg/IO/0'], 350)
        self.assertEqual(self._samples('process_threads_cpu_nanoseconds')['ZMQbg/IO/0'], 3)

    def test_main_named_threads_split_by_tid_up_to_cap(self):
        self.proc.set_thread(PID, 'python3', cpu_ns=1)
        extra = MAX_SPLIT_THREADS + 3
        for i in range(extra):
            self.proc.set_thread(5000 + i, 'python3', cpu_ns=10)
        self.recorder.setup()
        self.recorder.on_tick()

        cpu = self._frames('process_threads_cpu_nanoseconds')
        split = [name for name in cpu if name.startswith('python3/')
                 and name != 'python3/main']
        self.assertEqual(len(split), MAX_SPLIT_THREADS)
        self.assertIn('python3/5000', cpu)
        # The rest fold into the bare name.
        self.assertEqual(cpu['python3'], 10 * (extra - MAX_SPLIT_THREADS))
        self.assertEqual(cpu['python3/main'], 1)

    def test_renamed_main_thread_marks_inherited_names(self):
        # Torch renames the main thread after Python threads were created.
        self.proc.set_thread(PID, 'python3', cpu_ns=10)
        self.proc.set_thread(4301, 'python3', cpu_ns=5)
        self.recorder.setup()
        self.recorder.on_tick()

        self.proc.set_thread(PID, 'pt_main_thread', cpu_ns=30)
        self.proc.set_thread(4301, 'python3', cpu_ns=9)
        self.proc.set_thread(4302, 'pt_main_thread', cpu_ns=7)
        self.recorder.on_tick()

        cpu = self._frames('process_threads_cpu_nanoseconds')
        # The tick's delta goes to the current name; the old frame stays.
        self.assertEqual(cpu['python3/main'], 10)
        self.assertEqual(cpu['pt_main_thread/main'], 20)
        # Both names count as "named like the main thread": split by tid.
        self.assertEqual(cpu['python3/4301'], 9)
        self.assertEqual(cpu['pt_main_thread/4302'], 7)

    def test_exited_thread_keeps_frame_and_totals_never_decrease(self):
        self.proc.set_thread(PID, 'python3', cpu_ns=100, scheduled=1,
                             voluntary=5, involuntary=1)
        self.proc.set_thread(4300, 'tokenizer', cpu_ns=700, scheduled=3,
                             voluntary=20, involuntary=4)
        self.recorder.setup()
        self.recorder.on_tick()
        first_cpu = dict(self._frames('process_threads_cpu_nanoseconds'))
        first_vol = self._counter('process_context_switches', kind='voluntary')

        self.proc.remove_thread(4300)
        self.proc.set_thread(PID, 'python3', cpu_ns=150, scheduled=2,
                             voluntary=6, involuntary=1)
        self.recorder.on_tick()

        cpu = self._frames('process_threads_cpu_nanoseconds')
        self.assertEqual(cpu['tokenizer'], first_cpu['tokenizer'])
        self.assertEqual(cpu['python3/main'], 150)
        self.assertEqual(self._counter('process_context_switches', kind='voluntary'),
                         first_vol + 1)
        self.assertEqual(find_metric(self._export(), 'process_threads',
                                     self.tags).datapoint['value'], 1)

        # A thread reusing the tid with smaller counters adds its own values
        # rather than subtracting: the frame is a running total.
        self.proc.set_thread(4300, 'tokenizer', cpu_ns=50, scheduled=1,
                             voluntary=2, involuntary=0)
        self.recorder.on_tick()
        self.assertGreaterEqual(
            self._frames('process_threads_cpu_nanoseconds')['tokenizer'],
            first_cpu['tokenizer'])

    def test_frame_cap_folds_into_other(self):
        self.proc.set_thread(PID, 'python3', cpu_ns=1)
        for i in range(MAX_FRAMES + 20):
            self.proc.set_thread(6000 + i, f't{i}', cpu_ns=1)
        self.recorder.setup()
        self.recorder.on_tick()

        cpu = self._frames('process_threads_cpu_nanoseconds')
        self.assertEqual(len(cpu), MAX_FRAMES + 1)
        self.assertIn(OTHER_FRAME, cpu)
        self.assertEqual(cpu[OTHER_FRAME], 21)

    def test_no_task_tree_disables_recorder(self):
        recorder = ThreadRecorder(root_pid=1, pid=999999, proc_root=self.root)
        recorder.setup()
        recorder.on_tick()
        self.assertIsNone(find_metric(self._export(), 'process_threads',
                                      {'process.pid': '999999'}))

    def test_live_process(self):
        pid = os.getpid()
        recorder = ThreadRecorder(root_pid=pid, pid=pid)
        recorder.setup()
        recorder.on_tick()
        tags = {'process.pid': str(pid)}
        cpu = find_metric(self._export(), 'process_threads_cpu_nanoseconds', tags)
        self.assertIsNotNone(cpu)
        self.assertGreater(sum(cpu.datapoint['frames'].values()), 0)
        self.assertGreater(find_metric(self._export(), 'process_threads',
                                       tags).datapoint['value'], 0)
        self.assertGreater(
            find_metric(self._export(), 'process_cpu_nanoseconds',
                        dict(tags, mode='user')).datapoint['total'], 0)


if __name__ == '__main__':
    unittest.main()
