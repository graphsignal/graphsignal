import os
import random
import shutil
import tempfile
import time
import unittest

import graphsignal.watcher
from graphsignal.recorders.process_recorder import ProcessRecorder
from test.test_utils import configure_test_watcher, find_metric

mem = []


class ProcessRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def test_record(self):
        pid = os.getpid()
        recorder = ProcessRecorder(root_pid=pid, pid=pid,
                                   args='python -m unittest')
        recorder.setup()

        time.sleep(0.2)
        for _ in range(100000):
            random.random()
        global mem
        mem = [1] * 100000

        recorder.on_tick()

        exported = self.watcher.metric_store().export()
        pid_tag = {'process.pid': str(pid)}

        rss = find_metric(exported, 'process_memory_usage_bytes', pid_tag)
        self.assertEqual(rss.type, 'gauge')
        self.assertGreater(rss.datapoint['value'], 0)
        vms = find_metric(exported, 'process_memory_virtual_bytes', pid_tag)
        self.assertEqual(vms.type, 'gauge')
        self.assertGreater(vms.datapoint['value'], 0)

        resources = self.watcher.resource_store().export()
        process_resources = [r for r in resources if r['kind'] == 'process']
        self.assertEqual(len(process_resources), 1)
        resource = process_resources[0]
        self.assertEqual(resource['tags'].get('process.pid'), str(pid))
        # The root's own resource links to itself.
        self.assertEqual(resource['tags'].get('process.root_pid'), str(pid))
        self.assertIn('process.command_line', resource['attributes'])
        # When observing self, runtime attrs are also reported.
        self.assertIn('runtime.name', resource['attributes'])
        self.assertIn('runtime.version', resource['attributes'])

        allowed = find_metric(exported, 'process_cpu_allowed', pid_tag)
        self.assertEqual(allowed.type, 'gauge')
        self.assertGreater(allowed.datapoint['value'], 0)


class ProcessRecorderCgroupTest(unittest.TestCase):
    """cpu.stat / cpu.max of the process's cgroup, from fake /proc and
    /sys/fs/cgroup trees."""

    def setUp(self):
        self.watcher = configure_test_watcher()
        self.root = tempfile.mkdtemp()
        self.proc_root = os.path.join(self.root, 'proc')
        self.cgroup_root = os.path.join(self.root, 'cgroup')
        self.pid = os.getpid()
        os.makedirs(os.path.join(self.proc_root, str(self.pid)))
        os.makedirs(self.cgroup_root)
        self.tags = {'process.pid': str(self.pid)}

    def tearDown(self):
        graphsignal.watcher.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, *parts, text):
        path = os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)

    def _set_proc_cgroup(self, text):
        self._write(self.proc_root, str(self.pid), 'cgroup', text=text)

    def _recorder(self, pid=None, root_pid=None):
        pid = pid or self.pid
        recorder = ProcessRecorder(
            root_pid=root_pid or pid, pid=pid, args='python app.py',
            proc_root=self.proc_root, cgroup_root=self.cgroup_root)
        recorder.setup()
        return recorder

    def _metric(self, name, tags=None):
        return find_metric(self.watcher.metric_store().export(), name,
                           tags or self.tags)

    def test_cgroup_v2_throttling_and_quota(self):
        self._set_proc_cgroup('0::/kubepods/pod1/ctr1\n')
        self._write(self.cgroup_root, 'kubepods/pod1/ctr1/cpu.stat', text=(
            'usage_usec 1000\nuser_usec 800\nsystem_usec 200\n'
            'nr_periods 500\nnr_throttled 40\nthrottled_usec 123456\n'
            'nr_bursts 0\nburst_usec 0\n'))
        self._write(self.cgroup_root, 'kubepods/pod1/ctr1/cpu.max',
                    text='400000 100000\n')
        self._recorder().on_tick()

        periods = self._metric('process_cpu_throttled_periods')
        self.assertEqual(periods.type, 'counter')
        self.assertEqual(periods.datapoint['total'], 40)
        self.assertEqual(self._metric('process_cpu_throttled_nanoseconds')
                         .datapoint['total'], 123456000)
        quota = self._metric('process_cpu_quota')
        self.assertEqual(quota.type, 'gauge')
        self.assertEqual(quota.datapoint['value'], 4.0)

    def test_cgroup_v2_unlimited_quota_emits_no_quota(self):
        self._set_proc_cgroup('0::/\n')
        self._write(self.cgroup_root, 'cpu.stat', text=(
            'nr_periods 0\nnr_throttled 0\nthrottled_usec 0\n'))
        self._write(self.cgroup_root, 'cpu.max', text='max 100000\n')
        self._recorder().on_tick()

        # Measured zero throttling is reported as 0, not left out.
        self.assertEqual(self._metric('process_cpu_throttled_periods')
                         .datapoint['total'], 0)
        self.assertEqual(self._metric('process_cpu_throttled_nanoseconds')
                         .datapoint['total'], 0)
        self.assertIsNone(self._metric('process_cpu_quota'))

    def test_cgroup_v1_layout(self):
        self._set_proc_cgroup(
            '12:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n'
            '1:name=systemd:/docker/abc\n')
        base = os.path.join(self.cgroup_root, 'cpu,cpuacct', 'docker', 'abc')
        self._write(base, 'cpu.stat', text=(
            'nr_periods 100\nnr_throttled 7\nthrottled_time 987654321\n'))
        self._write(base, 'cpu.cfs_quota_us', text='150000\n')
        self._write(base, 'cpu.cfs_period_us', text='100000\n')
        self._recorder().on_tick()

        self.assertEqual(self._metric('process_cpu_throttled_periods')
                         .datapoint['total'], 7)
        self.assertEqual(self._metric('process_cpu_throttled_nanoseconds')
                         .datapoint['total'], 987654321)
        self.assertEqual(self._metric('process_cpu_quota').datapoint['value'], 1.5)

    def test_cgroup_v1_wins_on_hybrid_host(self):
        self._set_proc_cgroup('3:cpu,cpuacct:/docker/abc\n0::/user.slice\n')
        v1 = os.path.join(self.cgroup_root, 'cpu', 'docker', 'abc')
        self._write(v1, 'cpu.stat', text='nr_periods 1\nnr_throttled 2\nthrottled_time 3\n')
        self._write(v1, 'cpu.cfs_quota_us', text='-1\n')
        self._write(v1, 'cpu.cfs_period_us', text='100000\n')
        self._write(self.cgroup_root, 'user.slice', 'cpu.stat',
                    text='nr_periods 9\nnr_throttled 9\nthrottled_usec 9\n')
        self._recorder().on_tick()

        self.assertEqual(self._metric('process_cpu_throttled_periods')
                         .datapoint['total'], 2)
        self.assertEqual(self._metric('process_cpu_throttled_nanoseconds')
                         .datapoint['total'], 3)
        self.assertIsNone(self._metric('process_cpu_quota'))

    def test_missing_cgroup_files_emit_nothing(self):
        self._set_proc_cgroup('0::/nowhere\n')
        self._recorder().on_tick()
        self.assertIsNone(self._metric('process_cpu_throttled_periods'))
        self.assertIsNone(self._metric('process_cpu_throttled_nanoseconds'))
        self.assertIsNone(self._metric('process_cpu_quota'))

        # The directory exists but has no cpu accounting at all.
        self._set_proc_cgroup('0::/\n')
        self._recorder().on_tick()
        self.assertIsNone(self._metric('process_cpu_throttled_periods'))
        self.assertIsNone(self._metric('process_cpu_quota'))

    def test_child_recorder_does_not_report_cgroup(self):
        self._set_proc_cgroup('0::/\n')
        self._write(self.cgroup_root, 'cpu.stat', text='nr_throttled 5\nthrottled_usec 5\n')
        self._write(self.cgroup_root, 'cpu.max', text='100000 100000\n')
        # Observed as a child of another root process.
        self._recorder(root_pid=1).on_tick()
        self.assertIsNone(self._metric('process_cpu_throttled_periods'))
        self.assertIsNone(self._metric('process_cpu_quota'))
        # Its own per-process metrics are still there.
        self.assertIsNotNone(self._metric('process_memory_usage_bytes'))


if __name__ == '__main__':
    unittest.main()
