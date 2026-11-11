import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import graphsignal.watcher
from graphsignal.recorders import shm_files
from graphsignal.recorders import shm_recorder as shm_recorder_mod
from graphsignal.recorders.shm_recorder import ShmRecorder
from test.test_utils import configure_test_watcher, find_metric


class ShmRecorderImportTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.base_dir = tempfile.mkdtemp()
        self.pid = os.getpid()
        self.shm_dir = os.path.join(self.base_dir, f'graphsignal_{self.pid}')
        os.makedirs(self.shm_dir)
        self.recorder = ShmRecorder(
            root_pid=self.pid, pid=self.pid, args='python app.py',
            shm_base_dir=self.base_dir)
        self.recorder.setup()

    def tearDown(self):
        graphsignal.watcher.shutdown()
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def _write_file(self, data, name='cupti.json'):
        path = os.path.join(self.shm_dir, name)
        with open(path, 'w') as f:
            f.write(data if isinstance(data, str) else json.dumps(data))
        return path

    def _payload(self, metrics=None, log=None, start_ts=100, write_ts=200,
                 context=None, version=1):
        payload = {
            'version': version,
            'pid': self.pid,
            'start_ts': start_ts,
            'write_ts': write_ts,
            'metrics': metrics or [],
        }
        if context is not None:
            payload['context'] = context
        if log is not None:
            payload['log'] = log
        return payload

    def test_imports_gauge_counter_histogram_and_profile(self):
        self._write_file(self._payload(
            context={'rank': '0'},
            metrics=[
                {'name': 'x_gauge', 'type': 'gauge', 'value': 42.5},
                {'name': 'cuda_memcpy_bytes', 'type': 'counter',
                 'tags': {'kind': 'host_to_device'}, 'value': 1024},
                {'name': 'user_hist', 'type': 'histogram',
                 'tags': {'op': 'o1'},
                 'bins': [0, 10, 20, 30], 'counts': [1, 4, 3, 2]},
                {'name': 'exact_hist', 'type': 'histogram',
                 'tags': {'op': 'o2'},
                 'bins': [8], 'counts': [2],
                 'count': 2, 'sum': 19, 'min': 9, 'max': 10},
                {'name': 'cuda_kernels_nanoseconds', 'type': 'profile',
                 'tags': {}, 'frames': {'kern_a': [123, 4], 'kern_b': [7, 1]}},
            ]))

        self.recorder.on_tick()
        exported = self.watcher.metric_store().export()
        base_tags = {'process.pid': str(self.pid), 'process.rank': '0'}

        gauge = find_metric(exported, 'x_gauge', base_tags)
        self.assertIsNotNone(gauge)
        self.assertEqual(gauge.type, 'gauge')
        self.assertEqual(gauge.datapoint, {'ts': 200, 'value': 42.5})

        counter = find_metric(exported, 'cuda_memcpy_bytes',
                              {**base_tags, 'kind': 'host_to_device'})
        self.assertIsNotNone(counter)
        self.assertEqual(counter.type, 'counter')
        self.assertEqual(counter.datapoint, {'ts': 200, 'total': 1024})

        histogram = find_metric(exported, 'user_hist',
                                {**base_tags, 'op': 'o1'})
        self.assertIsNotNone(histogram)
        self.assertEqual(histogram.type, 'histogram')
        # The raw bins/counts pass through; the payload computes quantiles.
        self.assertEqual(histogram.datapoint, {
            'ts': 200, 'bins': [0, 10, 20, 30], 'counts': [1, 4, 3, 2]})

        exact = find_metric(exported, 'exact_hist', {**base_tags, 'op': 'o2'})
        self.assertIsNotNone(exact)
        # A writer that carries the exact aggregates gets them stored as is.
        self.assertEqual(exact.datapoint, {
            'ts': 200, 'bins': [8], 'counts': [2],
            'count': 2, 'sum': 19, 'min': 9, 'max': 10})

        profile = find_metric(exported, 'cuda_kernels_nanoseconds', base_tags)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.type, 'profile')
        # Frames pass through as string names; ids are assigned at upload.
        # A frame arrives as [value, samples] and is split into the two maps.
        self.assertEqual(profile.datapoint, {
            'ts': 200, 'frames': {'kern_a': 123, 'kern_b': 7},
            'samples': {'kern_a': 4, 'kern_b': 1}})

    def test_histogram_entry_with_neither_half_skipped_as_malformed(self):
        self._write_file(self._payload(metrics=[
            {'name': 'nothing', 'type': 'histogram'},
            {'name': 'counts_only', 'type': 'histogram', 'counts': [1, 2]},
            {'name': 'mismatched', 'type': 'histogram',
             'bins': [0, 10], 'counts': [1]},
            {'name': 'ok_hist', 'type': 'histogram',
             'bins': [0, 10], 'counts': [1, 2]},
            # The writer omits bins entirely for an unrecorded instrument, and
            # the exact totals alone are a whole histogram.
            {'name': 'totals_only', 'type': 'histogram',
             'count': 3, 'sum': 24, 'min': 4, 'max': 12},
        ]))

        self.recorder.on_tick()  # must not raise
        exported = self.watcher.metric_store().export()
        self.assertIsNone(find_metric(exported, 'nothing'))
        self.assertIsNone(find_metric(exported, 'counts_only'))
        self.assertIsNone(find_metric(exported, 'mismatched'))
        ok = find_metric(exported, 'ok_hist')
        self.assertIsNotNone(ok)
        self.assertEqual(ok.datapoint['bins'], [0, 10])
        self.assertEqual(ok.datapoint['counts'], [1, 2])
        totals = find_metric(exported, 'totals_only')
        self.assertIsNotNone(totals)
        self.assertEqual(totals.datapoint, {
            'ts': 200, 'count': 3, 'sum': 24, 'min': 4, 'max': 12})

    def test_profile_entry_without_frames_skipped_as_malformed(self):
        self._write_file(self._payload(metrics=[
            {'name': 'no_frames', 'type': 'profile'},
            {'name': 'ok_profile', 'type': 'profile',
             'frames': {'f1': [5, 2]}},
        ]))

        self.recorder.on_tick()  # must not raise
        exported = self.watcher.metric_store().export()
        self.assertIsNone(find_metric(exported, 'no_frames'))
        ok = find_metric(exported, 'ok_profile')
        self.assertIsNotNone(ok)
        self.assertEqual(ok.datapoint['frames'], {'f1': 5})
        self.assertEqual(ok.datapoint['samples'], {'f1': 2})

    def test_context_mapped_to_tags(self):
        self._write_file(self._payload(
            context={'rank': '3', 'local_rank': '1', 'world_size': '8',
                     'slurm_job_id': 'j1', 'master_addr': '10.0.0.1',
                     'unknown_key': 'ignored', 'master_port': ''},
            metrics=[{'name': 'x_gauge', 'type': 'gauge', 'value': 1.0}]))

        self.recorder.on_tick()
        metric = find_metric(self.watcher.metric_store().export(), 'x_gauge')
        self.assertEqual(metric.tags, {
            'process.pid': str(self.pid),
            'process.rank': '3',
            'process.local_rank': '1',
            'process.world_size': '8',
            'slurm.job_id': 'j1',
            'distributed.master_addr': '10.0.0.1',
        })

    def test_rewritten_file_overwrites_cumulative_datapoint(self):
        self._write_file(self._payload(
            write_ts=200,
            metrics=[{'name': 'c1', 'type': 'counter', 'value': 100}]))
        self.recorder.on_tick()

        self._write_file(self._payload(
            write_ts=300,
            metrics=[{'name': 'c1', 'type': 'counter', 'value': 250}]))
        self.recorder.on_tick()

        metric = find_metric(self.watcher.metric_store().export(), 'c1')
        self.assertEqual(metric.datapoint, {'ts': 300, 'total': 250})

    def test_unsupported_version_skipped(self):
        self._write_file(self._payload(
            version=2,
            metrics=[{'name': 'x_gauge', 'type': 'gauge', 'value': 1.0}]))
        self.recorder.on_tick()
        self.assertIsNone(
            find_metric(self.watcher.metric_store().export(), 'x_gauge'))

    def test_corrupt_json_skipped(self):
        self._write_file('{not json', name='corrupt.json')
        self._write_file(self._payload(
            metrics=[{'name': 'x_gauge', 'type': 'gauge', 'value': 1.0}]))
        self.recorder.on_tick()  # must not raise
        self.assertIsNotNone(
            find_metric(self.watcher.metric_store().export(), 'x_gauge'))

    def test_malformed_metric_entry_skipped(self):
        self._write_file(self._payload(metrics=[
            {'name': 'no_value', 'type': 'gauge'},
            'not-a-dict',
            {'type': 'gauge', 'value': 1.0},
            {'name': 'ok_gauge', 'type': 'gauge', 'value': 2.0},
        ]))
        self.recorder.on_tick()
        exported = self.watcher.metric_store().export()
        self.assertIsNone(find_metric(exported, 'no_value'))
        self.assertIsNotNone(find_metric(exported, 'ok_gauge'))

    def test_start_ts_change_tolerated(self):
        self._write_file(self._payload(
            start_ts=100, write_ts=200,
            metrics=[{'name': 'c1', 'type': 'counter', 'value': 10}]))
        self.recorder.on_tick()

        # Writer restart: new start_ts; importing continues.
        self._write_file(self._payload(
            start_ts=999, write_ts=300,
            metrics=[{'name': 'c1', 'type': 'counter', 'value': 5}]))
        self.recorder.on_tick()

        metric = find_metric(self.watcher.metric_store().export(), 'c1')
        self.assertEqual(metric.datapoint, {'ts': 300, 'total': 5})

    def test_missing_dir_is_noop(self):
        shutil.rmtree(self.shm_dir)
        self.recorder.on_tick()  # must not raise
        self.assertIsNone(
            find_metric(self.watcher.metric_store().export(), 'x_gauge'))

    def test_log_entries_reemitted_at_debug_and_deduped_by_ts(self):
        self._write_file(self._payload(
            log=[{'ts': 1000, 'msg': 'hello from workload'}]))

        with self.assertLogs('graphsignal', level='DEBUG') as cm:
            self.recorder.on_tick()
            self.recorder.on_tick()
        hits = [m for m in cm.output if 'hello from workload' in m]
        self.assertEqual(len(hits), 1)

        # A newer log entry is emitted; the old one stays deduped.
        self._write_file(self._payload(
            log=[{'ts': 1000, 'msg': 'hello from workload'},
                 {'ts': 2000, 'msg': 'second line'}]))
        with self.assertLogs('graphsignal', level='DEBUG') as cm:
            self.recorder.on_tick()
        self.assertEqual(
            len([m for m in cm.output if 'hello from workload' in m]), 0)
        self.assertEqual(len([m for m in cm.output if 'second line' in m]), 1)

    def test_shutdown_removes_own_dir(self):
        self.assertTrue(os.path.isdir(self.shm_dir))
        self.recorder.shutdown()
        self.assertFalse(os.path.isdir(self.shm_dir))


class ShmRecorderUntrustedDirTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.base_dir = tempfile.mkdtemp()
        self.pid = os.getpid()
        self.shm_dir = os.path.join(self.base_dir, f'graphsignal_{self.pid}')
        self.recorder = ShmRecorder(
            root_pid=self.pid, pid=self.pid, shm_base_dir=self.base_dir)

    def tearDown(self):
        graphsignal.watcher.shutdown()
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def _payload(self, name='g1'):
        return json.dumps({
            'version': 1, 'pid': self.pid, 'start_ts': 1, 'write_ts': 2,
            'metrics': [{'name': name, 'type': 'gauge', 'value': 1}]})

    def _names(self):
        return {m.name for m in self.watcher.metric_store().export()}

    def test_fifo_does_not_block_the_tick(self):
        os.makedirs(self.shm_dir)
        os.mkfifo(os.path.join(self.shm_dir, 'a.json'))
        with open(os.path.join(self.shm_dir, 'b.json'), 'w') as f:
            f.write(self._payload())

        thread = threading.Thread(target=self.recorder.on_tick, daemon=True)
        thread.start()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self._names(), {'g1'})

    def test_symlinked_file_is_not_read(self):
        os.makedirs(self.shm_dir)
        outside = os.path.join(self.base_dir, 'outside.json')
        with open(outside, 'w') as f:
            f.write(self._payload('outside'))
        os.symlink(outside, os.path.join(self.shm_dir, 'a.json'))

        self.recorder.on_tick()
        self.assertEqual(self._names(), set())

    def test_symlinked_dir_is_not_read(self):
        elsewhere = os.path.join(self.base_dir, 'elsewhere')
        os.makedirs(elsewhere)
        with open(os.path.join(elsewhere, 'a.json'), 'w') as f:
            f.write(self._payload())
        os.symlink(elsewhere, self.shm_dir)

        self.recorder.on_tick()
        self.assertEqual(self._names(), set())

    def test_dir_of_another_user_is_not_read(self):
        os.makedirs(self.shm_dir)
        with open(os.path.join(self.shm_dir, 'a.json'), 'w') as f:
            f.write(self._payload())
        with patch.object(shm_files.os, 'geteuid', return_value=os.geteuid() + 1), \
                patch.object(shm_files, '_process_uid', return_value=os.geteuid() + 1):
            self.recorder.on_tick()
        self.assertEqual(self._names(), set())

    @unittest.skipUnless(os.path.isdir('/proc/self'), 'needs /proc')
    def test_dir_of_the_observed_process_user_is_read(self):
        os.makedirs(self.shm_dir)
        with open(os.path.join(self.shm_dir, 'a.json'), 'w') as f:
            f.write(self._payload())
        with patch.object(shm_files.os, 'geteuid', return_value=os.geteuid() + 1):
            self.recorder.on_tick()
        self.assertEqual(self._names(), {'g1'})

    def test_oversized_file_is_skipped(self):
        os.makedirs(self.shm_dir)
        with open(os.path.join(self.shm_dir, 'a.json'), 'w') as f:
            f.write(self._payload())
        with patch.object(shm_recorder_mod, 'MAX_FILE_BYTES', 16):
            self.recorder.on_tick()
        self.assertEqual(self._names(), set())



class SweepStaleDirsTest(unittest.TestCase):
    def setUp(self):
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_sweeps_dead_pid_dirs_only(self):
        proc = subprocess.Popen(['true'])
        proc.wait()
        dead_pid = proc.pid
        alive_pid = os.getpid()

        dead_dir = os.path.join(self.base_dir, f'graphsignal_{dead_pid}')
        alive_dir = os.path.join(self.base_dir, f'graphsignal_{alive_pid}')
        log_dir = os.path.join(self.base_dir, f'graphsignal_log_{dead_pid}')
        junk_dir = os.path.join(self.base_dir, 'graphsignal_notapid')
        for d in (dead_dir, alive_dir, log_dir, junk_dir):
            os.makedirs(d)

        recorder = ShmRecorder(root_pid=alive_pid, pid=alive_pid,
                               shm_base_dir=self.base_dir)
        recorder.setup()

        self.assertFalse(os.path.isdir(dead_dir))
        self.assertTrue(os.path.isdir(alive_dir))
        self.assertTrue(os.path.isdir(log_dir))
        self.assertTrue(os.path.isdir(junk_dir))

    def test_setup_sweeps_only_for_root_recorder(self):
        proc = subprocess.Popen(['true'])
        proc.wait()
        dead_dir = os.path.join(self.base_dir, f'graphsignal_{proc.pid}')
        os.makedirs(dead_dir)

        # Child recorder (pid != root_pid): no sweep at setup.
        child = ShmRecorder(root_pid=os.getpid() + 1, pid=os.getpid(),
                            shm_base_dir=self.base_dir)
        child.setup()
        self.assertTrue(os.path.isdir(dead_dir))

        # Root recorder: sweeps at setup.
        root = ShmRecorder(root_pid=os.getpid(), pid=os.getpid(),
                           shm_base_dir=self.base_dir)
        root.setup()
        self.assertFalse(os.path.isdir(dead_dir))


if __name__ == '__main__':
    unittest.main()
