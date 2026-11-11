import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import graphsignal.watcher
from graphsignal.recorders import log_recorder as log_recorder_mod
from graphsignal.recorders import shm_files
from graphsignal.recorders.log_recorder import (
    LogRecorder, _StreamExtractor, _sweep_stale_shm_dirs)
from test.test_utils import configure_test_watcher


def extract_all(lines):
    """Feed lines through one extractor and return entries incl. flush."""
    extractor = _StreamExtractor()
    entries = []
    for line in lines:
        entries.extend(extractor.feed(1000, line))
    entries.extend(extractor.flush())
    return entries


class ExtractorVllmTest(unittest.TestCase):
    def test_error_with_prefixed_traceback_grouped(self):
        entries = extract_all([
            'INFO 07-16 10:22:33 [api_server.py:1523] vLLM API server version 0.9.0',
            'ERROR 07-16 10:22:33 [core.py:1186] EngineCore encountered a fatal error.',
            'ERROR 07-16 10:22:33 [core.py:1186] Traceback (most recent call last):',
            'ERROR 07-16 10:22:33 [core.py:1186]   File "/x/core.py", line 1, in run',
            'ERROR 07-16 10:22:33 [core.py:1186]     raise ValueError("boom")',
            'ERROR 07-16 10:22:33 [core.py:1186] ValueError: boom',
            'INFO 07-16 10:22:34 [core.py:99] shutting down',
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'], 'EngineCore encountered a fatal error.')
        self.assertIn('Traceback (most recent call last):', entries[0]['exception'])
        self.assertIn('ValueError: boom', entries[0]['exception'])

    def test_critical_line(self):
        entries = extract_all([
            'CRITICAL 07-16 10:22:33 [core.py:1414] vLLM shutdown signal failed to send.',
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'],
                         'vLLM shutdown signal failed to send.')

    def test_consecutive_standalone_errors_stay_separate(self):
        entries = extract_all([
            'ERROR 07-16 10:22:33 [a.py:1] first failure',
            'ERROR 07-16 10:22:33 [a.py:2] second failure',
        ])
        self.assertEqual([e['message'] for e in entries],
                         ['first failure', 'second failure'])

    def test_ansi_colored_error_line(self):
        entries = extract_all([
            '\x1b[31mERROR\x1b[0m \x1b[90m07-16 10:22:33 [a.py:1]\x1b[0m colored boom',
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'], 'colored boom')

    def test_info_and_debug_ignored(self):
        entries = extract_all([
            'INFO 07-16 10:22:33 [a.py:1] all good',
            'DEBUG 07-16 10:22:33 [a.py:2] details',
            'WARNING 07-16 10:22:33 [a.py:3] meh',
        ])
        self.assertEqual(entries, [])


class ExtractorSglangTest(unittest.TestCase):
    def test_scheduler_exception_with_embedded_traceback(self):
        # SGLang's format carries no level name; the scheduler crash is a
        # single logger.error call whose multi-line message arrives as raw
        # unprefixed lines after the first.
        entries = extract_all([
            '[2026-07-16 10:22:33 TP0] Prefill batch. #new-seq: 1',
            '[2026-07-16 10:22:33 TP0] Scheduler hit an exception: Traceback (most recent call last):',
            '  File "/x/scheduler.py", line 4008, in run_scheduler_process',
            '    raise RuntimeError("dead")',
            'RuntimeError: dead',
            '[2026-07-16 10:22:34 TP0] Decode batch. #running-req: 0',
        ])
        self.assertEqual(len(entries), 1)
        self.assertIn('Scheduler hit an exception', entries[0]['message'])
        self.assertIn('RuntimeError: dead', entries[0]['exception'])

    def test_crash_dump_marker(self):
        entries = extract_all([
            "[2026-07-16 10:22:33] Dumping requests before crash. self.crash_dump_folder='/tmp'",
        ])
        self.assertEqual(len(entries), 1)

    def test_info_lines_without_level_ignored(self):
        entries = extract_all([
            '[2026-07-16 10:22:33] server_args=ServerArgs(model_path=...)',
            '[2026-07-16 10:22:33 TP0] Prefill batch. #new-seq: 1',
            '[2026-07-16 10:22:33.123 DP0 TP0] Decode batch.',
        ])
        self.assertEqual(entries, [])

    def test_uvicorn_levelprefix_after_timestamp(self):
        entries = extract_all([
            '[2026-07-16 10:22:33] INFO:     Started server process [12345]',
            '[2026-07-16 10:22:33] ERROR:    Exception in ASGI application',
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'], 'Exception in ASGI application')


class ExtractorTrtllmTest(unittest.TestCase):
    def test_single_letter_error_level(self):
        entries = extract_all([
            '[TRT-LLM] [I] Set logger level to INFO',
            '[TRT-LLM] [E] [runtime] CUDA error 700',
            '[TRT-LLM] [W] [executor] slow path',
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'], '[runtime] CUDA error 700')

    def test_bootstrap_word_level_variant(self):
        entries = extract_all([
            '[TRT-LLM][ERROR] TLLM_LOG_LEVEL_BY_MODULE: unknown level "x"',
        ])
        self.assertEqual(len(entries), 1)


class ExtractorGenericTest(unittest.TestCase):
    def test_uvicorn_error_levelprefix(self):
        entries = extract_all(['ERROR:    Exception in ASGI application'])
        self.assertEqual(len(entries), 1)

    def test_error_colon_prefix(self):
        entries = extract_all(['Error: something failed'])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'], 'Error: something failed')

    def test_argparse_unrecognized_arguments(self):
        # The invalid-flag crash: argparse prints "<prog>: error: <msg>" to
        # stderr (prog may contain a space) with no level name or timestamp.
        entries = extract_all([
            'usage: sglang serve [-h] [--model-path MODEL_PATH]',
            'sglang serve: error: unrecognized arguments: --sdfasdfasdfasdf',
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(
            entries[0]['message'],
            'sglang serve: error: unrecognized arguments: --sdfasdfasdfasdf')

    def test_lowercase_leading_error_and_fatal(self):
        entries = extract_all(['error: bad config', 'fatal: cannot bind port'])
        self.assertEqual([e['message'] for e in entries],
                         ['error: bad config', 'fatal: cannot bind port'])

    def test_error_word_without_colon_not_matched(self):
        # "error"/"errors" not followed by a colon must not trip the CLI rule.
        entries = extract_all([
            'Total errors: 0',
            'measured error rate high',
            'retrying: error recovery in progress',
        ])
        self.assertEqual(entries, [])

    def test_bracket_error_prefix(self):
        entries = extract_all(['[ERROR] disk on fire'])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'], 'disk on fire')

    def test_bare_traceback_block(self):
        entries = extract_all([
            'Traceback (most recent call last):',
            '  File "app.py", line 1, in <module>',
            '    main()',
            'RuntimeError: dead',
            'unrelated trailing line',
        ])
        self.assertEqual(len(entries), 1)
        self.assertIn('RuntimeError: dead', entries[0]['exception'])

    def test_plain_lines_ignored(self):
        entries = extract_all([
            'hello world',
            'Loading model weights...',
            'ERRORS_TOTAL 5',  # no word boundary match
        ])
        self.assertEqual(entries, [])


class LogRecorderDrainTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.shm_dir = tempfile.mkdtemp()
        self.pid = os.getpid()
        self.recorder = LogRecorder(root_pid=self.pid, pid=self.pid, args='x')
        self.shm_patch = patch.object(
            self.recorder, '_shm_dir', return_value=self.shm_dir)
        self.shm_patch.start()

    def tearDown(self):
        self.shm_patch.stop()
        shutil.rmtree(self.shm_dir, ignore_errors=True)
        graphsignal.watcher.shutdown()

    def _write_segment(self, seq, lines, stream='stdout'):
        path = os.path.join(self.shm_dir, f'log_{seq}.jsonl')
        with open(path, 'a') as f:
            for line in lines:
                f.write(json.dumps({'ts': 1000, 'stream': stream, 'line': line}) + '\n')

    def _write_records(self, seq, records):
        path = os.path.join(self.shm_dir, f'log_{seq}.jsonl')
        with open(path, 'a') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

    def _workload_entries(self):
        return [e for e in self.watcher.log_store().export()
                if e['tags'].get('scope.name') == 'workload']

    def test_on_tick_drains(self):
        # on_tick is the production drain path (no private thread). The
        # trailing INFO line lets the extractor finalize the error mid-drain.
        self.recorder._disabled = False
        self._write_segment(0, [
            'ERROR 07-16 10:22:33 [a.py:1] via on_tick',
            'INFO 07-16 10:22:34 [a.py:2] next'])
        self.recorder.on_tick()
        messages = [e['message'] for e in self._workload_entries()]
        self.assertEqual(messages, ['via on_tick'])

    def test_disabled_on_tick_is_noop(self):
        self.recorder._disabled = True
        self._write_segment(0, ['ERROR 07-16 10:22:33 [a.py:1] boom'])
        self.recorder.on_tick()
        self.assertEqual(self._workload_entries(), [])

    def test_exit_code_record_emits_synthetic_error(self):
        self._write_segment(0, ['INFO 07-16 10:22:33 [a.py:1] starting up'])
        self._write_records(0, [{'ts': 2000, 'exit_code': 2}])
        self.recorder._drain()

        entries = self._workload_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry['message'], 'Process exited with non-zero code 2')
        self.assertEqual(entry['level'], 'error')
        self.assertEqual(entry['ts'], 2000)
        self.assertEqual(entry['tags'].get('stream'), 'exit')

    def test_signal_record_emits_synthetic_error(self):
        self._write_records(0, [{'ts': 2000, 'signal': 11}])
        self.recorder._drain()

        entries = self._workload_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['message'], 'Process terminated by signal 11')

    def test_clean_exit_code_record_ignored(self):
        self._write_records(0, [{'ts': 2000, 'exit_code': 0}])
        self.recorder._drain()
        self.assertEqual(self._workload_entries(), [])

    def test_extracts_errors_only_with_tags_and_level(self):
        self._write_segment(0, [
            'INFO 07-16 10:22:33 [a.py:1] fine',
            'ERROR 07-16 10:22:33 [a.py:2] boom happened',
            'INFO 07-16 10:22:34 [a.py:3] fine again',
        ], stream='stderr')
        self.recorder._drain()

        entries = self._workload_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry['message'], 'boom happened')
        self.assertEqual(entry['level'], 'error')
        self.assertEqual(entry['ts'], 1000)
        self.assertEqual(entry['tags'].get('process.pid'), str(self.pid))
        self.assertEqual(entry['tags'].get('stream'), 'stderr')

    def test_incremental_offsets_no_duplicates(self):
        self._write_segment(0, ['ERROR 07-16 10:22:33 [a.py:1] first'])
        self.recorder._drain()
        self._write_segment(0, ['ERROR 07-16 10:22:33 [a.py:2] second',
                                'INFO 07-16 10:22:34 [a.py:3] done'])
        self.recorder._drain()

        messages = [e['message'] for e in self._workload_entries()]
        self.assertEqual(messages, ['first', 'second'])

    def test_rotated_segment_picked_up(self):
        self._write_segment(0, ['ERROR 07-16 10:22:33 [a.py:1] from-seg0'])
        self.recorder._drain()
        self._write_segment(1, ['ERROR 07-16 10:22:33 [a.py:2] from-seg1',
                                'INFO 07-16 10:22:34 [a.py:3] done'])
        self.recorder._drain()

        messages = [e['message'] for e in self._workload_entries()]
        self.assertEqual(messages, ['from-seg0', 'from-seg1'])

    def test_partial_trailing_line_deferred(self):
        path = os.path.join(self.shm_dir, 'log_0.jsonl')
        full = json.dumps({'ts': 1000, 'stream': 'stdout',
                           'line': 'ERROR 07-16 10:22:33 [a.py:1] complete'}) + '\n'
        partial = '{"ts": 1000, "stream": "stdout", "line": "ERROR 07-16 10:2'
        with open(path, 'w') as f:
            f.write(full + partial)
        self.recorder._drain()
        # The complete record is pending in the extractor; the partial record
        # was not consumed. Completing it and draining again yields both.
        with open(path, 'a') as f:
            f.write('2:33 [a.py:2] later"}\n')
        self._write_segment(0, ['INFO 07-16 10:22:34 [a.py:3] done'])
        self.recorder._drain()

        messages = [e['message'] for e in self._workload_entries()]
        self.assertEqual(messages, ['complete', 'later'])

    def test_rate_cap_per_drain(self):
        lines = ['ERROR 07-16 10:22:33 [a.py:%d] err-%03d' % (i, i)
                 for i in range(60)]
        self._write_segment(0, lines)
        self.recorder._drain()

        entries = self._workload_entries()
        self.assertEqual(len(entries), log_recorder_mod.MAX_ENTRIES_PER_DRAIN)

    def test_dropped_marker_records_ignored(self):
        path = os.path.join(self.shm_dir, 'log_0.jsonl')
        with open(path, 'w') as f:
            f.write(json.dumps({'ts': 1000, 'dropped': 5}) + '\n')
            f.write(json.dumps({'ts': 1001, 'stream': 'stdout',
                                'line': 'ERROR 07-16 10:22:33 [a.py:1] real'}) + '\n')
            f.write(json.dumps({'ts': 1002, 'stream': 'stdout',
                                'line': 'INFO 07-16 10:22:34 [a.py:2] done'}) + '\n')
        self.recorder._drain()
        messages = [e['message'] for e in self._workload_entries()]
        self.assertEqual(messages, ['real'])

    def test_finalize_flushes_pending_trailing_error(self):
        # A trailing error line with nothing after it stays pending in the
        # extractor; finalize() (terminated path) must drain + flush it.
        self._write_segment(0, ['ERROR 07-16 10:22:33 [a.py:1] dying words'])
        self.recorder._disabled = False
        self.recorder.finalize()

        messages = [e['message'] for e in self._workload_entries()]
        self.assertEqual(messages, ['dying words'])

    def test_shutdown_cleans_dir_without_recording(self):
        # shutdown() is cleanup-only (the terminated path drains via
        # finalize()); it must remove the dir and not record anything itself.
        self._write_segment(0, ['ERROR 07-16 10:22:33 [a.py:1] leftover'])
        self.recorder._disabled = False
        self.recorder.shutdown()

        self.assertEqual(self._workload_entries(), [])
        self.assertFalse(os.path.isdir(self.shm_dir))


class LogRecorderUntrustedDirTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.base = tempfile.mkdtemp()
        self.shm_dir = os.path.join(self.base, 'graphsignal_log_1')
        self.recorder = LogRecorder(root_pid=os.getpid(), pid=os.getpid(), args='x')
        self.recorder._disabled = False
        self.shm_patch = patch.object(
            self.recorder, '_shm_dir', return_value=self.shm_dir)
        self.shm_patch.start()

    def tearDown(self):
        self.shm_patch.stop()
        shutil.rmtree(self.base, ignore_errors=True)
        graphsignal.watcher.shutdown()

    def _segment(self, path, message='boom'):
        with open(path, 'w') as f:
            f.write(json.dumps({'ts': 1000, 'stream': 'stderr',
                                'line': f'ERROR: {message}'}) + '\n')

    def _messages(self):
        return [e['message'] for e in self.watcher.log_store().export()
                if e['tags'].get('scope.name') == 'workload']

    def test_symlinked_segment_and_fifo_are_skipped(self):
        os.makedirs(self.shm_dir)
        outside = os.path.join(self.base, 'outside.jsonl')
        self._segment(outside, 'outside')
        os.symlink(outside, os.path.join(self.shm_dir, 'log_0.jsonl'))
        os.mkfifo(os.path.join(self.shm_dir, 'log_1.jsonl'))
        self._segment(os.path.join(self.shm_dir, 'log_2.jsonl'), 'real')

        thread = threading.Thread(target=self.recorder.finalize, daemon=True)
        thread.start()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self._messages(), ['real'])

    def test_symlinked_dir_is_not_read(self):
        elsewhere = os.path.join(self.base, 'elsewhere')
        os.makedirs(elsewhere)
        self._segment(os.path.join(elsewhere, 'log_0.jsonl'))
        os.symlink(elsewhere, self.shm_dir)

        self.recorder.finalize()
        self.assertEqual(self._messages(), [])

    def test_dir_of_another_user_is_not_read(self):
        os.makedirs(self.shm_dir)
        self._segment(os.path.join(self.shm_dir, 'log_0.jsonl'))
        with patch.object(shm_files.os, 'geteuid', return_value=os.geteuid() + 1):
            self.recorder.finalize()
        self.assertEqual(self._messages(), [])

    def test_record_longer_than_a_pass_is_skipped_not_reread(self):
        os.makedirs(self.shm_dir)
        path = os.path.join(self.shm_dir, 'log_0.jsonl')
        with open(path, 'w') as f:
            f.write('x' * 200)
        with patch.object(log_recorder_mod, 'MAX_BYTES_PER_DRAIN', 64):
            for _ in range(4):
                self.recorder.on_tick()
        self.assertEqual(self.recorder._offsets[path], 192)
        with open(path, 'a') as f:
            f.write('\n' + json.dumps({'ts': 1000, 'stream': 'stderr',
                                       'line': 'ERROR: after'}) + '\n')
        self.recorder.finalize()
        self.assertEqual(self._messages(), ['after'])



class SweepStaleShmDirsTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.base_patch = patch.object(log_recorder_mod, '_SHM_BASE', self.base)
        self.base_patch.start()

    def tearDown(self):
        self.base_patch.stop()
        shutil.rmtree(self.base, ignore_errors=True)

    def test_sweeps_dead_pid_dirs_only(self):
        proc = subprocess.Popen(['true'])
        proc.wait()
        dead_pid = proc.pid
        alive_pid = os.getpid()

        dead_dir = os.path.join(self.base, f'graphsignal_log_{dead_pid}')
        alive_dir = os.path.join(self.base, f'graphsignal_log_{alive_pid}')
        metrics_dir = os.path.join(self.base, f'graphsignal_{dead_pid}')
        junk_dir = os.path.join(self.base, 'graphsignal_log_notapid')
        for d in (dead_dir, alive_dir, metrics_dir, junk_dir):
            os.makedirs(d)

        _sweep_stale_shm_dirs()

        self.assertFalse(os.path.isdir(dead_dir))
        self.assertTrue(os.path.isdir(alive_dir))
        self.assertTrue(os.path.isdir(metrics_dir))
        self.assertTrue(os.path.isdir(junk_dir))


if __name__ == '__main__':
    unittest.main()
