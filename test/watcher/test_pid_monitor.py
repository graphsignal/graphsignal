import os
import subprocess
import sys
import time
import unittest

from graphsignal.watcher.pid_monitor import PidMonitor, redact_command_line


class _RecordingListener:
    def __init__(self):
        self.known = []
        self.created = 0
        self.children_created = []
        self.children_terminated = []
        self.terminated = 0

    def on_target_known(self, pid):
        self.known.append(pid)

    def on_target_created(self, args):
        self.created += 1

    def on_child_created(self, pid, args):
        self.children_created.append(pid)

    def on_child_terminated(self, pid):
        self.children_terminated.append(pid)

    def on_target_terminated(self):
        self.terminated += 1


class PidMonitorTest(unittest.TestCase):
    def test_on_target_known_emitted_at_setup_before_polling(self):
        # on_target_known fires synchronously from setup(), so a consumer of a
        # pid-derived artifact is wired up even if the target never polls alive.
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(listener)
        try:
            monitor.setup()
            self.assertEqual(listener.known, [os.getpid()])
        finally:
            monitor.shutdown()

    def test_on_target_created_for_live_target(self):
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(listener)
        try:
            monitor.setup()
            deadline = time.time() + 2.0
            while listener.created == 0 and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(listener.created, 1)
            self.assertEqual(listener.terminated, 0)
        finally:
            monitor.shutdown()

    def test_on_target_known_fires_even_for_dead_target(self):
        # Pick a pid that is (almost certainly) not running.
        dead_pid = 999999
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=dead_pid, poll_interval=0.05)
        monitor.add_listener(listener)
        try:
            monitor.setup()
            # Known fires from setup regardless of liveness; the poll then sees
            # it dead and reports terminated without ever reporting created.
            self.assertEqual(listener.known, [dead_pid])
            time.sleep(0.2)
            self.assertEqual(listener.created, 0)
            self.assertGreaterEqual(listener.terminated, 1)
        finally:
            monitor.shutdown()

    def test_child_created_and_terminated(self):
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(listener)
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])
        try:
            monitor.setup()
            deadline = time.time() + 3.0
            while child.pid not in listener.children_created \
                    and time.time() < deadline:
                time.sleep(0.02)
            self.assertIn(child.pid, listener.children_created)

            child.terminate()
            child.wait()
            deadline = time.time() + 3.0
            while child.pid not in listener.children_terminated \
                    and time.time() < deadline:
                time.sleep(0.02)
            self.assertIn(child.pid, listener.children_terminated)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            monitor.shutdown()

    def test_listener_exception_in_on_target_known_does_not_break_setup(self):
        class _Boom:
            def on_target_known(self, pid):
                raise RuntimeError('boom')

        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(_Boom())
        try:
            monitor.setup()  # must not raise
        finally:
            monitor.shutdown()


class RedactCommandLineTest(unittest.TestCase):
    def test_credential_flag_values_are_replaced(self):
        self.assertEqual(
            redact_command_line(['vllm', 'serve', 'm', '--api-key', 'sk-123', '--port', '8000']),
            'vllm serve m --api-key [redacted] --port 8000')
        self.assertEqual(
            redact_command_line(['x', '--hf-token=hf_abc', '--admin_api_key=k', '-token', 't']),
            'x --hf-token=[redacted] --admin_api_key=[redacted] -token [redacted]')
        self.assertEqual(
            redact_command_line(['x', '--ssl-keyfile-password', 'pw', '--secret', 's']),
            'x --ssl-keyfile-password [redacted] --secret [redacted]')

    def test_other_flags_are_left_alone(self):
        argv = ['vllm', 'serve', 'm', '--max-tokens', '100', '--tokenizer', 'tok',
                '--tokenizer-mode=auto', '--max-new-tokens=5', '--ssl-keyfile', 'k.pem']
        self.assertEqual(redact_command_line(argv), ' '.join(argv))

    def test_flag_without_a_value_is_left_alone(self):
        self.assertEqual(redact_command_line(['x', '--token', '--port', '1']),
                         'x --token --port 1')
        self.assertEqual(redact_command_line(['x', '--token']), 'x --token')



if __name__ == '__main__':
    unittest.main()
