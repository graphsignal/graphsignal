import socket
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

from graphsignal.launchers import command_utils
from graphsignal.launchers.command_utils import (
    bind_listen_socket, extract_host, extract_port, resolve_metrics_host,
    resolve_metrics_port, start_watcher)


class RemovedHelpersTest(unittest.TestCase):
    def test_workload_and_engine_helpers_removed(self):
        self.assertFalse(hasattr(command_utils, 'hash_workload_id'))
        self.assertFalse(hasattr(command_utils, 'engine_version'))


class ExtractHostTest(unittest.TestCase):
    def test_space_form(self):
        self.assertEqual(extract_host(['trtllm-serve', '--host', '0.0.0.0']),
                         '0.0.0.0')

    def test_equals_form(self):
        self.assertEqual(extract_host(['trtllm-serve', '--host=localhost']),
                         'localhost')

    def test_absent_returns_default(self):
        self.assertIsNone(extract_host(['trtllm-serve']))
        self.assertEqual(extract_host(['trtllm-serve'], default='localhost'),
                         'localhost')


class ResolveMetricsHostTest(unittest.TestCase):
    def test_explicit_host_wins(self):
        self.assertEqual(
            resolve_metrics_host('10.0.0.1', ['trtllm-serve', '--host', '0.0.0.0'],
                                 default='localhost'),
            '10.0.0.1')

    def test_falls_back_to_engine_host(self):
        self.assertEqual(
            resolve_metrics_host(None, ['trtllm-serve', '--host', '0.0.0.0'],
                                 default='localhost'),
            '0.0.0.0')

    def test_falls_back_to_default(self):
        self.assertEqual(
            resolve_metrics_host(None, ['trtllm-serve'], default='localhost'),
            'localhost')


class ExtractPortTest(unittest.TestCase):
    def test_space_form(self):
        self.assertEqual(extract_port(['vllm', 'serve', '--port', '8001']), 8001)

    def test_equals_form(self):
        self.assertEqual(extract_port(['vllm', 'serve', '--port=8001']), 8001)

    def test_absent_returns_default(self):
        self.assertIsNone(extract_port(['vllm', 'serve']))
        self.assertEqual(extract_port(['vllm', 'serve'], default=8000), 8000)

    def test_non_int_returns_default(self):
        self.assertEqual(extract_port(['vllm', '--port', 'abc'], default=8000), 8000)

    def test_trailing_port_flag_without_value(self):
        self.assertEqual(extract_port(['vllm', '--port'], default=8000), 8000)


class ResolveMetricsPortTest(unittest.TestCase):
    def test_explicit_port_wins(self):
        # Explicit --metrics-port overrides both the engine --port and default.
        self.assertEqual(
            resolve_metrics_port(9999, ['vllm', '--port', '8001'], default=8000), 9999)

    def test_falls_back_to_engine_port(self):
        self.assertEqual(
            resolve_metrics_port(None, ['vllm', '--port', '8001'], default=8000), 8001)

    def test_falls_back_to_default(self):
        self.assertEqual(
            resolve_metrics_port(None, ['vllm', 'serve'], default=8000), 8000)

    def test_no_default_returns_none(self):
        self.assertIsNone(resolve_metrics_port(None, ['app.py'], default=None))


class StartWatcherTest(unittest.TestCase):
    def test_spawn_args_default(self):
        fake_popen = MagicMock(name='Popen')
        with patch.object(subprocess, 'Popen', return_value=fake_popen) as popen_m:
            result = start_watcher(12345)

        self.assertIs(result, fake_popen)
        popen_m.assert_called_once()
        cmd = popen_m.call_args[0][0]
        kwargs = popen_m.call_args[1]

        self.assertEqual(cmd, [sys.executable,
                               '-m', 'graphsignal.commands.graphsignal_watch',
                               '--pid', '12345'])
        self.assertTrue(kwargs.get('start_new_session'))

    def test_spawn_args_with_metrics_port(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(cmd[5:], ['--metrics-port', '8000'])

    def test_spawn_args_with_metrics_path(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000,
                          metrics_path='/prometheus/metrics')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            cmd[5:],
            ['--metrics-port', '8000', '--metrics-path', '/prometheus/metrics'])

    def test_spawn_args_with_metrics_host(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000, metrics_host='localhost')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            cmd[5:], ['--metrics-port', '8000', '--metrics-host', 'localhost'])

    def test_spawn_args_with_listen_port(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, listen_port=18400)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(cmd[5:], ['--listen-port', '18400'])

    def test_spawn_args_with_listen_host(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, listen_host='0.0.0.0')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(cmd[5:], ['--listen-host', '0.0.0.0'])

    def test_spawn_args_with_all_kwargs(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000,
                          metrics_path='/prometheus/metrics',
                          metrics_host='localhost', listen_host='0.0.0.0',
                          listen_port=18400)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            cmd[5:],
            ['--metrics-port', '8000', '--metrics-path', '/prometheus/metrics',
             '--metrics-host', 'localhost', '--listen-host', '0.0.0.0',
             '--listen-port', '18400'])

    def test_returns_none_on_failure(self):
        with patch.object(subprocess, 'Popen', side_effect=OSError('boom')):
            result = start_watcher(1)
        self.assertIsNone(result)


class ListenSocketTest(unittest.TestCase):
    def test_binds_a_free_port(self):
        sock = bind_listen_socket('127.0.0.1', 0)
        try:
            self.assertGreater(sock.getsockname()[1], 0)
        finally:
            sock.close()

    def test_busy_port_raises_after_timeout(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(('127.0.0.1', 0))
            holder.listen(1)
            started = time.monotonic()
            with self.assertRaises(OSError):
                bind_listen_socket('127.0.0.1', holder.getsockname()[1], timeout=0.3)
            self.assertGreaterEqual(time.monotonic() - started, 0.3)

    def test_waits_for_a_port_that_frees_up(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(('127.0.0.1', 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        threading.Timer(0.3, holder.close).start()
        sock = bind_listen_socket('127.0.0.1', port, timeout=10)
        try:
            self.assertEqual(sock.getsockname()[1], port)
        finally:
            sock.close()

    def test_unbindable_address_raises_at_once(self):
        started = time.monotonic()
        with self.assertRaises(OSError):
            bind_listen_socket('203.0.113.1', 0, timeout=10)
        self.assertLess(time.monotonic() - started, 5)

    def test_watcher_inherits_the_socket(self):
        sock = bind_listen_socket('127.0.0.1', 0)
        try:
            with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
                start_watcher(54321, listen_socket=sock)
            cmd = popen_m.call_args[0][0]
            self.assertEqual(cmd[-2:], ['--listen-fd', str(sock.fileno())])
            self.assertEqual(popen_m.call_args[1]['pass_fds'], (sock.fileno(),))
        finally:
            sock.close()



if __name__ == '__main__':
    unittest.main()
