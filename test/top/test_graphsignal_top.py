import io
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from graphsignal.commands import graphsignal_top as cli
from graphsignal.top.client import SignalsClient, SignalsUnavailable
from graphsignal.top.state import TopState
from test.top import fixtures as fx
from test.test_utils import free_port


class SequenceServer:
    """Serves the given payloads to consecutive GET /signals requests, then
    repeats the last one. A 404 for any other path; a 500 for a non-dict."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.port = free_port()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/signals':
                    self.send_response(404)
                    self.end_headers()
                    return
                index = min(server.calls, len(server.payloads) - 1)
                server.calls += 1
                payload = server.payloads[index]
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(('127.0.0.1', self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self):
        opts = cli.parse_args([])
        self.assertEqual(opts['host'], '127.0.0.1')
        self.assertEqual(opts['port'], 18259)
        self.assertEqual(opts['interval'], 1.0)
        self.assertEqual(opts['window'], 5.0)
        self.assertFalse(opts['once'])
        self.assertTrue(opts['color'])

    def test_all_flags(self):
        opts = cli.parse_args(['--host', 'gpu-box', '--port', '1234', '--interval', '2',
                               '--window', '10', '--once', '--no-color'])
        self.assertEqual(opts['host'], 'gpu-box')
        self.assertEqual(opts['port'], 1234)
        self.assertEqual(opts['interval'], 2.0)
        self.assertEqual(opts['window'], 10.0)
        self.assertTrue(opts['once'])
        self.assertFalse(opts['color'])

    def test_equals_form(self):
        opts = cli.parse_args(['--host=h', '--port=99'])
        self.assertEqual((opts['host'], opts['port']), ('h', 99))

    def test_errors(self):
        for argv in (['--port'], ['--port', 'x'], ['--port', '0'], ['--interval', '0'],
                     ['bogus'], ['--host']):
            with self.assertRaises(cli.UsageError, msg=argv):
                cli.parse_args(argv)

    def test_help_and_version_exit_zero(self):
        from graphsignal.version import __version__
        for argv, expected in ((['-h'], 'Usage:'), (['--help'], 'Usage:'),
                               (['--version'], f'graphsignal-top {__version__}'),
                               (['-V'], f'graphsignal-top {__version__}')):
            with patch('sys.stdout', new=io.StringIO()) as out:
                with self.assertRaises(SystemExit) as cm:
                    cli.main(argv)
            self.assertEqual(cm.exception.code, 0, argv)
            self.assertIn(expected, out.getvalue(), argv)

    def test_usage_error_exits_two(self):
        with patch('sys.stderr', new=io.StringIO()) as err:
            with self.assertRaises(SystemExit) as cm:
                cli.main(['--nope'])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn('unknown argument', err.getvalue())


class ClientTest(unittest.TestCase):
    def test_fetch_and_endpoint(self):
        with SequenceServer([fx.build_payload(1)]) as server:
            client = SignalsClient('127.0.0.1', server.port)
            self.assertEqual(client.endpoint, f'127.0.0.1:{server.port}')
            self.assertEqual(client.url, f'http://127.0.0.1:{server.port}/signals')
            payload = client.fetch()
        self.assertEqual(payload['profiler']['version'], '1.1.0')

    def test_connection_refused(self):
        client = SignalsClient('127.0.0.1', free_port(), timeout=0.5)
        with self.assertRaises(SignalsUnavailable) as cm:
            client.fetch()
        self.assertIn('connection refused', str(cm.exception))

    def test_bad_json(self):
        with SequenceServer([b'not json']) as server:
            with self.assertRaises(SignalsUnavailable) as cm:
                SignalsClient('127.0.0.1', server.port).fetch()
        self.assertEqual(str(cm.exception), 'invalid JSON')

    def test_ipv6_endpoint_is_bracketed(self):
        self.assertEqual(SignalsClient('::1', 1).endpoint, '[::1]:1')


class OnceTest(unittest.TestCase):
    def test_once_prints_a_frame_with_rates(self):
        with SequenceServer([fx.build_payload(10), fx.build_payload(11)]) as server:
            client = SignalsClient('127.0.0.1', server.port)
            state = TopState(window=5.0, endpoint=client.endpoint)
            out = io.StringIO()
            with patch.object(cli.shutil, 'get_terminal_size',
                              return_value=type('S', (), {'columns': 120, 'lines': 40})()):
                code = cli.render_once(client, state, interval=0.01, out=out)
        self.assertEqual(code, 0)
        self.assertEqual(server.calls, 2)
        text = out.getvalue()
        self.assertIn('graphsignal-top', text)
        self.assertIn('flash attention prefill', text)
        self.assertIn('gen tok/s      1.84k', text)
        self.assertIn('31.4', text)
        self.assertNotIn('q quit', text)

    def test_once_unavailable_exits_one(self):
        client = SignalsClient('127.0.0.1', free_port(), timeout=0.5)
        state = TopState()
        with patch('sys.stderr', new=io.StringIO()) as err:
            code = cli.render_once(client, state, interval=0.01, out=io.StringIO())
        self.assertEqual(code, 1)
        self.assertIn('connection refused', err.getvalue())

    def test_main_once(self):
        with SequenceServer([fx.build_payload(10), fx.build_payload(11)]) as server:
            with patch('sys.stdout', new=io.StringIO()) as out:
                with self.assertRaises(SystemExit) as cm:
                    cli.main(['--port', str(server.port), '--once', '--interval', '0.01'])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn('KERNELS', out.getvalue())

    def test_interactive_refuses_without_tty(self):
        with patch('sys.stdout', new=io.StringIO()), \
             patch('sys.stderr', new=io.StringIO()) as err:
            with self.assertRaises(SystemExit) as cm:
                cli.main(['--port', '1'])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn('not a terminal', err.getvalue())


if __name__ == '__main__':
    unittest.main()
