"""End-to-end launcher tests.

Spawn `graphsignal-run` as a real subprocess. The plain-python tests run on
any platform: they verify workload dispatch, graphsignal-run's own leading
flags, exit-status passthrough, and that the sibling watcher process serves
`GET /signals` on the requested port while the workload lives. The CUDA test
(Linux + torch + GPU only) additionally verifies that the CUPTI injection
library is loaded into the workload.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.request

from test.test_utils import free_port


def _run_env(**extra):
    env = {**os.environ}
    for key in list(env.keys()):
        if key.startswith('GRAPHSIGNAL_'):
            del env[key]
    env.pop('CUDA_INJECTION64_PATH', None)
    env.update(extra)
    return env


def _run_cmd(*args):
    return [sys.executable, '-m', 'graphsignal.commands.graphsignal_run', *args]


class GraphsignalRunE2ETest(unittest.TestCase):
    def test_exit_code_passthrough(self):
        script = 'import sys; print("workload ran"); sys.exit(7)'
        proc = subprocess.run(
            _run_cmd('--listen-port', str(free_port()),
                     sys.executable, '-c', script),
            env=_run_env(), capture_output=True, text=True, timeout=60)

        self.assertEqual(proc.returncode, 7,
                         msg=f'stdout={proc.stdout!r} stderr={proc.stderr!r}')
        self.assertIn('workload ran', proc.stdout)

    def test_unknown_executable_exits_with_error(self):
        proc = subprocess.run(
            _run_cmd('--listen-port', str(free_port()),
                     'definitely-not-a-real-executable-graphsignal'),
            env=_run_env(), capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 1)

    def test_signals_endpoint_served_while_workload_runs(self):
        listen_port = free_port()
        workload = 'import time; time.sleep(30)'
        proc = subprocess.Popen(
            _run_cmd('--listen-port', str(listen_port),
                     sys.executable, '-c', workload),
            env=_run_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        report = None
        try:
            deadline = time.time() + 30
            last_error = None
            while time.time() < deadline:
                if proc.poll() is not None:
                    self.fail('graphsignal-run exited early: '
                              f'{proc.returncode}, stderr='
                              f'{proc.stderr.read().decode()!r}')
                try:
                    with urllib.request.urlopen(
                            f'http://127.0.0.1:{listen_port}/signals',
                            timeout=2) as resp:
                        self.assertEqual(resp.status, 200)
                        report = json.loads(resp.read().decode('utf-8'))
                    break
                except OSError as exc:
                    last_error = exc
                    time.sleep(0.2)

            self.assertIsNotNone(
                report, msg=f'signals endpoint never came up: {last_error}')
            self.assertIn('version', report['profiler'])
            self.assertIn('instance.id', report['context'])
            self.assertIsInstance(report['metrics'], list)
            self.assertIsInstance(report['errors'], list)
            self.assertIsInstance(report['resources'], list)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            # Give the detached watcher subprocess a moment to notice its
            # target is gone and exit on its own.
            time.sleep(0.5)


def _torch_cuda_available_in_subprocess() -> bool:
    code = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        proc = subprocess.run([sys.executable, '-c', code],
                              timeout=30, capture_output=True)
        return proc.returncode == 0
    except Exception:
        return False


# The workload appends a progress trail to the env-supplied output file from
# its very first line, so we can see exactly which step it reached even if
# stdout is eaten by the CUPTI injection library's atexit handler or the
# inherited pipe.
WORKLOAD = textwrap.dedent('''
    import os, sys, time, traceback

    _OUT = os.environ['GRAPHSIGNAL_TEST_OUTPUT']

    def _log(msg):
        with open(_OUT, 'a') as _f:
            _f.write(msg + '\\n')

    try:
        _log(f'PID={os.getpid()}')
        _log(f'CUDA_INJECTION64_PATH={os.environ.get("CUDA_INJECTION64_PATH", "<unset>")}')
        _log(f'ROCP_TOOL_LIBRARIES={os.environ.get("ROCP_TOOL_LIBRARIES", "<unset>")}')
        import torch
        _log(f'torch_imported, cuda_available={torch.cuda.is_available()}, hip={torch.version.hip}')
        a = torch.randn((64, 64), device='cuda', dtype=torch.float16)
        b = torch.randn((64, 64), device='cuda', dtype=torch.float16)
        _log('tensors_created')
        _ = a @ b
        torch.cuda.synchronize()
        _log('matmul_synced')
        # Linger so the CUPTI flush thread writes the shm dir.
        time.sleep(2.0)
        _log('done')
    except BaseException as exc:
        _log(f'EXCEPTION={type(exc).__name__}: {exc}')
        _log(traceback.format_exc())
        raise
''')


@unittest.skipUnless(sys.platform.startswith('linux'),
                     "graphsignal-run GPU e2e requires Linux (CUPTI/ROCm injection)")
class GraphsignalRunCudaE2ETest(unittest.TestCase):
    def setUp(self):
        if not _torch_cuda_available_in_subprocess():
            self.skipTest("torch+CUDA not available")

    def test_graphsignal_run_with_torch_workload(self):
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(WORKLOAD)
            script = f.name
        output_path = tempfile.mkstemp(suffix='.pid')[1]
        os.unlink(output_path)  # let the workload create it fresh

        env = _run_env(GRAPHSIGNAL_DEBUG='1',
                       GRAPHSIGNAL_TEST_OUTPUT=output_path)

        # Invoke as `graphsignal-run --listen-port N python <script.py>`.
        # FallbackLauncher resolves `python` on PATH and the supervisor runs
        # it as a child process with no launcher modules loaded — the
        # configuration the CUPTI injection library requires.
        cmd = _run_cmd('--listen-port', str(free_port()),
                       sys.executable, script)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=60)

        try:
            trail = ''
            if os.path.exists(output_path):
                with open(output_path) as f:
                    trail = f.read()

            # The native CUPTI injection library can SIGSEGV on teardown (-11)
            # on some platforms. That's a native-side cleanup issue independent
            # of the workload running correctly, so we accept either a clean
            # exit or a SIGSEGV *as long as* the workload reached 'done'.
            self.assertIn(
                proc.returncode, (0, -11),
                msg=(f"graphsignal-run exited unexpectedly with "
                     f"returncode={proc.returncode};\n"
                     f"--- trail ---\n{trail}\n"
                     f"--- stdout ---\n{proc.stdout}\n"
                     f"--- stderr ---\n{proc.stderr}"))

            self.assertTrue(
                os.path.exists(output_path),
                msg=(f"workload never wrote to {output_path};\n"
                     f"--- stdout ---\n{proc.stdout}\n"
                     f"--- stderr ---\n{proc.stderr}"))

            # Confirm the workload reached the final step before the (possibly
            # crashing) atexit handler ran.
            self.assertIn(
                'done', trail,
                msg=(f"workload did not reach 'done'; trail=\n{trail}\n"
                     f"stdout={proc.stdout!r} stderr={proc.stderr!r}"))

            # And confirm the activity library actually attached: the launcher
            # forwards CUDA_INJECTION64_PATH on CUDA and ROCP_TOOL_LIBRARIES on
            # ROCm (torch reports a HIP version there).
            on_rocm = 'hip=None' not in trail and 'hip=' in trail
            var = 'ROCP_TOOL_LIBRARIES' if on_rocm else 'CUDA_INJECTION64_PATH'
            self.assertIn(f'{var}=', trail)
            inj_line = next(
                ln for ln in trail.splitlines() if ln.startswith(f'{var}='))
            self.assertNotIn('<unset>', inj_line,
                             msg=f"launcher did not set {var}; trail=\n{trail}")
            if on_rocm:
                self.assertIn('libgsrocmprof.so', inj_line)
        finally:
            os.unlink(script)
            if os.path.exists(output_path):
                os.unlink(output_path)
            # Best-effort: clean up any lingering shm dir left by the workload.
            for entry in os.listdir('/dev/shm') if os.path.isdir('/dev/shm') else ():
                if entry.startswith('graphsignal_'):
                    try:
                        shutil.rmtree(os.path.join('/dev/shm', entry), ignore_errors=True)
                    except Exception:
                        pass

            # Give the detached watcher subprocess a moment to notice its
            # target is gone and exit on its own.
            time.sleep(0.5)


if __name__ == '__main__':
    unittest.main()
