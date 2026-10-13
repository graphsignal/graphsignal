import os
import pathlib
import tempfile
import unittest
from unittest import mock

from graphsignal.profilers import rocm_profiler


class PackagedRocmSoPathTest(unittest.TestCase):
    """The wheel ships one libgsrocmprof.so per ROCm major. The exact major
    wins; a newer ROCm than any packaged build gets the newest older build
    (rocprofiler-sdk keeps its ABI across majors); an older ROCm gets none."""

    def _with_native(self, majors):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        for major in majors:
            d = root / '_native' / f'amd64-rocm{major}'
            d.mkdir(parents=True)
            (d / 'libgsrocmprof.so').write_bytes(b'')
        (root / '_native' / 'amd64-cu12').mkdir(parents=True)
        return root

    def _lookup(self, root, major):
        with mock.patch('importlib.resources.files', return_value=root):
            return rocm_profiler._packaged_rocm_so_path(major)

    def test_exact_major(self):
        root = self._with_native([7, 10])
        self.assertEqual(self._lookup(root, 7),
                         str(root / '_native' / 'amd64-rocm7' / 'libgsrocmprof.so'))
        self.assertEqual(self._lookup(root, 10),
                         str(root / '_native' / 'amd64-rocm10' / 'libgsrocmprof.so'))

    def test_newer_rocm_falls_back_to_newest_older_build(self):
        root = self._with_native([7, 10])
        self.assertEqual(self._lookup(root, 11),
                         str(root / '_native' / 'amd64-rocm10' / 'libgsrocmprof.so'))
        root = self._with_native([7])
        self.assertEqual(self._lookup(root, 10),
                         str(root / '_native' / 'amd64-rocm7' / 'libgsrocmprof.so'))

    def test_older_rocm_gets_nothing(self):
        root = self._with_native([7, 10])
        self.assertIsNone(self._lookup(root, 6))

    def test_no_native_dir(self):
        root = pathlib.Path(tempfile.mkdtemp())
        self.assertIsNone(self._lookup(root, 7))


if __name__ == '__main__':
    unittest.main()


class RocprofilerSdkVersionTest(unittest.TestCase):
    def _lib_dir(self, names):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for name in names:
            pathlib.Path(tmp.name, name).write_bytes(b'')
        return tmp.name

    def test_version_from_library_file_name(self):
        lib_dir = self._lib_dir(['librocprofiler-sdk.so', 'librocprofiler-sdk.so.1',
                                 'librocprofiler-sdk.so.1.3.5',
                                 'librocprofiler-sdk-roctx.so.1.3.5'])
        self.assertEqual(rocm_profiler._rocprofiler_sdk_version(lib_dir), (1, 3, 5))

    def test_unversioned_install(self):
        lib_dir = self._lib_dir(['librocprofiler-sdk.so', 'librocprofiler-sdk.so.1'])
        self.assertIsNone(rocm_profiler._rocprofiler_sdk_version(lib_dir))
        self.assertIsNone(rocm_profiler._rocprofiler_sdk_version('/nonexistent'))

    def _setup(self, lib_dir, so_path='/pkg/_native/amd64-rocm7/libgsrocmprof.so',
               rocm_kernel_trace=None, inherited=None):
        with mock.patch.object(rocm_profiler.sys, 'platform', 'linux'), \
             mock.patch.object(rocm_profiler, '_detect_rocm_major', return_value=7), \
             mock.patch.object(rocm_profiler, '_packaged_rocm_so_path', return_value=so_path), \
             mock.patch.object(rocm_profiler, '_ensure_librocprofiler_ld_library_path',
                               return_value=lib_dir), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ROCP_TOOL_LIBRARIES', None)
            os.environ.pop(rocm_profiler.ROCM_KERNEL_TRACE_ENV_VAR, None)
            if inherited is not None:
                os.environ[rocm_profiler.ROCM_KERNEL_TRACE_ENV_VAR] = inherited
            with self.assertLogs('graphsignal', level='DEBUG') as cm:
                ok = rocm_profiler.RocmProfiler.setup_env_vars(
                    rocm_kernel_trace=rocm_kernel_trace)
            env = os.environ.get(rocm_profiler.ROCM_KERNEL_TRACE_ENV_VAR)
            return ok, cm.output, env

    def _warnings(self, output):
        return [line for line in output if line.startswith('WARNING')]

    def test_warns_on_graph_crashing_sdk_with_kernel_trace(self):
        lib_dir = self._lib_dir(['librocprofiler-sdk.so.1.1.0'])
        ok, output, env = self._setup(lib_dir, rocm_kernel_trace='all')
        self.assertTrue(ok)
        self.assertEqual(env, 'all')
        warnings = self._warnings(output)
        self.assertEqual(len(warnings), 1, output)
        self.assertIn('rocprofiler-sdk 1.1.0', warnings[0])
        self.assertIn('HIP graphs', warnings[0])
        self.assertIn('--disable-cuda-graph', warnings[0])

    def test_warns_when_kernel_trace_inherited_from_env(self):
        lib_dir = self._lib_dir(['librocprofiler-sdk.so.1.1.0'])
        ok, output, env = self._setup(lib_dir, inherited='all')
        self.assertTrue(ok)
        self.assertEqual(env, 'all')
        self.assertEqual(len(self._warnings(output)), 1, output)

    def test_silent_without_kernel_trace(self):
        # The crash is in kernel-dispatch tracing; the default never enables it.
        lib_dir = self._lib_dir(['librocprofiler-sdk.so.1.1.0'])
        ok, output, env = self._setup(lib_dir)
        self.assertTrue(ok)
        self.assertIsNone(env)
        self.assertFalse(self._warnings(output), output)
        ok, output, env = self._setup(lib_dir, rocm_kernel_trace='none')
        self.assertEqual(env, 'none')
        self.assertFalse(self._warnings(output), output)

    def test_flag_beats_inherited_value(self):
        lib_dir = self._lib_dir(['librocprofiler-sdk.so.1.3.5'])
        ok, output, env = self._setup(lib_dir, rocm_kernel_trace='none', inherited='all')
        self.assertEqual(env, 'none')
        ok, output, env = self._setup(lib_dir, inherited='bogus')
        self.assertEqual(env, 'bogus')  # left alone; the native default (none) applies
        self.assertFalse(self._warnings(output), output)

    def test_silent_on_fixed_sdk(self):
        lib_dir = self._lib_dir(['librocprofiler-sdk.so.1.3.5'])
        ok, output, env = self._setup(lib_dir, rocm_kernel_trace='all')
        self.assertTrue(ok)
        self.assertFalse(self._warnings(output), output)
