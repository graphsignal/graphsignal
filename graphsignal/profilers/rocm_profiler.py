import logging
import os
import platform
import re
import sys
from typing import Optional

logger = logging.getLogger("graphsignal")

# rocprofiler-sdk releases below this crash the workload inside hipGraphLaunch
# once kernel dispatch tracing is on — AMD's own rocprofv3 --kernel-trace
# included. Verified: 1.1.0 (ROCm 7.2) crashes SGLang on its first decode
# graph replay; 1.3.5 (ROCm 10.0) runs the same workload clean.
MIN_GRAPH_SAFE_ROCPROFILER_SDK = (1, 3, 5)

# Per-kernel tracing on ROCm. The native library reads this at injection time
# (none|all, default none): rocprofiler-sdk intercepts the completion of every
# traced dispatch at tens of microseconds each, so kernel tracing is an
# investigation mode on ROCm, not the always-on default it is on CUDA.
ROCM_KERNEL_TRACE_ENV_VAR = "GRAPHSIGNAL_ROCM_KERNEL_TRACE"
ROCM_KERNEL_TRACE_MODES = ("none", "all")


class RocmProfiler:
    """Holds ROCm environment setup that must run before the target's HIP/ROCm initialization.

    Launchers call setup_env_vars() before exec'ing the target so the
    rocprofiler-sdk tool library (libgsrocmprof.so) is registered via
    ROCP_TOOL_LIBRARIES.

    Unlike the CUPTI backend there is no pip wheel for the ROCm profiling stack:
    rocprofiler-sdk ships only as a system ROCm package (preinstalled in the
    target images, e.g. rocm/dev-ubuntu-*-complete or rocm/pytorch). If it is
    not found this method warns once and returns False (profiling stays off,
    the target runs unaffected), so it is safe to call unconditionally alongside
    CuptiProfiler.
    """

    @staticmethod
    def setup_env_vars(cuda_graph_trace: Optional[str] = None,
                       rocm_kernel_trace: Optional[str] = None) -> bool:
        if cuda_graph_trace:
            # Accepted and ignored. rocprofiler-sdk reports every kernel
            # dispatch individually, graph-replayed ones included, so there is
            # no graph-vs-node granularity to select on ROCm.
            logger.debug(
                "--cuda-graph-trace=%s ignored on ROCm: rocprofiler-sdk reports "
                "kernel dispatches individually", cuda_graph_trace)

        if not sys.platform.startswith("linux"):
            logger.debug("ROCm not supported on this platform")
            return False

        rocm_major = _detect_rocm_major()
        if not rocm_major:
            logger.debug("ROCm not available, skipping ROCm env setup")
            return False

        tool_lib = _packaged_rocm_so_path(rocm_major)
        if not tool_lib:
            logger.warning(
                "ROCm profiler shared library (libgsrocmprof.so) not found for "
                "ROCm %s, skipping ROCm env setup.",
                rocm_major,
            )
            return False

        lib_dir = _ensure_librocprofiler_ld_library_path()
        if not lib_dir:
            logger.warning(
                "librocprofiler-sdk.so not found (ROCm %s detected), skipping ROCm env setup. "
                "Install the rocprofiler-sdk system package (it ships with ROCm) "
                "or use a ROCm image that includes it (e.g. rocm/dev-ubuntu-*-complete "
                "or rocm/pytorch). There is no pip package for the ROCm profiling stack.",
                rocm_major,
            )
            return False

        kernel_trace = _set_rocm_kernel_trace(rocm_kernel_trace)
        if kernel_trace == "all":
            sdk_version = _rocprofiler_sdk_version(lib_dir)
            if sdk_version and sdk_version < MIN_GRAPH_SAFE_ROCPROFILER_SDK:
                logger.warning(
                    "rocprofiler-sdk %s crashes a workload that replays HIP graphs while "
                    "kernel tracing is on (fixed upstream; %s ships with ROCm 10). Run the "
                    "workload in eager mode under this ROCm (SGLang: --disable-cuda-graph, "
                    "vLLM: --enforce-eager), drop --rocm-kernel-trace all, or upgrade ROCm.",
                    ".".join(map(str, sdk_version)),
                    ".".join(map(str, MIN_GRAPH_SAFE_ROCPROFILER_SDK)),
                )

        _ensure_rocp_tool_libraries(tool_lib)

        logger.debug("ROCm env setup complete (rocm %s)", rocm_major)
        return True


def _set_rocm_kernel_trace(rocm_kernel_trace: Optional[str]) -> str:
    """Publish the per-kernel tracing mode to the child's environment and
    return the mode in effect. `graphsignal-run --rocm-kernel-trace` beats a
    value inherited from the environment; without the flag an inherited value
    is left alone (an unrecognized one counts as the native default, none)."""
    if rocm_kernel_trace:
        os.environ[ROCM_KERNEL_TRACE_ENV_VAR] = rocm_kernel_trace
        logger.debug("%s=%s", ROCM_KERNEL_TRACE_ENV_VAR, rocm_kernel_trace)
        return rocm_kernel_trace
    inherited = os.environ.get(ROCM_KERNEL_TRACE_ENV_VAR, "").strip().lower()
    return inherited if inherited in ROCM_KERNEL_TRACE_MODES else "none"


def _packaged_rocm_so_path(rocm_major: int) -> Optional[str]:
    """The packaged tool library for this ROCm major, or the newest packaged
    build for an older major when there is none for this one.

    rocprofiler-sdk keeps its ABI across ROCm majors (the soname stays
    librocprofiler-sdk.so.1, and every record carries its own size so a tool
    compiled against older headers reads newer records), so a build for an
    older ROCm is a working choice on a newer one — the alternative is no GPU
    profiling at all on a ROCm release that is newer than this wheel.
    """
    try:
        from importlib import resources

        # ROCm is amd64 only.
        native_dir = resources.files("graphsignal").joinpath("_native")
        with resources.as_file(native_dir) as native_path:
            exact = os.path.join(native_path, f"amd64-rocm{rocm_major}", "libgsrocmprof.so")
            if os.path.exists(exact):
                return exact
            logger.debug("ROCm profiler shared library not found for path: %s", exact)

            candidates = []
            for entry in os.listdir(native_path) if os.path.isdir(native_path) else ():
                m = re.match(r'^amd64-rocm(\d+)$', entry)
                if not m:
                    continue
                major = int(m.group(1))
                path = os.path.join(native_path, entry, "libgsrocmprof.so")
                if major < rocm_major and os.path.exists(path):
                    candidates.append((major, path))
            if candidates:
                major, path = max(candidates)
                logger.debug(
                    "No ROCm profiler shared library built for ROCm %s; using the "
                    "ROCm %s build (rocprofiler-sdk keeps its ABI across majors)",
                    rocm_major, major)
                return path
    except Exception:
        logger.debug("Error locating the ROCm profiler shared library", exc_info=True)

    return None


def _detect_rocm_major() -> Optional[int]:
    for var in ('ROCM_VERSION', 'ROCM_TOOLKIT_VERSION'):
        val = os.environ.get(var, '')
        if val:
            m = re.match(r'^(\d+)', val.strip())
            if m:
                return int(m.group(1))

    torch_mod = sys.modules.get('torch')
    if torch_mod is not None:
        try:
            s = getattr(torch_mod.version, 'hip', None)
            if isinstance(s, str):
                m = re.match(r'^(\d+)', s.strip())
                if m:
                    return int(m.group(1))
        except Exception:
            pass

    # Infer from the ROCm install. Try, in order: the version file ROCm ships
    # (.info/version, e.g. "7.2.3-123"), then the resolved install-dir name
    # (e.g. /opt/rocm -> /opt/rocm-7.2.3). ROCM_PATH is often the unversioned
    # symlink "/opt/rocm", so resolve it before matching.
    #
    # NOTE: deliberately NOT using ctypes.util.find_library('rocprofiler-sdk'):
    # it returns the ABI soname (librocprofiler-sdk.so.1), whose "1" is the
    # library ABI version, not the ROCm major — it would wrongly report ROCm 1.
    for base in (os.environ.get('ROCM_PATH'), os.environ.get('ROCM_HOME'), '/opt/rocm'):
        if not base:
            continue
        major = _read_rocm_version_file(base)
        if major:
            return major
        for candidate in (base, os.path.realpath(base)):
            m = re.search(r'rocm-(\d+)', candidate)
            if m:
                return int(m.group(1))

    # Fall back to scanning /opt for a versioned ROCm install dir (e.g.
    # /opt/rocm-7.2.3). Pick the highest major found.
    try:
        majors = []
        for entry in os.listdir('/opt'):
            m = re.match(r'^rocm-(\d+)', entry)
            if m and os.path.isdir(os.path.join('/opt', entry)):
                majors.append(int(m.group(1)))
        if majors:
            return max(majors)
    except Exception:
        pass

    return None


def _read_rocm_version_file(base: str) -> Optional[int]:
    # ROCm ships a plain-text version file under <install>/.info/ (e.g.
    # "7.2.3-123"). Return its major, or None if unavailable/unparsable.
    for fname in ('.info/version', '.info/version-dev', '.info/version-utils'):
        try:
            with open(os.path.join(base, fname)) as f:
                content = f.read().strip()
        except Exception:
            continue
        m = re.match(r'^(\d+)', content)
        if m:
            return int(m.group(1))
    return None


def _ensure_rocp_tool_libraries(tool_lib: str) -> None:
    existing = os.environ.get("ROCP_TOOL_LIBRARIES", "")
    entries = [e for e in existing.split(":") if e]
    if tool_lib not in entries:
        os.environ["ROCP_TOOL_LIBRARIES"] = ":".join([tool_lib] + entries) if entries else tool_lib
        logger.debug("Added ROCm tool library to ROCP_TOOL_LIBRARIES: %s", tool_lib)


def _rocprofiler_sdk_version(lib_dir: str) -> Optional[tuple]:
    """The rocprofiler-sdk release in `lib_dir`, from its fully versioned
    library file name (librocprofiler-sdk.so.1.3.5), or None."""
    best = None
    try:
        for entry in os.listdir(lib_dir):
            m = re.match(r'^librocprofiler-sdk\.so\.(\d+)\.(\d+)\.(\d+)$', entry)
            if m:
                version = tuple(int(x) for x in m.groups())
                if best is None or version > best:
                    best = version
    except Exception:
        return None
    return best


def _ensure_librocprofiler_ld_library_path() -> Optional[str]:
    """Make librocprofiler-sdk.so loadable by the target and return the
    directory it was found in (None when it is not installed)."""
    def _has_lib(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        try:
            return any(e.startswith("librocprofiler-sdk.so") for e in os.listdir(d))
        except Exception:
            return False

    def _prepend(lib_dir: str) -> None:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        dirs = [d for d in existing.split(":") if d]
        if lib_dir not in dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir] + dirs)
            logger.debug("Added rocprofiler-sdk lib dir to LD_LIBRARY_PATH: %s", lib_dir)

    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    candidates = [
        os.path.join(rocm_path, "lib"),
        os.path.join(rocm_path, "lib64"),
        "/opt/rocm/lib",
        "/opt/rocm/lib64",
    ]

    import ctypes.util
    found = ctypes.util.find_library("rocprofiler-sdk")
    if found:
        # Already loadable; if it resolves to a concrete path, prepend its dir.
        d = os.path.dirname(found) if os.path.sep in found else ""
        if d and _has_lib(d):
            _prepend(d)
            return d
        # find_library reports the soname only: the install dir is one of
        # the ROCm locations (the loader would not have found it otherwise).
        for base in candidates:
            if _has_lib(base):
                return base
        return found

    for base in candidates:
        if _has_lib(base):
            _prepend(base)
            return base

    return None
