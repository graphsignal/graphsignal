import os
import re
import subprocess
import sys

import pytest
from _pytest.reports import TestReport

# Environment variable used to break the recursion: when the subprocess that
# runs a cuda-marked test loads this conftest, it sees the variable and falls
# back to normal (non-subprocess) execution.
_CUDA_SUBPROCESS_VAR = "PYTEST_CUDA_SUBPROCESS"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda: test uses real CUDA; run in a fresh subprocess to avoid fork+CUDA incompatibility",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if not item.get_closest_marker("cuda"):
        return None
    if os.environ.get(_CUDA_SUBPROCESS_VAR):
        return None  # already inside a subprocess invocation — run normally

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

    env = {**os.environ, _CUDA_SUBPROCESS_VAR: "1"}
    # `test_cupti_recorder.py`'s module-level `CuptiProfiler.setup_env_vars()`
    # leaves `CUDA_INJECTION64_PATH` set on the parent pytest process. Other
    # cuda-marked tests (e.g. NVML) must not inherit it — loading the CUPTI
    # injection library into a process that already has the SDK running
    # crashes inside the CUDA driver init. Strip the env var here; tests that
    # do want the injection (the CUPTI tests themselves) re-set it at module
    # import inside their own subprocess.
    env.pop('CUDA_INJECTION64_PATH', None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", item.nodeid,
         "-p", "no:forked", "--tb=short", "-v"],
        capture_output=True,
        text=True,
        env=env,
    )
    passed = proc.returncode == 0
    output = (proc.stdout + proc.stderr).strip()
    # A test that skipped inside the subprocess also exits 0; report it as
    # skipped (with its reason) rather than letting it pass for a GPU test
    # that never ran.
    skipped = passed and re.search(r"\b1 skipped\b", output) is not None
    skip_reason = "skipped in subprocess"
    m = re.search(r"^SKIPPED \[\d+\] [^:]+:\d+: (.*)$", output, re.M)
    if m:
        skip_reason = m.group(1).strip()

    for when in ("setup", "call", "teardown"):
        if when == "call":
            if skipped:
                outcome = "skipped"
                longrepr = (str(item.fspath), item.location[1] or 0, skip_reason)
            else:
                outcome = "passed" if passed else "failed"
                longrepr = None if passed else output
        else:
            outcome = "passed"
            longrepr = None

        report = TestReport(
            nodeid=item.nodeid,
            location=item.location,
            keywords={m.name: True for m in item.iter_markers()},
            outcome=outcome,
            longrepr=longrepr,
            when=when,
            sections=[("subprocess output", output)] if output and when == "call" else [],
            duration=0.0,
            user_properties=[],
        )
        ihook.pytest_runtest_logreport(report=report)

    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True
