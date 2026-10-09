"""graphsignal-top: a live terminal view of a workload profiled by graphsignal-run.

Everything here reads the watcher's `GET /signals` payload over HTTP and
renders it; nothing in this package touches the stores, the recorders, or
the workload. Every module except `ui.py` is terminal-free so the view can be
unit-tested from fixture payloads and printed once with `--once`.
"""
