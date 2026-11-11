"""Watcher lifecycle functions.

`graphsignal-watch` (and any code that configures the watcher process) imports
these. Access the active watcher via `graphsignal.watcher.watcher()` (raises if
not configured).
"""

from typing import Dict, Optional
import atexit
import logging
import os

from graphsignal.watcher.env_vars import read_config_param, read_config_tags
from graphsignal.watcher.watcher import Watcher
from graphsignal.signals.routes import DEFAULT_LISTEN_HOST, DEFAULT_LISTEN_PORT

logger = logging.getLogger('graphsignal')

# Module-private singleton. Read via watcher(); not part of the public surface.
_watcher: Optional[Watcher] = None


def configure(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    debug_mode: Optional[bool] = None,
    target_pid: Optional[int] = None,
    metrics_port: Optional[int] = None,
    metrics_path: Optional[str] = None,
    metrics_host: Optional[str] = None,
    listen_host: Optional[str] = None,
    listen_port: Optional[int] = None,
    listen_fd: Optional[int] = None,
    listen_token: Optional[str] = None,
) -> None:
    global _watcher

    if _watcher:
        logger.warning("Watcher already configured")
        return

    # The api key is optional: it only enables the production feedback loop
    # (signal uploads via the collector). The local profiler works without it.
    api_key = read_config_param("api_key", str, api_key)
    api_base = read_config_param("api_base", str, api_base)
    tags = read_config_tags(tags)
    debug_mode = read_config_param("debug", bool, debug_mode, default_value=False)
    # 127.0.0.1 unless explicitly overridden: any other listen host exposes
    # the /signals endpoint to the network the host is reachable on.
    listen_host = read_config_param(
        "listen_host", str, listen_host, default_value=DEFAULT_LISTEN_HOST)
    listen_port = read_config_param(
        "listen_port", int, listen_port, default_value=DEFAULT_LISTEN_PORT)
    listen_token = read_config_param("listen_token", str, listen_token)

    if target_pid is None:
        target_pid = os.getpid()

    _watcher = Watcher(
        api_key=api_key,
        api_base=api_base,
        tags=tags,
        debug_mode=debug_mode,
        target_pid=target_pid,
        metrics_port=metrics_port,
        metrics_path=metrics_path,
        metrics_host=metrics_host,
        listen_host=listen_host,
        listen_port=listen_port,
        listen_fd=listen_fd,
        listen_token=listen_token)
    try:
        _watcher.setup()
    except Exception:
        failed, _watcher = _watcher, None
        failed._auto_tick = False
        try:
            failed.shutdown()
        except Exception:
            pass
        raise

    atexit.register(shutdown)

    logger.debug('Watcher configured')


def watcher() -> Watcher:
    """Return the configured watcher singleton, or raise if not configured."""
    if _watcher is None:
        raise RuntimeError('Watcher not configured; call graphsignal.watcher.configure() first')
    return _watcher


def is_configured() -> bool:
    """True iff `configure()` has run and `shutdown()` has not."""
    return _watcher is not None


def tick(block: bool = False, force: bool = False) -> None:
    watcher().tick(block=block, force=force)


def shutdown() -> None:
    global _watcher
    if not _watcher:
        return

    atexit.unregister(shutdown)
    _watcher.shutdown()
    _watcher = None

    logger.debug('Watcher shutdown')


__all__ = [
    'Watcher',
    'configure',
    'watcher',
    'is_configured',
    'tick',
    'shutdown',
]
