from typing import Dict, Optional
import hashlib
import importlib
import logging
import os
import time
import threading
import uuid

from graphsignal.watcher.pid_monitor import PidMonitor
from graphsignal.signals.metrics import MetricStore
from graphsignal.signals.logs import LogStore
from graphsignal.signals.resources import ResourceStore
from graphsignal.signals.routes import SignalsEndpoint, DEFAULT_LISTEN_HOST, DEFAULT_LISTEN_PORT

logger = logging.getLogger('graphsignal')


def uuid_sha1(size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(str(uuid.uuid4()).encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


class GraphsignalLogHandler(logging.Handler):
    def __init__(self, watcher):
        super().__init__()
        self._watcher = watcher

    def emit(self, record):
        try:
            exception = None
            if record.exc_info and isinstance(record.exc_info, tuple):
                exception = self.format(record)

            self._watcher.log_store().log_watcher_message(
                level=record.levelname,
                message=record.getMessage(),
                exception=exception)
        except Exception:
            pass


class Watcher:
    TICK_DELAY_SEC = 1
    TICK_INTERVAL_SEC = 1
    MAX_TAGS = 25

    def __init__(
            self,
            api_key=None,
            api_base=None,
            tags=None,
            debug_mode=False,
            target_pid=None,
            metrics_port=None,
            metrics_path=None,
            metrics_host=None,
            listen_host=None,
            listen_port=None):
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        self._api_key = api_key
        self._api_base = api_base
        self._tags = {}
        if tags:
            self._tags.update(tags)

        self.debug_mode = debug_mode

        self._target_pid = int(target_pid) if target_pid is not None else os.getpid()
        self._metrics_port = int(metrics_port) if metrics_port is not None else None
        self._metrics_path = metrics_path
        self._metrics_host = metrics_host
        self._listen_host = str(listen_host) if listen_host is not None else DEFAULT_LISTEN_HOST
        self._listen_port = int(listen_port) if listen_port is not None else DEFAULT_LISTEN_PORT

        self._tick_timer_thread = None
        self._tick_stop_event = threading.Event()
        self._tick_lock = threading.Lock()
        self._tick_run_thread = None
        self._python_log_handler = None
        self._metric_store = None
        self._log_store = None
        self._resource_store = None
        self._signals_endpoint = None
        self._collector = None

        self._pid_monitor = None
        self._global_recorders: list = []
        self._child_recorders: dict = {}
        self._recorders_lock = threading.Lock()

        self._target_terminated_event = threading.Event()
        self._shutdown_started = False
        self._shutdown_lock = threading.Lock()

        self._start_ns = time.time_ns()
        self._last_tick_ts = time.time()
        self._auto_tick = True

    def setup(self):
        self._log_store = LogStore()
        if not self._python_log_handler:
            self._python_log_handler = GraphsignalLogHandler(self)
            logger.addHandler(self._python_log_handler)

        logger.info('Watcher started: pid=%s', os.getpid())
        logger.debug('Watcher setup started')

        if 'instance.id' not in self._tags:
            self.set_tag('instance.id', uuid_sha1(size=12))

        self._metric_store = MetricStore()
        self._resource_store = ResourceStore()

        if self._api_key:
            from graphsignal.collector.collector import Collector
            self._collector = Collector(self._api_key, api_base=self._api_base)
            self._collector.setup()
            logger.debug('Collector enabled (api key provided)')

        self._signals_endpoint = SignalsEndpoint(
            host=self._listen_host, port=self._listen_port)
        self._signals_endpoint.setup()

        self._pid_monitor = PidMonitor(self._target_pid)
        self._pid_monitor.add_listener(self)
        self._pid_monitor.setup()

        self._start_tick_timer()

        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(after_in_child=self._shutdown_in_fork_child)

        logger.debug('Watcher setup complete')

    def on_target_known(self, pid):
        # Fired from PidMonitor.setup(), before the target is observed alive.
        # The console-log dir is a pid-derived artifact written by the
        # graphsignal-run supervisor; it exists (and must be drained) even if
        # the target dies before on_target_created — so the LogRecorder is
        # created from the pid alone, kept out of the on_target_created set.
        from graphsignal.recorders.log_recorder import LogRecorder

        recorder = LogRecorder(root_pid=self._target_pid, pid=pid, args=None)
        # Registered before on_target_created runs; that handler extends this
        # list rather than replacing it, so this recorder is preserved.
        with self._recorders_lock:
            self._global_recorders.append(recorder)

        try:
            recorder.setup()
        except Exception:
            logger.error('Failed to set up recorder %s for target pid %s',
                         type(recorder).__name__, pid, exc_info=True)

    def on_target_created(self, args):
        from graphsignal.watcher.version_check import start_version_check

        # Fires here rather than at setup: this callback runs exactly once, when
        # the target is first seen alive, so a launch that never starts never
        # asks. It returns immediately — the check runs on a thread of its own,
        # because this one is the pid monitor's poll loop.
        start_version_check(self._api_base, self._tick_stop_event)

        # Each recorder is imported and constructed on its own: an import that
        # fails (a vendor library absent from this install) or a constructor
        # that raises must cost that one recorder, never the ones after it.
        def create(module_name, class_name, **kwargs):
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                return cls(root_pid=self._target_pid, pid=self._target_pid,
                           args=args, **kwargs)
            except Exception:
                logger.error('Failed to create recorder %s for target pid %s',
                             class_name, self._target_pid, exc_info=True)
                return None

        # HostRecorder goes first: it sets the host tags.
        specs = [
            ('graphsignal.recorders.host_recorder', 'HostRecorder', {}),
            ('graphsignal.recorders.nvml_recorder', 'NVMLRecorder', {}),
            ('graphsignal.recorders.amdsmi_recorder', 'AmdSmiRecorder', {}),
        ]
        if self._metrics_port is not None:
            specs.append((
                'graphsignal.recorders.prometheus_recorder', 'PrometheusRecorder',
                dict(metrics_port=self._metrics_port,
                     metrics_path=self._metrics_path,
                     metrics_host=self._metrics_host)))
        specs.extend([
            ('graphsignal.recorders.process_recorder', 'ProcessRecorder', {}),
            ('graphsignal.recorders.shm_recorder', 'ShmRecorder', {}),
        ])
        recorders = []
        for module_name, class_name, kwargs in specs:
            recorder = create(module_name, class_name, **kwargs)
            if recorder is not None:
                recorders.append(recorder)

        # Extend (not replace): a LogRecorder registered earlier by
        # on_target_known already lives in this list.
        with self._recorders_lock:
            self._global_recorders.extend(recorders)

        for recorder in recorders:
            try:
                recorder.setup()
            except Exception:
                logger.error('Failed to set up recorder %s for target pid %s',
                             type(recorder).__name__, self._target_pid, exc_info=True)

    def on_child_created(self, pid, args):
        from graphsignal.recorders.process_recorder import ProcessRecorder
        from graphsignal.recorders.shm_recorder import ShmRecorder

        recorders = [
            ProcessRecorder(root_pid=self._target_pid, pid=pid, args=args),
            ShmRecorder(root_pid=self._target_pid, pid=pid, args=args),
        ]
        with self._recorders_lock:
            self._child_recorders[pid] = recorders

        for recorder in recorders:
            try:
                recorder.setup()
            except Exception:
                logger.error('Failed to set up recorder %s for child pid %s',
                             type(recorder).__name__, pid, exc_info=True)

    def on_child_terminated(self, pid):
        with self._recorders_lock:
            recorders = self._child_recorders.pop(pid, [])
        for recorder in recorders:
            try:
                recorder.shutdown()
            except Exception:
                logger.error('Failed to shutdown recorder %s for child pid %s',
                             type(recorder).__name__, pid, exc_info=True)

    def on_target_terminated(self):
        # Run a final tick + shutdown on a background thread so the monitor loop
        # can return promptly; signal target_terminated_event after completion.
        def _finalize():
            try:
                self._auto_tick = False
                # Let recorders push any buffered final data into the stores
                # first — so a target's dying words (incl. a trailing error
                # line or the supervisor's exit-status record) land in the
                # stores before the final tick.
                for recorder in self.recorders():
                    try:
                        recorder.finalize()
                    except Exception:
                        logger.error('Error finalizing recorder %s',
                                     type(recorder).__name__, exc_info=True)
                self.tick(block=True, force=True)
            except Exception:
                logger.error('Error during target_terminated final tick', exc_info=True)
            finally:
                self._target_terminated_event.set()

        threading.Thread(target=_finalize, daemon=True).start()

    def _start_tick_timer(self):
        self._tick_stop_event = threading.Event()

        def _tick_loop():
            if not self._tick_stop_event.wait(Watcher.TICK_DELAY_SEC):
                try:
                    if self._auto_tick:
                        self.tick(force=True)
                except Exception as exc:
                    logger.error('Error in initial tick: %s', exc, exc_info=True)

            while not self._tick_stop_event.wait(Watcher.TICK_INTERVAL_SEC):
                try:
                    if self._auto_tick:
                        self.tick()
                except Exception as exc:
                    logger.error('Error in tick timer: %s', exc, exc_info=True)

        self._tick_timer_thread = threading.Thread(target=_tick_loop, daemon=True)
        self._tick_timer_thread.start()

    def _shutdown_in_fork_child(self):
        # The watcher runs only in the parent. In a forked child, shut
        # everything down via the module-level shutdown() so the singleton is
        # cleared as well (otherwise `is_configured()` would still report True).
        try:
            import graphsignal.watcher as gwatcher
            gwatcher.shutdown()
        except Exception:
            pass

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        if self._auto_tick:
            try:
                self.tick(block=True, force=True)
            except Exception:
                logger.error('Error in final tick during shutdown', exc_info=True)

        if self._tick_stop_event:
            self._tick_stop_event.set()

        if self._tick_timer_thread:
            try:
                self._tick_timer_thread.join(timeout=5.0)
            except Exception:
                pass
            self._tick_timer_thread = None

        if self._tick_run_thread:
            try:
                self._tick_run_thread.join(timeout=5.0)
            except Exception:
                pass
            self._tick_run_thread = None

        if self._pid_monitor:
            try:
                self._pid_monitor.shutdown()
            except Exception:
                pass
            self._pid_monitor = None

        for recorder in self.recorders():
            try:
                recorder.shutdown()
            except Exception:
                logger.error('Error shutting down recorder', exc_info=True)
        with self._recorders_lock:
            self._global_recorders = []
            self._child_recorders = {}

        if self._collector:
            try:
                self._collector.shutdown()
            except Exception:
                pass
            self._collector = None

        if self._signals_endpoint:
            try:
                self._signals_endpoint.shutdown()
            except Exception:
                pass
            self._signals_endpoint = None

        self._metric_store = None
        self._log_store = None
        self._resource_store = None

        self._tags = None

        if self._python_log_handler:
            logger.removeHandler(self._python_log_handler)
            self._python_log_handler = None

    def target_pid(self) -> int:
        return self._target_pid

    def target_terminated_event(self) -> threading.Event:
        return self._target_terminated_event

    def metric_store(self):
        return self._metric_store

    def log_store(self):
        return self._log_store

    def resource_store(self):
        return self._resource_store

    def signals_endpoint(self):
        return self._signals_endpoint

    def collector(self):
        return self._collector

    def recorders(self):
        with self._recorders_lock:
            global_recs = list(self._global_recorders)
            child_recs = [r for rs in self._child_recorders.values() for r in rs]
        yield from global_recs
        yield from child_recs

    def tags(self) -> Dict[str, str]:
        if self._tags is None:
            return {}
        return self._tags.copy()

    def emit_tick(self):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_tick()
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def set_tag(self, key: str, value: str, append_uuid: Optional[bool] = False) -> None:
        if not key:
            logger.error('set_tag: key must be provided')
            return

        if value is None:
            self._tags.pop(key, None)
            return

        if len(self._tags) > Watcher.MAX_TAGS:
            logger.error('set_tag: too many tags (>{0})'.format(Watcher.MAX_TAGS))
            return

        if append_uuid:
            if not value:
                value = uuid_sha1(size=12)
            else:
                value = '{0}-{1}'.format(value, uuid_sha1(size=12))

        self._tags[key] = value

    def start_ns(self) -> int:
        """When this instance started, epoch nanoseconds."""
        return self._start_ns

    def get_tag(self, key: str) -> Optional[str]:
        return self._tags.get(key, None)

    def remove_tag(self, key: str) -> None:
        self._tags.pop(key, None)

    def set_gauge(self, name, value, measurement_ts, tags=None):
        self._metric_store.set_gauge(
            name=name, value=value, measurement_ts=measurement_ts, tags=tags)

    def set_counter(self, name, total, measurement_ts, tags=None):
        self._metric_store.set_counter(
            name=name, total=total, measurement_ts=measurement_ts, tags=tags)

    def set_histogram(self, name, bins=None, counts=None, measurement_ts=None,
                      tags=None, count=None, sum_val=None, min_val=None,
                      max_val=None):
        self._metric_store.set_histogram(
            name=name, bins=bins, counts=counts,
            measurement_ts=measurement_ts, tags=tags, count=count,
            sum_val=sum_val, min_val=min_val, max_val=max_val)

    def set_profile(self, name, frames, samples=None, measurement_ts=None,
                    tags=None):
        self._metric_store.set_profile(
            name=name, frames=frames, samples=samples,
            measurement_ts=measurement_ts, tags=tags)

    def log_message(self, message: str, *, tags: Optional[Dict[str, str]] = None,
                    level: Optional[str] = None, exception: Optional[str] = None):
        self.log_store().log_message(
            message=message, tags=tags, level=level, exception=exception)

    def update_resource(self, kind, tags=None, attributes=None,
                        first_seen_ts=None, last_seen_ts=None):
        self._resource_store.update_resource(
            kind=kind, tags=tags, attributes=attributes,
            first_seen_ts=first_seen_ts, last_seen_ts=last_seen_ts)

    def tick(self, block=False, force=False):
        now = time.time()
        if not force and (now - self._last_tick_ts) < Watcher.TICK_INTERVAL_SEC - 1:
            return

        if not self._tick_lock.acquire(blocking=False):
            return

        self._last_tick_ts = now

        def _run_tick():
            try:
                try:
                    self.emit_tick()
                except Exception:
                    logger.error('Error in tick recorder loop', exc_info=True)

                if self._collector:
                    self._collector.on_tick(self)
            except Exception as exc:
                logger.error('Error in tick execution: %s', exc, exc_info=True)
            finally:
                self._tick_lock.release()

        try:
            self._tick_run_thread = threading.Thread(target=_run_tick, daemon=True)
            self._tick_run_thread.start()
        except Exception:
            self._tick_lock.release()
            raise

        if block:
            self._tick_run_thread.join()
