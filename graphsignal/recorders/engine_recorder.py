import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.prometheus_scraper import (
    PrometheusScraper, build_metrics_endpoint, buckets_to_bins,
    format_metrics_host, is_finite_number, iter_metric_groups,
    normalize_metrics_path)

logger = logging.getLogger('graphsignal')

HEALTH_INTERVAL_SEC = 5.0
HEALTH_TIMEOUT_SEC = 2.0
DEFAULT_HEALTH_HOST = '127.0.0.1'

_STARTING, _UP, _DOWN = 'starting', 'up', 'down'


class _HealthTimes:
    """Accrual for the health check, fed one probe result at a time.

    Every probe moves the wall time since the previous probe (for the first,
    since the probe started) into the state that probe observes: `starting`
    until the first healthy probe — model loading is not downtime — then `up`
    or `down`. A failure is only known when observed, so the gap before it
    counts as down: conservative, and off by at most one interval.

    A failed probe is held, not counted: a server that is shutting down (or
    has crashed and is exiting) refuses connections before the process is
    gone, so a failure alone cannot tell an outage from an exit. The held
    failure — its interval, and the down event when it ends an up period —
    is committed at the next probe if the engine process is still alive then,
    and dropped if it is not or if no next probe comes (the target
    terminated). Downtime therefore shows up one probe late, and an exit is
    never downtime.
    """

    def __init__(self, start_ns: int):
        self.state = _STARTING
        self.last_ns = start_ns
        self.startup_ns = 0
        self.uptime_ns = 0
        self.downtime_ns = 0
        self.down_events = 0
        self.held_ns: Optional[int] = None   # interval of a held failure
        self.held_down_event = False

    def record(self, healthy: bool, now_ns: int, engine_alive: bool = True) -> None:
        elapsed = max(0, now_ns - self.last_ns)
        self.last_ns = now_ns
        if self.held_ns is not None:
            if engine_alive:
                self.downtime_ns += self.held_ns
                if self.held_down_event:
                    self.down_events += 1
            self.held_ns = None
            self.held_down_event = False
        if self.state == _STARTING:
            # The interval that ends in the first healthy probe is still
            # startup: up is only known from that probe on.
            self.startup_ns += elapsed
            if healthy:
                self.state = _UP
            return
        if healthy:
            self.uptime_ns += elapsed
            self.state = _UP
        else:
            self.held_ns = elapsed
            self.held_down_event = self.state == _UP
            self.state = _DOWN


class EngineRecorder(BaseRecorder):
    """Imports the inference engine's own metrics from its Prometheus endpoint,
    and probes its health endpoint for uptime and downtime.

    The scrape port is resolved by the launcher (from `--metrics-port` or the
    engine's serving port) and passed in explicitly. The path defaults to
    `/metrics` (vLLM, SGLang); TensorRT-LLM uses `/prometheus/metrics`.
    We never enumerate or probe a process's other listening sockets —
    blindly connecting to them can corrupt internal IPC channels (e.g.
    TensorRT-LLM's ZeroMQ queues).

    The health probe runs on its own thread, so a hung server costs that
    thread a timeout, never the watcher's tick. The launcher picks the probe
    path per engine (one that does no engine work) and the engine's own
    serving port, never `--metrics-port`.
    """

    def __init__(self, root_pid=None, pid=None, args=None, metrics_port=None,
                 metrics_path: Optional[str] = None,
                 metrics_host: Optional[str] = None,
                 health_port: Optional[int] = None,
                 health_path: Optional[str] = None,
                 health_host: Optional[str] = None,
                 health_interval_sec: float = HEALTH_INTERVAL_SEC,
                 health_timeout_sec: float = HEALTH_TIMEOUT_SEC,
                 clock_ns: Callable[[], int] = time.monotonic_ns):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._scraper: Optional[PrometheusScraper] = None
        if metrics_port is not None:
            self._scraper = PrometheusScraper(build_metrics_endpoint(
                metrics_port, metrics_path=metrics_path, metrics_host=metrics_host))

        # `health_path` may list fallbacks, comma-separated: a 404 (the
        # engine's release has no such route) moves the probe to the next
        # path for the rest of the run.
        self._health_urls: List[str] = []
        if health_port is not None:
            host = format_metrics_host(health_host or DEFAULT_HEALTH_HOST)
            paths = [p.strip() for p in (health_path or '/health').split(',') if p.strip()]
            self._health_urls = [
                f'http://{host}:{int(health_port)}{normalize_metrics_path(p)}'
                for p in (paths or ['/health'])]
        self._health_interval_sec = health_interval_sec
        self._health_timeout_sec = health_timeout_sec
        self._clock_ns = clock_ns
        self._health_lock = threading.Lock()
        self._health: Optional[_HealthTimes] = None
        self._health_stop = threading.Event()
        self._health_thread: Optional[threading.Thread] = None

    @property
    def _health_url(self) -> Optional[str]:
        return self._health_urls[0] if self._health_urls else None

    def setup(self):
        # Scraping is lazy; the first attempt waits the initial delay since the
        # server may not be listening yet right after launch.
        if self._scraper is not None:
            self._scraper.start()
        if self._health_url is not None:
            # The recorder is created when the target is first seen alive:
            # startup time counts from here.
            self._health = _HealthTimes(self._clock_ns())
            self._health_thread = threading.Thread(
                target=self._health_loop, name='graphsignal-health', daemon=True)
            self._health_thread.start()

    def on_tick(self):
        self._emit_health()
        if self._scraper is None:
            return
        body = self._scraper.scrape()
        if body is None:
            return
        try:
            self._parse_and_emit(body)
        except Exception as exc:
            logger.error('Failed to parse Prometheus metrics: %s', exc, exc_info=True)

    def finalize(self):
        # The target has terminated: a process exit is not downtime, so stop
        # accruing and publish the totals as they stand.
        self._health_stop.set()
        self._emit_health()

    def shutdown(self):
        self._health_stop.set()
        thread, self._health_thread = self._health_thread, None
        if thread is not None:
            thread.join(timeout=self._health_timeout_sec + 1.0)

    def _health_loop(self):
        while not self._health_stop.wait(self._health_interval_sec):
            healthy = self._probe_health()
            # A probe that was in flight when the target terminated is
            # dropped: its answer says nothing about the run.
            if self._health_stop.is_set():
                return
            with self._health_lock:
                self._health.record(healthy, self._clock_ns(),
                                    engine_alive=self._target_alive())

    def _target_alive(self) -> bool:
        """Whether the engine process still exists; True when unknown."""
        if not self.pid:
            return True
        try:
            os.kill(int(self.pid), 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return True  # PermissionError and the like: it exists

    def _probe_health(self) -> bool:
        """Healthy means an HTTP 2xx within the timeout. Never raises."""
        while True:
            url = self._health_url
            try:
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=self._health_timeout_sec) as resp:
                    return 200 <= resp.status < 300
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and len(self._health_urls) > 1:
                    # This release has no such route: use the fallback from
                    # now on and probe it right away.
                    self._health_urls.pop(0)
                    logger.debug('Health probe %s not found; falling back to %s',
                                 url, self._health_url)
                    continue
                logger.debug('Health probe %s failed: %s', url, exc)
                return False
            except Exception as exc:
                logger.debug('Health probe %s failed: %s', url, exc)
                return False

    def _emit_health(self):
        if self._health is None or not graphsignal.watcher.is_configured():
            return
        try:
            with self._health_lock:
                totals = (
                    ('engine_startup_nanoseconds', self._health.startup_ns),
                    ('engine_uptime_nanoseconds', self._health.uptime_ns),
                    ('engine_downtime_nanoseconds', self._health.downtime_ns),
                    ('engine_down_events', self._health.down_events),
                )
            watcher = graphsignal.watcher.watcher()
            now_ns = time.time_ns()
            for name, total in totals:
                watcher.set_counter(name=name, total=total, measurement_ts=now_ns)
        except Exception:
            logger.error('Failed to record engine health counters', exc_info=True)

    def _parse_and_emit(self, body: str) -> None:
        # Scraped values pass through as-is: gauges and counters as the raw
        # cumulative values, histogram and summary families as one histogram
        # each — exact `_count`/`_sum`, plus `le` buckets converted to
        # non-cumulative bins where the family has them.
        watcher = graphsignal.watcher.watcher()
        now_ns = time.time_ns()

        for group in iter_metric_groups(body):
            name = group.name
            samples = group.samples
            tags = group.tags

            if group.type == 'gauge':
                s = samples.get(name)
                if s is not None and is_finite_number(s.value):
                    watcher.set_gauge(name=name, tags=tags, value=s.value, measurement_ts=now_ns)

            elif group.type == 'counter':
                s = samples.get(f'{name}_total') or samples.get(name)
                if s is not None and is_finite_number(s.value):
                    watcher.set_counter(name=name, tags=tags, total=s.value, measurement_ts=now_ns)

            elif group.type in ('histogram', 'summary'):
                # Both families land as ONE histogram. A Prometheus summary
                # has _count/_sum and quantiles we do not keep, so it
                # arrives with no bins; a histogram adds its `le` buckets on
                # top of the same totals. One name, one type, whichever the
                # engine happens to expose.
                c = samples.get(f'{name}_count')
                su = samples.get(f'{name}_sum')
                if c is not None and su is not None and is_finite_number(c.value) and is_finite_number(su.value):
                    bins, counts = buckets_to_bins(group.buckets)
                    watcher.set_histogram(name=name, tags=tags,
                                          bins=bins, counts=counts,
                                          count=int(c.value), sum_val=su.value,
                                          measurement_ts=now_ns)
