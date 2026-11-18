import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.engine_specs import engine_spec
from graphsignal.recorders.prometheus_scraper import (
    PrometheusScraper, build_metrics_endpoint, buckets_to_bins,
    format_metrics_host, is_finite_number, iter_metric_groups,
    normalize_metrics_path)
from graphsignal.recorders.log_recorder import console_segment_paths
from graphsignal.recorders.shm_files import read_regular_file
from graphsignal.watcher.pid_monitor import CREDENTIAL_NAME_RE, REDACTED

logger = logging.getLogger('graphsignal')

HEALTH_INTERVAL_SEC = 5.0
HEALTH_TIMEOUT_SEC = 2.0
DEFAULT_HEALTH_HOST = '127.0.0.1'

ENGINE_RESOURCE_KIND = 'engine'
ENGINE_INFO_TIMEOUT_SEC = 5.0
# The configuration does not change while the engine runs. The resource is
# re-recorded on this period only so that it is re-uploaded: the row stays
# inside the server's retention, and a first upload lost in an outage arrives.
ENGINE_INFO_REFRESH_SEC = 3600.0

# How long the engine resource waits, from the first healthy probe, for a
# scrape to bring the engine's config-info metrics. Past it the resource is
# recorded without them rather than not at all.
ENGINE_CONFIG_INFO_WAIT_SEC = 60.0
# Engines publish settings as the labels of an info gauge, valued 1:
# `vllm:cache_config_info`, `sglang:cache_config_info`. Any gauge named so is
# configuration, whichever the engine.
CONFIG_INFO_SUFFIX = 'config_info'

# Where the parsed start-up line goes in the configuration.
STARTUP_CONFIG_KEY = 'startup_config'
# Bytes read from one console segment per scan; the rest on later scans.
CONFIG_LINE_SCAN_BYTES = 4 * 1024 * 1024

# Runtime state, not configuration, and it changes while the engine runs:
# SGLang's per-scheduler copies of its state.
_VOLATILE_INFO_KEYS = frozenset({'internal_states'})
# A field of a `key=value, key=value` line starts with a name and `=`.
_CONFIG_FIELD_START_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*=')
_INT_RE = re.compile(r'^-?\d+$')
_FLOAT_RE = re.compile(r'^-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$')


def parse_config_line(text: str) -> Dict[str, object]:
    """`key=value, key=value, …` as a dict, as vLLM prints its configuration.

    A value may itself hold commas, inside brackets (`ObservabilityConfig(a=1,
    b=2)`, `{'mode': …}`) or quotes, so the line is split only on a `, ` at
    depth zero that is followed by the next `name=`. Plain values are typed
    (None, True and False, numbers, quoted strings unquoted); anything else,
    nested reprs included, stays the string it was printed as.
    """
    parts = []
    depth = 0
    quote = None
    start = 0
    i = 0
    while i < len(text):
        c = text[i]
        if quote is not None:
            if c == '\\':
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in '\'"':
            quote = c
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth = max(0, depth - 1)
        elif c == ',' and depth == 0 and text.startswith(' ', i + 1) \
                and _CONFIG_FIELD_START_RE.match(text, i + 2):
            parts.append(text[start:i])
            start = i + 2
            i = start
            continue
        i += 1
    parts.append(text[start:])

    fields = {}
    for part in parts:
        key, sep, value = part.strip().partition('=')
        if sep and key:
            fields[key] = _config_value(value.strip())
    return fields


def _config_value(value: str):
    if value == 'None':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in '\'"':
        return value[1:-1]
    return value
# `scheme://user:password@host`: the credentials part of a URL.
_URL_CREDENTIALS_RE = re.compile(r'\b([a-z][a-z0-9+.-]*://)[^/@\s]+@', re.IGNORECASE)

_STARTING, _UP, _DOWN = 'starting', 'up', 'down'


def redact_engine_info(value):
    """`value` with every credential replaced, at any depth.

    A key named like a credential (`api_key`, `admin_api_key`,
    `ssl_keyfile_password`, ...; the names the command line's redaction knows)
    has its value replaced when one is set; an unset one stays null, since
    "no key configured" is configuration, not a secret. The credentials part
    of any URL in a string is replaced too.
    """
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if CREDENTIAL_NAME_RE.match(str(key)) and item is not None and item != '':
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_engine_info(item)
        return redacted
    if isinstance(value, list):
        return [redact_engine_info(item) for item in value]
    if isinstance(value, str):
        return _URL_CREDENTIALS_RE.sub(r'\1' + REDACTED + '@', value)
    return value


def build_engine_resource(
        engine_name: Optional[str], info,
        config_info: Optional[Dict[str, Dict[str, str]]] = None,
        startup_config: Optional[Dict[str, object]] = None,
) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    """The engine resource's (tags, attributes) from the engine's info answer,
    or None when the answer is not a JSON object.

    `config_info` maps a config-info metric's name to its labels; each one is
    added to the configuration under that name. The names carry the engine's
    metric prefix (`vllm:`), so they cannot collide with the info answer's keys.
    `startup_config`, the parsed start-up line, is added under
    `STARTUP_CONFIG_KEY`.

    The name and version are tags, beside the run's own (`instance.id`), so
    the resource's identity is the instance and its engine: a restart on new
    flags replaces the configuration rather than adding a resource. The
    configuration is one attribute, serialised with sorted keys.
    """
    if not isinstance(info, dict):
        return None
    config = {key: item for key, item in info.items() if key not in _VOLATILE_INFO_KEYS}
    for name, labels in (config_info or {}).items():
        config[name] = dict(labels)
    if startup_config is not None:
        config[STARTUP_CONFIG_KEY] = dict(startup_config)
    config = redact_engine_info(config)
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'), default=str)
    tags = {}
    if engine_name:
        tags['engine.name'] = engine_name
    version = config.get('version')
    if isinstance(version, str) and version:
        tags['engine.version'] = version
    return tags, {'engine.config': payload}


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

    The launcher names the engine (`engine`); where that engine publishes its
    version and configuration is looked up in `engine_specs.py`. Once the
    engine is first healthy, the same thread reads them from the spec's
    `info_path` on the probe's host and port, and records them as the
    `engine` resource, credentials redacted here, before
    anything leaves the process. A 4xx answer (no such route, or an engine
    started with an API key) ends the attempts for the run; any other failure
    is retried on the next healthy probe.

    The labels of the engine's config-info metrics, seen by the scrape, are
    part of the configuration: they are vLLM's resolved cache settings. So is
    the console line that the spec's `config_line_marker` marks (vLLM's start-up
    summary), read from the supervisor's console segments and parsed. The
    resource waits for both, up to `ENGINE_CONFIG_INFO_WAIT_SEC`, and is built
    once, never rebuilt: what is recorded is the whole configuration from the
    first upload on.
    """

    def __init__(self, root_pid=None, pid=None, args=None, metrics_port=None,
                 metrics_path: Optional[str] = None,
                 metrics_host: Optional[str] = None,
                 health_port: Optional[int] = None,
                 health_path: Optional[str] = None,
                 health_host: Optional[str] = None,
                 health_interval_sec: float = HEALTH_INTERVAL_SEC,
                 health_timeout_sec: float = HEALTH_TIMEOUT_SEC,
                 engine: Optional[str] = None,
                 engine_info_refresh_sec: float = ENGINE_INFO_REFRESH_SEC,
                 engine_config_info_wait_sec: float = ENGINE_CONFIG_INFO_WAIT_SEC,
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

        # The engine the launcher named; None, or a name with no spec, leaves
        # the engine resource off.
        spec = engine_spec(engine)
        self._engine_name = spec.name if spec else None
        self._engine_info_url: Optional[str] = None
        if health_port is not None and spec is not None:
            host = format_metrics_host(health_host or DEFAULT_HEALTH_HOST)
            self._engine_info_url = (
                f'http://{host}:{int(health_port)}{normalize_metrics_path(spec.info_path)}')
        self._engine_info_refresh_ns = int(engine_info_refresh_sec * 1e9)
        self._engine_config_info_wait_ns = int(engine_config_info_wait_sec * 1e9)
        # The info answer, once fetched; the resource, once built from it.
        self._engine_info: Optional[dict] = None
        self._engine_resource: Optional[Tuple[Dict[str, str], Dict[str, str]]] = None
        self._engine_resource_due_ns = 0
        self._engine_config_info_deadline_ns: Optional[int] = None
        self._engine_info_done = False
        # Written by the scrape (the tick thread), read by the health thread.
        self._config_info_lock = threading.Lock()
        self._config_info: Dict[str, Dict[str, str]] = {}
        self._config_info_scraped = False
        # Only the health thread touches these.
        self._config_line_marker = spec.config_line_marker if spec else None
        self._config_line_offsets: Dict[str, int] = {}
        self._startup_config: Optional[Dict[str, object]] = None

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
            return
        with self._config_info_lock:
            self._config_info_scraped = True

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
            if healthy:
                self._record_engine_resource()

    def _record_engine_resource(self):
        """Fetch the engine's info once, then re-record it every refresh
        period. Runs on the health thread, after a healthy probe. Never raises."""
        if self._engine_info_url is None or self._engine_info_done:
            return
        try:
            now_ns = self._clock_ns()
            if self._engine_resource is None:
                if self._engine_info is None:
                    self._engine_info = self._fetch_engine_info()
                    if self._engine_info is None:
                        return
                if self._config_line_marker and self._startup_config is None:
                    self._startup_config = self._find_startup_config()
                if not self._config_complete():
                    if self._engine_config_info_deadline_ns is None:
                        self._engine_config_info_deadline_ns = (
                            now_ns + self._engine_config_info_wait_ns)
                    if now_ns < self._engine_config_info_deadline_ns:
                        return
                with self._config_info_lock:
                    config_info = dict(self._config_info)
                self._engine_resource = build_engine_resource(
                    self._engine_name, self._engine_info, config_info,
                    self._startup_config)
            elif now_ns < self._engine_resource_due_ns:
                return
            self._engine_resource_due_ns = now_ns + self._engine_info_refresh_ns
            if not graphsignal.watcher.is_configured():
                return
            tags, attributes = self._engine_resource
            graphsignal.watcher.watcher().update_resource(
                ENGINE_RESOURCE_KIND, tags=tags, attributes=attributes)
        except Exception:
            logger.error('Failed to record the engine resource', exc_info=True)

    def _config_complete(self) -> bool:
        """Every source of configuration this recorder has has been read."""
        if self._scraper is not None:
            with self._config_info_lock:
                if not self._config_info_scraped:
                    return False
        if self._config_line_marker and self._startup_config is None:
            return False
        return True

    def _find_startup_config(self) -> Optional[Dict[str, object]]:
        """The parsed console line that the marker marks, or None while it has
        not been printed. Reads only what earlier scans have not."""
        marker = self._config_line_marker
        needle = marker.encode('utf-8')
        for path in console_segment_paths(self.pid):
            offset = self._config_line_offsets.get(path, 0)
            data = read_regular_file(path, CONFIG_LINE_SCAN_BYTES, offset)
            if not data:
                continue
            # Complete records only; a partly written one is read next time.
            end = data.rfind(b'\n')
            if end < 0:
                continue
            self._config_line_offsets[path] = offset + end + 1
            for raw in data[:end].split(b'\n'):
                if needle not in raw:
                    continue
                try:
                    line = json.loads(raw).get('line')
                except (ValueError, AttributeError):
                    continue
                if isinstance(line, str) and marker in line:
                    return parse_config_line(line.split(marker, 1)[1])
        return None

    def _fetch_engine_info(self) -> Optional[dict]:
        """The info endpoint's JSON object; None when not (yet) available.
        Ends the attempts on an answer that will not change."""
        url = self._engine_info_url
        try:
            req = urllib.request.Request(url, method='GET',
                                         headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=ENGINE_INFO_TIMEOUT_SEC) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                logger.debug('Engine info %s answered %s; not recording the engine '
                             'resource', url, exc.code)
                self._engine_info_done = True
            else:
                logger.debug('Engine info %s failed: %s', url, exc)
            return None
        except Exception as exc:
            logger.debug('Engine info %s failed: %s', url, exc)
            return None
        try:
            info = json.loads(body)
        except ValueError:
            logger.debug('Engine info %s is not JSON; not recording the engine '
                         'resource', url)
            self._engine_info_done = True
            return None
        if not isinstance(info, dict):
            logger.debug('Engine info %s is not a JSON object; not recording the '
                         'engine resource', url)
            self._engine_info_done = True
            return None
        return info

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
                if name.endswith(CONFIG_INFO_SUFFIX):
                    # Labels as the engine sent them, before any tag cap.
                    # A data-parallel engine sends one series per engine,
                    # alike but for its index: the first is kept.
                    with self._config_info_lock:
                        self._config_info.setdefault(name, dict(tags))

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
