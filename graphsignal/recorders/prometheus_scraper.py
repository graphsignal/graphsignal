"""Shared Prometheus scraping: endpoint building, fetch with detect-and-backoff,
and the family walk. Recorders that read a Prometheus endpoint (the engine's
own, dcgm-exporter) build on this and keep only their emit policy."""

import logging
import math
import time
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

try:
    from prometheus_client.parser import text_string_to_metric_families
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

logger = logging.getLogger('graphsignal')

INITIAL_DETECT_DELAY_SEC = 2.0
MAX_DETECT_DELAY_SEC = 60.0
DEFAULT_TIMEOUT_SEC = 2.0
DEFAULT_METRICS_PATH = '/metrics'
DEFAULT_METRICS_HOST = '127.0.0.1'

# Prometheus labels used only for exposition (buckets, multiprocess); not useful
# as Graphsignal metric tags and they explode cardinality if kept.
STRIP_LABELS = frozenset({'le', 'quantile', 'gt', 'pid'})


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def normalize_metrics_path(path: Optional[str]) -> str:
    normalized = (path or DEFAULT_METRICS_PATH).strip()
    if not normalized.startswith('/'):
        normalized = '/' + normalized
    return normalized


def format_metrics_host(host: Optional[str]) -> str:
    host = (host or DEFAULT_METRICS_HOST).strip()
    if ':' in host and not host.startswith('['):
        return f'[{host}]'
    return host


def build_metrics_endpoint(metrics_port: int, metrics_path: Optional[str] = None,
                           metrics_host: Optional[str] = None) -> str:
    path = normalize_metrics_path(metrics_path)
    host = format_metrics_host(metrics_host)
    return f'http://{host}:{int(metrics_port)}{path}'


class PrometheusScraper:
    """Fetches one known Prometheus endpoint, never any other socket.

    The first attempt waits `INITIAL_DETECT_DELAY_SEC` (the server may not be
    listening yet right after launch). Until a body that looks like Prometheus
    text has been seen, failures back off exponentially up to
    `MAX_DETECT_DELAY_SEC`. Once verified, the endpoint is fetched at most
    every `interval` seconds (0 = on every call); a failure drops it back to
    unverified and into backoff.
    """

    def __init__(self, endpoint: str, timeout: float = DEFAULT_TIMEOUT_SEC,
                 interval: float = 0.0,
                 clock: Callable[[], float] = time.time):
        self._endpoint = endpoint
        self._timeout = timeout
        self._interval = interval
        self._clock = clock
        self._verified = False
        self._next_attempt_ts = 0.0
        self._detect_delay_sec = INITIAL_DETECT_DELAY_SEC

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def verified(self) -> bool:
        return self._verified

    def start(self) -> None:
        self._next_attempt_ts = self._clock() + INITIAL_DETECT_DELAY_SEC

    def scrape(self) -> Optional[str]:
        """Returns a verified body when a fetch is due and succeeds, else None.
        Never raises."""
        if not PROMETHEUS_AVAILABLE or not HTTP_AVAILABLE:
            return None

        now = self._clock()
        if now < self._next_attempt_ts:
            return None

        try:
            body = self._fetch(self._endpoint)
        except Exception as exc:
            logger.debug('Failed to fetch %s: %s', self._endpoint, exc)
            # Server may still be starting up; back off and retry the same port.
            self._verified = False
            self._back_off(now)
            return None

        if not self._verified:
            if not looks_like_prometheus(body):
                self._back_off(now)
                return None
            self._verified = True
            logger.debug('Prometheus endpoint confirmed: %s', self._endpoint)

        self._next_attempt_ts = now + self._interval
        return body

    def _back_off(self, now: float) -> None:
        self._detect_delay_sec = min(self._detect_delay_sec * 2, MAX_DETECT_DELAY_SEC)
        self._next_attempt_ts = now + self._detect_delay_sec

    def _fetch(self, url: str) -> str:
        req = urllib.request.Request(url, headers={'Accept': 'text/plain'})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = resp.read()
            charset = resp.headers.get_content_charset() or 'utf-8'
            return data.decode(charset, errors='replace')


class MetricGroup(NamedTuple):
    """One label set of one metric family."""
    name: str
    type: str
    tags: Dict[str, str]
    # Sample name (`<name>`, `<name>_total`, `<name>_count`, `<name>_sum`, …)
    # → sample.
    samples: Dict[str, object]
    # (`le` bound string, cumulative value) pairs; None when the family has
    # no buckets for this label set.
    buckets: Optional[List[Tuple[str, float]]]


def iter_metric_groups(body: str) -> Iterator[MetricGroup]:
    """Walks the families of a Prometheus text body and yields one group per
    label set, with exposition-only labels stripped and `le` buckets
    collected separately."""
    for family in text_string_to_metric_families(body):
        name = family.name

        sample_groups = {}
        bucket_groups = {}
        for sample in family.samples:
            # SGLang GaugeHistogram emits one gauge per bucket (gt/le labels).
            # These are heatmap buckets, not scalar gauges — skip them.
            if 'gt' in sample.labels:
                continue
            labels = {k: v for k, v in sample.labels.items() if k not in STRIP_LABELS}
            group_key = frozenset(labels.items())
            if sample.name == f'{name}_bucket' and 'le' in sample.labels:
                bucket_groups.setdefault(group_key, []).append(
                    (sample.labels['le'], sample.value))
                continue
            sample_groups.setdefault(group_key, {})[sample.name] = sample

        for group_key, sample_map in sample_groups.items():
            yield MetricGroup(name=name, type=family.type, tags=dict(group_key),
                              samples=sample_map,
                              buckets=bucket_groups.get(group_key))


def buckets_to_bins(buckets):
    """Convert Prometheus cumulative `le` buckets to non-cumulative bins.

    Bin values are the buckets' upper bounds; the `+Inf` remainder is folded
    into a final bin at the largest finite bound. Returns (None, None) when
    no usable buckets exist.
    """
    if not buckets:
        return None, None

    finite = []
    inf_count = None
    for le, value in buckets:
        if not is_finite_number(value):
            continue
        try:
            bound = float(le)
        except (TypeError, ValueError):
            continue
        if math.isinf(bound):
            inf_count = value
            continue
        finite.append((bound, value))

    if not finite:
        return None, None
    finite.sort(key=lambda b: b[0])

    bins = []
    counts = []
    previous_cumulative = 0.0
    for bound, cumulative in finite:
        delta = cumulative - previous_cumulative
        previous_cumulative = cumulative
        if delta > 0:
            bins.append(bound)
            counts.append(int(delta))
    if inf_count is not None:
        remainder = inf_count - previous_cumulative
        if remainder > 0:
            if bins and bins[-1] == finite[-1][0]:
                counts[-1] += int(remainder)
            else:
                bins.append(finite[-1][0])
                counts.append(int(remainder))

    if not bins:
        return None, None
    return bins, counts


def looks_like_prometheus(body: str) -> bool:
    if not body:
        return False
    # OpenMetrics/Prometheus payloads begin with HELP/TYPE comments or metric samples.
    for line in body.splitlines():
        if not line:
            continue
        if line.startswith('# HELP') or line.startswith('# TYPE'):
            return True
        if line.startswith('#'):
            continue
        if ' ' in line and not line.startswith('<'):
            return True
        return False
    return False
