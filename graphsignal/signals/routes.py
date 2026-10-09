"""The watcher's local HTTP routes and the /signals payload.

`GET /signals` returns the payload built from the in-memory stores: metrics
(with per-type statistics), recent errors, resources, and the watcher's global
tags as a top-level `context` object. Values follow the convention:
null = not measured, 0 = measured zero.
"""

import http.server
import json
import logging
import math
import threading
import time

from graphsignal.signals import metrics as metrics_module
from graphsignal.version import __version__

logger = logging.getLogger('graphsignal')

DEFAULT_LISTEN_HOST = '127.0.0.1'
DEFAULT_LISTEN_PORT = 18259


def quantiles_from_bins(bins, counts, qs=(('p50', 0.5), ('p95', 0.95))):
    """Nearest-rank quantiles over cumulative histogram bins.

    Quantile values are bin values, so resolution is bounded by the writer's
    bin grid. Returns None when bins are absent or empty.
    """
    if not bins or not counts or len(bins) != len(counts):
        return None
    total_count = sum(counts)
    if not total_count:
        return None

    quantiles = {}
    for name, q in qs:
        rank = max(1, math.ceil(q * total_count))
        cumulative = 0
        value = bins[-1]
        for bin_value, count in zip(bins, counts):
            cumulative += count
            if cumulative >= rank:
                value = bin_value
                break
        quantiles[name] = value
    return quantiles


def _gauge_stats(dp):
    return {'value': dp.get('value')}


def _counter_stats(dp):
    return {'total': dp.get('total')}


def _histogram_stats(dp):
    """count/sum/min/max exactly as measured; mean, p50, p95 derived.

    The exact totals win over the bins for the mean: a bin-weighted mean takes
    every observation at its bin's lower bound, and the writer's own sum did
    not. Quantiles have no such alternative — they only exist where there are
    bins, and are null for a source that reports totals alone.
    """
    bins = dp.get('bins')
    counts = dp.get('counts')
    count = dp.get('count')
    total = dp.get('sum')
    mean = None
    if count is not None and total is not None:
        if count:
            mean = total / count
    elif bins and counts and len(bins) == len(counts):
        total_count = sum(counts)
        if total_count:
            mean = sum(b * c for b, c in zip(bins, counts)) / total_count
    quantiles = quantiles_from_bins(bins, counts) or {}
    has_bins = bool(bins and counts and len(bins) == len(counts))
    return {
        'count': count,
        'sum': total,
        'min': dp.get('min'),
        'max': dp.get('max'),
        'mean': mean,
        'p50': quantiles.get('p50'),
        'p95': quantiles.get('p95'),
        # The bins themselves, so a consumer that diffs two snapshots
        # (graphsignal-top) can compute quantiles over its own interval.
        # null when the source reported totals without buckets.
        'bins': list(bins) if has_bins else None,
        'counts': list(counts) if has_bins else None,
    }


def _profile_stats(dp):
    frames = dp.get('frames') or {}
    samples = dp.get('samples') or {}
    return {
        'frames': [
            {'name': name, 'value': value, 'samples': samples.get(name, 0)}
            for name, value in sorted(
                frames.items(), key=lambda kv: kv[1], reverse=True)],
    }


def build_payload():
    import graphsignal.watcher as gwatcher

    payload = {
        'profiler': {'version': __version__},
        'payload_ns': time.time_ns(),
    }

    if not gwatcher.is_configured():
        payload['start_ns'] = None
        payload['context'] = {}
        payload['metrics'] = []
        payload['errors'] = []
        payload['resources'] = []
        return payload

    watcher = gwatcher.watcher()

    # Counters are cumulative since the instance started; start_ns names the
    # accumulation base explicitly, like the platform's windowed snapshot does.
    payload['start_ns'] = watcher.start_ns()
    payload['context'] = watcher.tags()

    metric_records = []
    for metric in watcher.metric_store().export():
        if metric.datapoint is None:
            continue
        if metric.type == metrics_module.GAUGE:
            stats = _gauge_stats(metric.datapoint)
        elif metric.type == metrics_module.COUNTER:
            stats = _counter_stats(metric.datapoint)
        elif metric.type == metrics_module.HISTOGRAM:
            stats = _histogram_stats(metric.datapoint)
        else:
            stats = _profile_stats(metric.datapoint)
        metric_records.append({
            'name': metric.name,
            'type': metric.type,
            'tags': metric.tags,
            'stats': stats,
            'updated_ns': metric.datapoint.get('ts'),
        })
    payload['metrics'] = sorted(
        metric_records, key=lambda r: (r['name'], r['type'], sorted(r['tags'].items())))

    payload['errors'] = [
        {
            'level': entry.get('level'),
            'message': entry.get('message'),
            'exception': entry.get('exception'),
            'tags': entry.get('tags'),
            'log_ns': entry.get('ts'),
        }
        for entry in watcher.log_store().last_entries(min_level='warning', limit=10)]

    # The stores keep nanoseconds; resource lifetimes read fine at seconds
    # (the `_ts` suffix means seconds on the wire).
    payload['resources'] = [
        {**resource,
         'first_seen_ts': resource['first_seen_ts'] // 1_000_000_000,
         'last_seen_ts': resource['last_seen_ts'] // 1_000_000_000}
        for resource in watcher.resource_store().export()]

    return payload


class _SignalsRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/signals':
                self._respond_json(200, build_payload())
            elif self.path == '/health':
                self._respond_json(200, {'status': 'ok'})
            else:
                self._respond_json(404, {'error': 'not found'})
        except Exception:
            logger.error('Error handling signals request', exc_info=True)
            try:
                self._respond_json(500, {'error': 'internal error'})
            except Exception:
                pass

    def _respond_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logger.debug('signals endpoint: ' + format, *args)


class SignalsEndpoint:
    # A busy port at startup is usually the previous run's watcher finishing
    # its last tick, so the bind is retried for a while before giving up.
    BIND_RETRY_INTERVAL_S = 2.0
    BIND_RETRY_TIMEOUT_S = 120.0

    def __init__(self, host=DEFAULT_LISTEN_HOST, port=DEFAULT_LISTEN_PORT,
                 bind_retry_interval=None, bind_retry_timeout=None):
        self._host = str(host)
        self._port = int(port)
        self._server = None
        self._serve_thread = None
        self._retry_thread = None
        self._stopped = threading.Event()
        self._retry_interval = (self.BIND_RETRY_INTERVAL_S if bind_retry_interval is None
                                else float(bind_retry_interval))
        self._retry_timeout = (self.BIND_RETRY_TIMEOUT_S if bind_retry_timeout is None
                               else float(bind_retry_timeout))

    def host(self):
        return self._host

    def port(self):
        return self._port

    def is_running(self):
        return self._server is not None

    def setup(self):
        if self._bind():
            return
        if self._retry_timeout <= 0:
            logger.error('Signals endpoint failed to bind %s:%d', self._host, self._port)
            return
        logger.warning('Signals endpoint could not bind %s:%d; retrying for up to %.0fs',
                       self._host, self._port, self._retry_timeout)
        self._retry_thread = threading.Thread(target=self._retry_bind, daemon=True)
        self._retry_thread.start()

    def _bind(self):
        """One bind attempt; True when the endpoint is serving."""
        try:
            server = http.server.ThreadingHTTPServer(
                (self._host, self._port), _SignalsRequestHandler)
        except OSError as exc:
            logger.debug('Signals endpoint bind %s:%d: %s', self._host, self._port, exc)
            return False
        server.daemon_threads = True
        self._server = server
        self._serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._serve_thread.start()
        logger.debug('Signals endpoint listening on %s:%d', self._host, self._port)
        return True

    def _retry_bind(self):
        deadline = time.monotonic() + self._retry_timeout
        while not self._stopped.is_set() and time.monotonic() < deadline:
            if self._stopped.wait(self._retry_interval):
                return
            if self._bind():
                return
        if not self._stopped.is_set():
            logger.error('Signals endpoint failed to bind %s:%d after %.0fs; endpoint disabled',
                         self._host, self._port, self._retry_timeout)

    def shutdown(self):
        self._stopped.set()
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            logger.debug('Error shutting down signals endpoint', exc_info=True)
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=2.0)
            self._serve_thread = None
        self._server = None
