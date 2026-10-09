"""Derived state for the top view.

`/signals` is a cumulative snapshot: counters are totals since the watcher
started, profiles hold cumulative time per frame, histograms cumulative
bins. `TopState` keeps the snapshots of the last `window` seconds and derives
everything the screen shows as a rate over that window — tokens per second,
kernel time per second, interval latency quantiles — the same way the
collector derives its deltas. `cumulative=True` on the views returns the raw
totals instead; the details overlay shows both for one row.

Everything here is terminal-free and deterministic given the payloads, so
the whole view can be tested from fixtures and printed once with `--once`.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from graphsignal.signals.routes import quantiles_from_bins
from graphsignal.top.engines import ENGINES, detect_engine
from graphsignal.top.kernels import KernelName, graph_name, readable_name

NS_PER_S = 1_000_000_000

GAUGE = 'gauge'
COUNTER = 'counter'
HISTOGRAM = 'histogram'
PROFILE = 'profile'

# Profile metrics behind each table tab: one or more groups, each with its
# CUDA and ROCm spellings (the watcher only ever carries one family, so the
# first present name of a group wins). The memcpy tab folds memsets in as a
# second group — they are rare and tiny, not worth a tab of their own.
TAB_PROFILES = {
    'kernels': (('cuda_kernels_nanoseconds', 'rocm_kernels_nanoseconds'),),
    'graphs': (('cuda_graphs_nanoseconds',),),
    'memcpy': (('cuda_memcpy_nanoseconds', 'rocm_memcpy_nanoseconds'),
               ('cuda_memset_nanoseconds',)),
    'sync': (('cuda_sync_nanoseconds', 'rocm_sync_nanoseconds'),),
}
# Byte counters matching the groups above, by frame kind.
TAB_BYTES = {
    'memcpy': (('cuda_memcpy_bytes', 'rocm_memcpy_bytes'), ('cuda_memset_bytes',)),
}
# Category label per (tab, group index) for tabs whose frames are kinds.
TAB_GROUP_CATEGORY = {('memcpy', 0): 'memcpy', ('memcpy', 1): 'memset', ('sync', 0): 'sync'}
TABS = ('kernels', 'graphs', 'memcpy', 'sync')
# The GPU-time breakdown keeps memset as its own share.
GPU_TIME_PROFILES = {
    'kernels': ('cuda_kernels_nanoseconds', 'rocm_kernels_nanoseconds'),
    'graphs': ('cuda_graphs_nanoseconds',),
    'memcpy': ('cuda_memcpy_nanoseconds', 'rocm_memcpy_nanoseconds'),
    'memset': ('cuda_memset_nanoseconds',),
    'sync': ('cuda_sync_nanoseconds', 'rocm_sync_nanoseconds'),
}


def _tag_key(tags):
    return tuple(sorted((tags or {}).items()))


class Snapshot:
    """One `/signals` payload, indexed by (name, type)."""

    def __init__(self, payload, received_ts=None):
        self.payload = payload or {}
        payload_ns = self.payload.get('payload_ns')
        self.ts = (payload_ns / NS_PER_S) if payload_ns else (received_ts or 0.0)
        self.start_ns = self.payload.get('start_ns')
        self._metrics = {}
        for record in self.payload.get('metrics') or []:
            name = record.get('name')
            mtype = record.get('type')
            if not name or not mtype:
                continue
            self._metrics.setdefault((name, mtype), []).append(record)

    def names(self):
        return [name for name, _ in self._metrics.keys()]

    def records(self, name, mtype):
        return self._metrics.get((name, mtype), [])

    def has(self, name, mtype):
        return (name, mtype) in self._metrics

    def gauge_sum(self, name):
        values = [r['stats'].get('value') for r in self.records(name, GAUGE)]
        values = [v for v in values if v is not None]
        return sum(values) if values else None

    def gauge_by_tag(self, name, tag):
        out = {}
        for r in self.records(name, GAUGE):
            key = (r.get('tags') or {}).get(tag)
            value = r['stats'].get('value')
            if key is not None and value is not None:
                out[key] = value
        return out

    def counter_sum(self, name):
        values = [r['stats'].get('total') for r in self.records(name, COUNTER)]
        values = [v for v in values if v is not None]
        return sum(values) if values else None

    def counter_updated(self, name):
        """Newest `updated_ns` among a counter's records: when the source
        last measured it, which for a scraped engine metric is the scrape
        time rather than the poll time."""
        stamps = [r.get('updated_ns') for r in self.records(name, COUNTER)]
        stamps = [t for t in stamps if t]
        return max(stamps) if stamps else None

    def counter_by_tag(self, name, tag):
        out = {}
        for r in self.records(name, COUNTER):
            key = (r.get('tags') or {}).get(tag)
            value = r['stats'].get('total')
            if key is not None and value is not None:
                out[key] = out.get(key, 0) + value
        return out

    def histogram_merged(self, name):
        """Bins, count and sum of a histogram family merged across tag sets.

        Returns (bins: {bin_value: count}, count, sum) or None when absent.
        Each half may be None when no source reported it.
        """
        records = self.records(name, HISTOGRAM)
        if not records:
            return None
        bins = {}
        count = None
        total = None
        for r in records:
            stats = r['stats']
            # The endpoint reports bins/counts under the datapoint keys only
            # when the writer supplied them; the stats carry the quantiles.
            dp_bins = r.get('bins') or stats.get('bins')
            dp_counts = r.get('counts') or stats.get('counts')
            if dp_bins and dp_counts and len(dp_bins) == len(dp_counts):
                for b, c in zip(dp_bins, dp_counts):
                    bins[b] = bins.get(b, 0) + (c or 0)
            if stats.get('count') is not None:
                count = (count or 0) + stats['count']
            if stats.get('sum') is not None:
                total = (total or 0) + stats['sum']
        return bins, count, total

    def profile_frames(self, name):
        """{frame: (value, samples)} summed across processes, and the
        per-process map {pid: {frame: (value, samples)}}."""
        merged = {}
        per_process = {}
        for r in self.records(name, PROFILE):
            pid = (r.get('tags') or {}).get('process.pid', '?')
            frames = per_process.setdefault(pid, {})
            for frame in r['stats'].get('frames') or []:
                fname = frame.get('name')
                if fname is None:
                    continue
                value = frame.get('value') or 0
                samples = frame.get('samples') or 0
                v, s = merged.get(fname, (0, 0))
                merged[fname] = (v + value, s + samples)
                v, s = frames.get(fname, (0, 0))
                frames[fname] = (v + value, s + samples)
        return merged, per_process


@dataclass
class DeviceView:
    index: int
    uuid: str
    name: Optional[str]
    utilization: Optional[float]
    utilization_history: List[Optional[float]]
    memory_used: Optional[float]
    memory_total: Optional[float]
    memory_utilization: Optional[float]
    power: Optional[float]
    power_limit: Optional[float]
    temperature: Optional[float]
    sm_clock_mhz: Optional[float]
    pcie_rx_bytes_per_s: Optional[float]
    pcie_tx_bytes_per_s: Optional[float]
    xid_errors: Optional[float]
    throttle_reasons: Optional[float]
    throttle_history: List[bool]

    @property
    def memory_fraction(self):
        if self.memory_used is None or not self.memory_total:
            return None
        return self.memory_used / self.memory_total

    @property
    def throttled(self):
        return bool(self.throttle_reasons)


@dataclass
class HistogramView:
    p50: Optional[float] = None
    p95: Optional[float] = None
    mean: Optional[float] = None
    count: Optional[float] = None


@dataclass
class EngineView:
    engine: str
    running: Optional[float] = None
    waiting: Optional[float] = None
    kv_cache_pct: Optional[float] = None
    prefix_hit_pct: Optional[float] = None
    prompt_tokens_per_s: Optional[float] = None
    gen_tokens_per_s: Optional[float] = None
    prompt_history: List[Optional[float]] = field(default_factory=list)
    gen_history: List[Optional[float]] = field(default_factory=list)
    ttft: HistogramView = field(default_factory=HistogramView)
    tpot: HistogramView = field(default_factory=HistogramView)
    e2e: HistogramView = field(default_factory=HistogramView)


@dataclass
class GpuTimeView:
    kernels: Optional[float] = None
    graphs: Optional[float] = None
    memcpy: Optional[float] = None
    memset: Optional[float] = None
    sync: Optional[float] = None
    idle: Optional[float] = None
    available: bool = False


@dataclass
class FrameRow:
    key: str                # frame name as reported (symbol, hash, kind)
    category: str
    label: str
    short: str
    raw: str
    value_ns: float         # over the window, or cumulative
    samples: float
    share_pct: Optional[float]
    ns_per_s: Optional[float]
    calls_per_s: Optional[float]
    avg_ns: Optional[float]
    bytes_per_s: Optional[float] = None
    per_process: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    kernel: Optional[KernelName] = None


@dataclass
class HeaderView:
    endpoint: str = ''
    command_line: Optional[str] = None
    pid: Optional[str] = None
    host: Optional[str] = None
    uptime_s: Optional[float] = None
    profiler_version: Optional[str] = None
    graph_trace_mode: Optional[str] = None  # 'graph' | 'node' | None
    window_s: Optional[float] = None
    unavailable: Optional[str] = None
    rocm: bool = False


class TopState:
    def __init__(self, window=5.0, history=600, endpoint=''):
        self.window = float(window)
        self.history = int(history)
        self.endpoint = endpoint
        self._snaps = deque()
        self._series = {}
        self._series_ts = deque(maxlen=self.history)
        self.latest = None
        self.base = None
        self.unavailable = None
        self._kernel_names = {}

    # ------------------------------------------------------------ updates

    def update(self, payload, received_ts=None):
        snap = Snapshot(payload, received_ts)
        if self._snaps and abs((snap.start_ns or 0) - (self._snaps[-1].start_ns or 0)) >= NS_PER_S:
            # A different watcher run: every cumulative base is void.
            self._snaps.clear()
            self._series.clear()
            self._series_ts.clear()
        if self._snaps and snap.ts <= self._snaps[-1].ts:
            # Same payload served twice (paused watcher, cached proxy): no
            # new information, and a zero-length window would divide by zero.
            self.unavailable = None
            return
        self._snaps.append(snap)
        while len(self._snaps) >= 2 and snap.ts - self._snaps[1].ts >= self.window:
            self._snaps.popleft()
        self.latest = snap
        self.base = self._snaps[0] if len(self._snaps) > 1 else None
        self.unavailable = None
        self._series_ts.append(snap.ts)
        self._push_series()

    def mark_unavailable(self, reason):
        self.unavailable = str(reason)

    @property
    def dt(self):
        """Seconds between the window base and the latest snapshot."""
        if self.latest is None or self.base is None:
            return None
        return self.latest.ts - self.base.ts

    def series(self, key):
        return list(self._series.get(key, ()))

    def series_span(self, width):
        """Seconds covered by the last `width` samples of any series."""
        if len(self._series_ts) < 2:
            return None
        ts = list(self._series_ts)[-max(2, int(width)):]
        return ts[-1] - ts[0]

    def _push(self, key, value):
        self._series.setdefault(key, deque(maxlen=self.history)).append(value)

    def _push_series(self):
        snap = self.latest
        for uuid, value in snap.gauge_by_tag('gpu_utilization_percent', 'device.uuid').items():
            self._push(f'gpu.util.{uuid}', value)
        throttle = snap.gauge_by_tag('gpu_clocks_throttle_reasons', 'device.uuid')
        for uuid in self._device_uuids(snap):
            if f'gpu.util.{uuid}' not in self._series:
                self._push(f'gpu.util.{uuid}', None)
            self._push(f'gpu.throttle.{uuid}', bool(throttle.get(uuid)))
        engine = self._engine_table()
        if engine is not None:
            # Per-poll token rates, so the sparkline keeps its 1-sample resolution.
            prev = self._snaps[-2] if len(self._snaps) > 1 else None
            self._push('engine.prompt_tps', self._poll_rate(engine, 'prompt_tokens', prev, snap))
            self._push('engine.gen_tps', self._poll_rate(engine, 'gen_tokens', prev, snap))

    def _poll_rate(self, table, field_name, prev, snap):
        name = self._first_present(snap, table[field_name], COUNTER)
        if name is None or prev is None:
            if name is None and field_name == 'gen_tokens' and table.get('gen_tps_gauge'):
                return self._first_gauge(snap, table['gen_tps_gauge'])
            return None
        now = snap.counter_sum(name)
        before = prev.counter_sum(name)
        if now is None or before is None:
            return None
        # Rate over the source's own measurement interval. A scraped counter
        # advances only when the engine is scraped, so a poll that saw no new
        # scrape repeats the last rate instead of charting a zero followed by
        # a double-sized spike on the next one.
        updated_now = snap.counter_updated(name)
        updated_before = prev.counter_updated(name)
        if updated_now and updated_before:
            if updated_now == updated_before:
                history = self._series.get(self._series_key(field_name))
                return history[-1] if history else None
            dt = (updated_now - updated_before) / NS_PER_S
        else:
            dt = snap.ts - prev.ts
        if dt <= 0:
            return None
        return max(0.0, (now - before) / dt)

    @staticmethod
    def _series_key(field_name):
        return 'engine.prompt_tps' if field_name == 'prompt_tokens' else 'engine.gen_tps'

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _first_present(snap, names, mtype):
        for name in names or ():
            if snap.has(name, mtype):
                return name
        return None

    def _first_gauge(self, snap, names):
        name = self._first_present(snap, names, GAUGE)
        return snap.gauge_sum(name) if name else None

    def _engine_table(self):
        if self.latest is None:
            return None
        engine = detect_engine(self.latest.names())
        return ENGINES.get(engine) if engine else None

    def _counter_rate(self, names, cumulative=False):
        snap = self.latest
        name = self._first_present(snap, names, COUNTER)
        if name is None:
            return None
        now = snap.counter_sum(name)
        if cumulative:
            return now
        if self.base is None or not self.dt:
            return None
        before = self.base.counter_sum(name)
        if now is None or before is None:
            return None
        dt = self.dt
        updated_now = snap.counter_updated(name)
        updated_before = self.base.counter_updated(name)
        if updated_now and updated_before and updated_now > updated_before:
            dt = (updated_now - updated_before) / NS_PER_S
        return max(0.0, (now - before) / dt)

    def _histogram_view(self, names, cumulative=False):
        snap = self.latest
        name = self._first_present(snap, names, HISTOGRAM)
        if name is None:
            return HistogramView()
        now = snap.histogram_merged(name)
        if now is None:
            return HistogramView()
        bins_now, count_now, sum_now = now
        if cumulative or self.base is None:
            if cumulative:
                bins, count, total = bins_now, count_now, sum_now
            else:
                return HistogramView()
        else:
            before = self.base.histogram_merged(name)
            bins_before, count_before, sum_before = before if before else ({}, None, None)
            bins = {b: max(0, c - bins_before.get(b, 0)) for b, c in bins_now.items()}
            count = (count_now - count_before) if (count_now is not None and count_before is not None) else None
            total = (sum_now - sum_before) if (sum_now is not None and sum_before is not None) else None
        view = HistogramView(count=count)
        if count and total is not None:
            view.mean = total / count
        if bins:
            ordered = sorted(bins.items())
            quantiles = quantiles_from_bins([b for b, _ in ordered], [c for _, c in ordered]) or {}
            view.p50 = quantiles.get('p50')
            view.p95 = quantiles.get('p95')
        return view

    @staticmethod
    def _device_uuids(snap):
        uuids = set()
        for name in ('gpu_utilization_percent', 'gpu_memory_used_bytes', 'gpu_memory_total_bytes',
                     'gpu_power_usage_watts', 'gpu_temperature_celsius'):
            uuids.update(snap.gauge_by_tag(name, 'device.uuid').keys())
        for resource in snap.payload.get('resources') or []:
            if resource.get('kind') == 'device':
                uuid = (resource.get('tags') or {}).get('device.uuid')
                if uuid:
                    uuids.add(uuid)
        return uuids

    def _profile_names(self, tab):
        """[(group index, metric name)] present for a tab."""
        if self.latest is None:
            return []
        out = []
        for index, group in enumerate(TAB_PROFILES.get(tab, ())):
            name = self._first_present(self.latest, group, PROFILE)
            if name is not None:
                out.append((index, name))
        return out

    def device_count(self):
        if self.latest is None:
            return 0
        return len(self._device_uuids(self.latest))

    # ------------------------------------------------------------ views

    def header(self):
        view = HeaderView(endpoint=self.endpoint, unavailable=self.unavailable,
                          window_s=self.dt)
        snap = self.latest
        if snap is None:
            return view
        payload = snap.payload
        view.profiler_version = (payload.get('profiler') or {}).get('version')
        view.host = (payload.get('context') or {}).get('host.name')
        if snap.start_ns and payload.get('payload_ns'):
            view.uptime_s = (payload['payload_ns'] - snap.start_ns) / NS_PER_S
        mode = snap.gauge_sum('cuda_graph_trace_mode')
        if mode is not None:
            view.graph_trace_mode = 'node' if mode >= 1 else 'graph'
        view.rocm = any(name.startswith('rocm_') for name in snap.names())

        root = None
        for resource in payload.get('resources') or []:
            if resource.get('kind') != 'process':
                continue
            tags = resource.get('tags') or {}
            pid = tags.get('process.pid')
            root_pid = tags.get('process.root_pid')
            if root_pid and root_pid != pid:
                continue
            if root is None or resource.get('first_seen_ts', 0) < root.get('first_seen_ts', 0):
                root = resource
        if root is not None:
            view.pid = (root.get('tags') or {}).get('process.pid')
            view.command_line = (root.get('attributes') or {}).get('process.command_line')
        return view

    def devices(self):
        snap = self.latest
        if snap is None:
            return []
        names = {}
        for resource in snap.payload.get('resources') or []:
            if resource.get('kind') == 'device':
                uuid = (resource.get('tags') or {}).get('device.uuid')
                if uuid:
                    names[uuid] = (resource.get('attributes') or {}).get('device.name')
        bus_ids = {}
        for r in snap.records('gpu_utilization_percent', GAUGE) + snap.records('gpu_memory_total_bytes', GAUGE):
            tags = r.get('tags') or {}
            if tags.get('device.uuid'):
                bus_ids[tags['device.uuid']] = tags.get('device.bus_id', '')

        def g(name):
            return snap.gauge_by_tag(name, 'device.uuid')

        util = g('gpu_utilization_percent')
        mem_used = g('gpu_memory_used_bytes')
        mem_total = g('gpu_memory_total_bytes')
        mem_util = g('gpu_memory_utilization_percent')
        power = g('gpu_power_usage_watts')
        power_limit = g('gpu_power_management_limit_watts')
        temp = g('gpu_temperature_celsius')
        sm_clock = g('gpu_clock_sm_megahertz')
        pcie_rx = g('gpu_pcie_throughput_rx_kilobytes_per_second')
        pcie_tx = g('gpu_pcie_throughput_tx_kilobytes_per_second')
        throttle = g('gpu_clocks_throttle_reasons')
        xid = snap.counter_by_tag('gpu_xid_critical_errors', 'device.uuid')

        uuids = sorted(self._device_uuids(snap), key=lambda u: (bus_ids.get(u, ''), u))
        views = []
        for index, uuid in enumerate(uuids):
            views.append(DeviceView(
                index=index, uuid=uuid, name=names.get(uuid),
                utilization=util.get(uuid),
                utilization_history=self.series(f'gpu.util.{uuid}'),
                memory_used=mem_used.get(uuid), memory_total=mem_total.get(uuid),
                memory_utilization=mem_util.get(uuid),
                power=power.get(uuid), power_limit=power_limit.get(uuid),
                temperature=temp.get(uuid), sm_clock_mhz=sm_clock.get(uuid),
                pcie_rx_bytes_per_s=(pcie_rx[uuid] * 1000.0) if uuid in pcie_rx else None,
                pcie_tx_bytes_per_s=(pcie_tx[uuid] * 1000.0) if uuid in pcie_tx else None,
                xid_errors=xid.get(uuid),
                throttle_reasons=throttle.get(uuid),
                throttle_history=self.series(f'gpu.throttle.{uuid}'),
            ))
        return views

    def engine(self, cumulative=False):
        table = self._engine_table()
        if table is None:
            return None
        snap = self.latest
        engine = detect_engine(snap.names())
        view = EngineView(engine=engine)
        view.running = self._first_gauge(snap, table['running'])
        view.waiting = self._first_gauge(snap, table['waiting'])
        kv = self._first_gauge(snap, table['kv_cache'])
        if kv is not None:
            view.kv_cache_pct = kv * 100.0 if kv <= 1.0 else kv
        hit = self._first_gauge(snap, table['prefix_hit'])
        if hit is not None:
            view.prefix_hit_pct = hit * 100.0 if hit <= 1.0 else hit
        elif table.get('prefix_hit_counters'):
            hits_name, queries_name = table['prefix_hit_counters']
            hits = self._counter_rate([hits_name], cumulative)
            queries = self._counter_rate([queries_name], cumulative)
            if hits is not None and queries:
                view.prefix_hit_pct = hits / queries * 100.0
        view.prompt_tokens_per_s = self._counter_rate(table['prompt_tokens'], cumulative)
        view.gen_tokens_per_s = self._counter_rate(table['gen_tokens'], cumulative)
        if view.gen_tokens_per_s is None and not cumulative and table.get('gen_tps_gauge'):
            view.gen_tokens_per_s = self._first_gauge(snap, table['gen_tps_gauge'])
        view.prompt_history = self.series('engine.prompt_tps')
        view.gen_history = self.series('engine.gen_tps')
        view.ttft = self._histogram_view(table['ttft'], cumulative)
        view.tpot = self._histogram_view(table['tpot'], cumulative)
        view.e2e = self._histogram_view(table['e2e'], cumulative)
        return view

    def _profile_delta(self, name):
        """{frame: (value delta, samples delta)}, per-process map, and the
        interval the deltas cover — or cumulative totals and the uptime when
        there is no base yet."""
        snap = self.latest
        now, per_process_now = snap.profile_frames(name)
        if self.base is None or not self.dt:
            return None, None, None
        before, per_process_before = self.base.profile_frames(name)
        delta = {}
        for frame, (v, s) in now.items():
            bv, bs = before.get(frame, (0, 0))
            delta[frame] = (max(0, v - bv), max(0, s - bs))
        per_process = {}
        for pid, frames in per_process_now.items():
            base_frames = per_process_before.get(pid, {})
            per_process[pid] = {
                frame: (max(0, v - base_frames.get(frame, (0, 0))[0]),
                        max(0, s - base_frames.get(frame, (0, 0))[1]))
                for frame, (v, s) in frames.items()}
        return delta, per_process, self.dt

    def _profile_cumulative(self, name):
        snap = self.latest
        now, per_process = snap.profile_frames(name)
        elapsed = None
        if snap.start_ns and snap.payload.get('payload_ns'):
            elapsed = (snap.payload['payload_ns'] - snap.start_ns) / NS_PER_S
        return now, per_process, elapsed

    def gpu_time(self, cumulative=False):
        view = GpuTimeView()
        if self.latest is None:
            return view
        devices = max(1, self.device_count())
        busy = {}
        any_profile = False
        for tab, names in GPU_TIME_PROFILES.items():
            name = self._first_present(self.latest, names, PROFILE)
            if name is None:
                continue
            any_profile = True
            if cumulative:
                frames, _, elapsed = self._profile_cumulative(name)
            else:
                frames, _, elapsed = self._profile_delta(name)
            if frames is None or not elapsed:
                continue
            total_ns = sum(v for v, _ in frames.values())
            busy[tab] = total_ns / (elapsed * NS_PER_S * devices) * 100.0
        if not any_profile or not busy:
            return view
        view.available = True
        view.kernels = busy.get('kernels')
        view.graphs = busy.get('graphs')
        view.memcpy = busy.get('memcpy')
        view.memset = busy.get('memset')
        view.sync = busy.get('sync')
        view.idle = max(0.0, 100.0 - sum(
            v for v in (view.kernels, view.graphs, view.memcpy, view.memset) if v))
        return view

    def _name_for(self, tab, group, frame):
        if tab == 'kernels':
            kernel = self._kernel_names.get(frame)
            if kernel is None:
                kernel = readable_name(frame)
                if len(self._kernel_names) < 20000:
                    self._kernel_names[frame] = kernel
            return kernel.category, kernel.label, kernel.short, kernel
        if tab == 'graphs':
            return 'graph', graph_name(frame), frame, None
        category = TAB_GROUP_CATEGORY.get((tab, group), tab)
        return category, f'{category} {str(frame).replace("_", " ")}', str(frame), None

    def _bytes_by_kind(self, tab, group, cumulative):
        groups = TAB_BYTES.get(tab, ())
        if group >= len(groups):
            return {}
        counter = self._first_present(self.latest, groups[group], COUNTER)
        if counter is None:
            return {}
        now = self.latest.counter_by_tag(counter, 'kind')
        if cumulative:
            return dict(now)
        if self.base is None or not self.dt:
            return {}
        before = self.base.counter_by_tag(counter, 'kind')
        return {k: max(0.0, v - before.get(k, 0)) / self.dt for k, v in now.items()}

    def table(self, tab='kernels', cumulative=False):
        """Rows for a tab, sorted by time descending. `cumulative` returns
        totals since start (the details overlay shows both)."""
        if self.latest is None:
            return []
        devices = max(1, self.device_count())
        rows = []
        for group, name in self._profile_names(tab):
            if cumulative:
                frames, per_process, elapsed = self._profile_cumulative(name)
            else:
                frames, per_process, elapsed = self._profile_delta(name)
            if frames is None:
                continue
            bytes_by_kind = self._bytes_by_kind(tab, group, cumulative)
            for frame, (value, samples) in frames.items():
                category, label, short, kernel = self._name_for(tab, group, frame)
                share = (value / (elapsed * NS_PER_S * devices) * 100.0) if elapsed else None
                rows.append(FrameRow(
                    key=f'{name}:{frame}', category=category, label=label, short=short,
                    raw=str(frame), value_ns=value, samples=samples, share_pct=share,
                    ns_per_s=(value / elapsed) if elapsed else None,
                    calls_per_s=(samples / elapsed) if elapsed else None,
                    avg_ns=(value / samples) if samples else None,
                    bytes_per_s=bytes_by_kind.get(frame),
                    per_process={pid: fr[frame] for pid, fr in (per_process or {}).items() if frame in fr},
                    kernel=kernel))
        rows.sort(key=lambda r: (-r.value_ns, r.label))
        return rows

    def has_tab(self, tab):
        return bool(self._profile_names(tab))

    def errors(self):
        if self.latest is None:
            return []
        return list(self.latest.payload.get('errors') or [])
