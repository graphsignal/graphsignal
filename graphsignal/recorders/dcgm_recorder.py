import collections
import logging
import re
import statistics
import time
from typing import Callable, Dict, Optional, Set

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.prometheus_scraper import (
    PrometheusScraper, is_finite_number, iter_metric_groups)

logger = logging.getLogger('graphsignal')

# dcgm-exporter refreshes every 30 s by default; scraping it every tick would
# re-read the same values. A short timeout, since the exporter may be remote.
SCRAPE_INTERVAL_SEC = 5.0
SCRAPE_TIMEOUT_SEC = 1.0

# Only the profiling (DCP) fields: NVMLRecorder already covers what the
# DCGM_FI_DEV_* fields report.
PROF_PREFIX = 'DCGM_FI_PROF_'

# The GPU UUID label: `UUID` by default; newer exporters make its
# name configurable, so a label whose value looks like a GPU or MIG UUID is
# accepted too.
_UUID_LABELS = ('UUID', 'uuid', 'gpu_uuid')
_UUID_VALUE_RE = re.compile(r'^(?:GPU|MIG)-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4,12})+$')
_MIG_INSTANCE_LABEL = 'GPU_I_ID'

# Graphics-engine activity, the field the refresh interval is estimated from:
# older exporters name it ..._ACTIVE, newer ones ..._UTIL_RATIO.
_STALENESS_FIELDS = frozenset({
    'DCGM_FI_PROF_GR_ENGINE_ACTIVE', 'DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO'})
STALENESS_CHANGES = 8
STALENESS_METRIC = 'dcgm_exporter_update_interval_nanoseconds'


def _uuid_of(labels: Dict[str, str]) -> Optional[str]:
    for name in _UUID_LABELS:
        value = labels.get(name)
        if value:
            return value
    for value in labels.values():
        if isinstance(value, str) and _UUID_VALUE_RE.match(value):
            return value
    return None


class DcgmRecorder(BaseRecorder):
    """Imports DCGM profiling metrics (DCGM_FI_PROF_*) from a dcgm-exporter
    endpoint the user named with `--dcgm-exporter`. Never probes for one.

    Kept as the exporter reports them — gauges stay gauges under their DCGM
    names, tagged `device.uuid` (and `mig.instance` on MIG) — for the GPUs
    this workload uses (the `device` resources NVMLRecorder writes; all GPUs
    when there are none). Two misconfigurations are reported once each as a
    warning, which lands in `/signals` errors: an exporter without profiling
    fields, and one that reports none of this workload's GPUs (a cluster
    Service that load-balances to another node's exporter).
    """

    def __init__(self, root_pid=None, pid=None, args=None,
                 exporter_url: Optional[str] = None,
                 clock_ns: Callable[[], int] = time.monotonic_ns):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._scraper: Optional[PrometheusScraper] = None
        if exporter_url:
            self._scraper = PrometheusScraper(
                exporter_url, timeout=SCRAPE_TIMEOUT_SEC, interval=SCRAPE_INTERVAL_SEC)
        self._clock_ns = clock_ns
        self._warned: Set[str] = set()
        # Per GPU: the last graphics-activity value seen, and when it changed.
        self._last_activity: Dict[str, float] = {}
        self._activity_changes: Dict[str, collections.deque] = {}

    def setup(self):
        if self._scraper is not None:
            self._scraper.start()

    def on_tick(self):
        if self._scraper is None:
            return
        body = self._scraper.scrape()
        if body is None:
            return
        try:
            self._parse_and_emit(body)
        except Exception as exc:
            logger.error('Failed to parse dcgm-exporter metrics: %s', exc, exc_info=True)

    def _workload_uuids(self) -> Optional[Set[str]]:
        uuids = set()
        for resource in graphsignal.watcher.watcher().resource_store().export():
            if resource.get('kind') == 'device':
                uuid = resource.get('tags', {}).get('device.uuid')
                if uuid:
                    uuids.add(uuid)
        return uuids or None

    def _parse_and_emit(self, body: str) -> None:
        watcher = graphsignal.watcher.watcher()
        ours = self._workload_uuids()
        now_ns = time.time_ns()
        now_mono_ns = self._clock_ns()

        seen_gpus = False
        seen_prof = False
        matched = False
        for group in iter_metric_groups(body):
            uuid = _uuid_of(group.tags)
            if uuid is not None:
                seen_gpus = True
            if not group.name.startswith(PROF_PREFIX):
                continue
            seen_prof = True
            if ours is not None and uuid not in ours:
                continue
            matched = True

            tags = {}
            if uuid is not None:
                tags['device.uuid'] = uuid
            mig_instance = group.tags.get(_MIG_INSTANCE_LABEL)
            if mig_instance not in (None, ''):
                tags['mig.instance'] = mig_instance

            if group.type == 'counter':
                sample = group.samples.get(f'{group.name}_total') or group.samples.get(group.name)
                if sample is not None and is_finite_number(sample.value):
                    watcher.set_counter(name=group.name, total=sample.value,
                                        measurement_ts=now_ns, tags=tags)
                continue
            if group.type not in ('gauge', 'unknown'):
                continue
            sample = group.samples.get(group.name)
            if sample is None or not is_finite_number(sample.value):
                continue
            watcher.set_gauge(name=group.name, value=sample.value,
                              measurement_ts=now_ns, tags=tags)
            if group.name in _STALENESS_FIELDS and uuid is not None and not mig_instance:
                self._observe_activity(uuid, sample.value, now_mono_ns)

        if not seen_prof:
            self._warn_once(
                'no_prof',
                'dcgm-exporter at %s exposes no DCGM_FI_PROF_* fields; enable '
                'the profiling (DCP) fields in its counters CSV',
                self._scraper.endpoint)
        elif ours is not None and not matched and not seen_gpus:
            self._warn_once(
                'no_uuid',
                'dcgm-exporter at %s labels its metrics with no GPU UUID, so '
                'they cannot be matched to this workload\'s GPUs',
                self._scraper.endpoint)
        elif ours is not None and not matched:
            self._warn_once(
                'wrong_node',
                'dcgm-exporter at %s reports none of this workload\'s GPUs (%s); '
                'point --dcgm-exporter at the exporter on the same node, not a '
                'cluster Service',
                self._scraper.endpoint, ', '.join(sorted(ours)))

        interval_ns = self._update_interval_ns()
        if interval_ns is not None:
            watcher.set_gauge(name=STALENESS_METRIC, value=interval_ns,
                              measurement_ts=now_ns)

    def _observe_activity(self, uuid: str, value: float, now_ns: int) -> None:
        last = self._last_activity.get(uuid)
        self._last_activity[uuid] = value
        if last is None or value == last:
            return
        changes = self._activity_changes.setdefault(
            uuid, collections.deque(maxlen=STALENESS_CHANGES + 1))
        changes.append(now_ns)

    def _update_interval_ns(self) -> Optional[int]:
        """Median time between observed changes of the graphics-activity
        field over the last STALENESS_CHANGES changes per GPU. None until two
        changes are seen: an idle GPU reads a constant 0."""
        intervals = []
        for changes in self._activity_changes.values():
            stamps = list(changes)
            intervals.extend(b - a for a, b in zip(stamps, stamps[1:]))
        if not intervals:
            return None
        return int(statistics.median(intervals))

    def _warn_once(self, key: str, message: str, *args) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)
