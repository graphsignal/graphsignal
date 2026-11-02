"""The metric store in the Prometheus text exposition format (`GET /metrics`).

A renderer over `MetricStore.export()`: the store already holds cumulative,
non-destructive snapshots, which is exactly what a scraper reads. Format 0.0.4,
written by hand rather than through a `prometheus_client` registry, so the
output is exact (no `_created` series) and nothing is constructed per request.

Mapping:
- gauge     → gauge `<name>`
- counter   → counter `<name>_total` (the suffix is added unless present)
- histogram → histogram `<name>_bucket{le}` / `_count` / `_sum`, plus gauges
              `<name>_min` / `<name>_max` when the source keeps the extremes
- profile   → counters `<name>_total` and `<name>_samples_total`, one series
              per frame, the frame in labels

Every series carries its metric's tags and the watcher's context tags as
labels, except `instance.id`, which goes on `graphsignal_profiler_info` alone —
a per-run id on every series would start a new series set on each restart.
"""

import functools
import logging
import math
import re

from graphsignal.signals import metrics as metrics_module
from graphsignal.signals.kernels import demangle_short, readable_name
from graphsignal.version import __version__

logger = logging.getLogger('graphsignal')

CONTENT_TYPE = 'text/plain; version=0.0.4; charset=utf-8'
INFO_METRIC = 'graphsignal_profiler_info'

# Context tags kept off the series; `instance.id` is on the info gauge.
EXCLUDED_CONTEXT_TAGS = frozenset({'instance.id'})

# Profiles whose frames are kernel symbols / CUDA graph signatures; every
# other profile's frames are plain names (memcpy kind, sync type, probe frame).
KERNEL_PROFILES = frozenset({'cuda_kernels_nanoseconds', 'rocm_kernels_nanoseconds'})
GRAPH_PROFILES = frozenset({'cuda_graphs_nanoseconds'})

_INVALID_NAME_CHARS = re.compile(r'[^a-zA-Z0-9_:]')
_INVALID_LABEL_CHARS = re.compile(r'[^a-zA-Z0-9_]')

def sanitize_name(name):
    """A valid metric name: `user.hist` → `user_hist`; `vllm:x` keeps its colon."""
    name = _INVALID_NAME_CHARS.sub('_', str(name))
    if not name or name[0].isdigit():
        name = '_' + name
    return name


def sanitize_label_name(key):
    """A valid label name: `process.pid` → `process_pid`. No colons, and the
    `__` prefix is Prometheus's own."""
    key = _INVALID_LABEL_CHARS.sub('_', str(key))
    if not key or key[0].isdigit():
        key = '_' + key
    if key.startswith('__'):
        key = 'tag' + key
    return key


def escape_label_value(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')


def escape_help(text):
    return str(text).replace('\\', '\\\\').replace('\n', '\\n')


def format_value(value):
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return '+Inf' if value > 0 else '-Inf'
    return repr(value)


@functools.lru_cache(maxsize=4096)
def _kernel_frame_labels(symbol):
    kernel = readable_name(symbol)
    return (
        ('kernel', demangle_short(symbol) or symbol),
        ('symbol', symbol),
        ('category', kernel.category),
        ('library', kernel.library or ''),
    )


def _frame_labels(profile_name, frame):
    if profile_name in KERNEL_PROFILES:
        return _kernel_frame_labels(frame)
    if profile_name in GRAPH_PROFILES:
        return (('graph', frame),)
    return (('frame', frame),)


def _label_set(tags, context, own=(), reserved=()):
    """Labels for one series: the series' own labels (`le`, frame labels), then
    the metric's tags, then the context tags. A tag whose label the series
    owns, or that `reserved` names (`le` on a histogram's `_count`, so every
    series of the histogram agrees), goes out as `exported_<label>`; on any
    other collision the first one wins, so a metric tag beats a context tag."""
    own = dict(own)
    reserved = set(reserved) | set(own)
    labels = {}
    for source in (tags, context):
        for key, value in source.items():
            label = sanitize_label_name(key)
            if label in reserved:
                label = 'exported_' + label
            if label not in labels:
                labels[label] = str(value)
    labels.update(own)
    return labels


def _family_names(family_name, family_type):
    """Every name a family puts on the wire, for collision checks."""
    if family_type == 'histogram':
        return (family_name, family_name + '_bucket', family_name + '_count',
                family_name + '_sum')
    if family_type == 'counter' and family_name.endswith('_total'):
        # Parsers file a counter under its name without `_total`.
        return (family_name, family_name[:-len('_total')])
    return (family_name,)


class _Exposition:
    def __init__(self):
        self._families = {}
        self._taken = {}

    def add(self, entries):
        """Add one metric's series, all or nothing.

        `entries` is a list of (family_name, family_type, help, samples), with
        samples as (sample_name, labels, value). A metric whose family name is
        taken by a family of another type, or whose series already exist, is
        skipped — the first one wins. A value that does not format raises
        before anything is added.
        """
        entries = [
            (family_name, family_type, help_text,
             [(sample_name, labels, format_value(value))
              for sample_name, labels, value in samples])
            for family_name, family_type, help_text, samples in entries]
        for family_name, family_type, _, samples in entries:
            family = self._families.get(family_name)
            if family is None:
                for name in _family_names(family_name, family_type):
                    if name in self._taken:
                        return False
            elif family['type'] != family_type:
                return False
            else:
                for sample_name, labels, _ in samples:
                    if (sample_name, frozenset(labels.items())) in family['keys']:
                        return False
        for family_name, family_type, help_text, samples in entries:
            family = self._families.get(family_name)
            if family is None:
                family = {'type': family_type, 'help': help_text,
                          'samples': [], 'keys': set()}
                self._families[family_name] = family
                for name in _family_names(family_name, family_type):
                    self._taken[name] = family_name
            for sample_name, labels, value in samples:
                family['keys'].add((sample_name, frozenset(labels.items())))
                family['samples'].append((sample_name, labels, value))
        return True

    def text(self):
        lines = []
        for family_name, family in self._families.items():
            lines.append(f'# HELP {family_name} {escape_help(family["help"])}')
            lines.append(f'# TYPE {family_name} {family["type"]}')
            for sample_name, labels, value in family['samples']:
                label_text = ','.join(
                    f'{key}="{escape_label_value(val)}"'
                    for key, val in labels.items() if val != '')
                if label_text:
                    lines.append(f'{sample_name}{{{label_text}}} {value}')
                else:
                    lines.append(f'{sample_name} {value}')
        return '\n'.join(lines) + '\n' if lines else ''


def _gauge_entries(metric, context):
    name = sanitize_name(metric.name)
    labels = _label_set(metric.tags, context)
    return [(name, 'gauge', f'Graphsignal gauge {metric.name}',
             [(name, labels, metric.datapoint['value'])])]


def _counter_entries(metric, context):
    name = sanitize_name(metric.name)
    if not name.endswith('_total'):
        name += '_total'
    labels = _label_set(metric.tags, context)
    return [(name, 'counter', f'Graphsignal counter {metric.name}',
             [(name, labels, metric.datapoint['total'])])]


def _histogram_entries(metric, context):
    dp = metric.datapoint
    name = sanitize_name(metric.name)
    count = dp.get('count')
    total = dp.get('sum')

    # A bin value is the bin's inclusive upper bound, which is what `le` is.
    buckets = {}
    bins = dp.get('bins')
    counts = dp.get('counts')
    if bins and counts and len(bins) == len(counts):
        for bound, bin_count in zip(bins, counts):
            buckets[bound] = buckets.get(bound, 0) + bin_count

    samples = []
    cumulative = 0
    for bound in sorted(buckets):
        if math.isinf(bound):
            continue
        cumulative += buckets[bound]
        samples.append((name + '_bucket',
                        _label_set(metric.tags, context, (('le', format_value(bound)),)),
                        cumulative))
    observed = sum(buckets.values())
    if count is None:
        count = observed
    samples.append((name + '_bucket',
                    _label_set(metric.tags, context, (('le', '+Inf'),)),
                    max(count, observed)))
    labels = _label_set(metric.tags, context, reserved=('le',))
    samples.append((name + '_count', labels, count))
    if total is not None:
        samples.append((name + '_sum', labels, total))

    entries = [(name, 'histogram', f'Graphsignal histogram {metric.name}', samples)]
    for extreme, word in (('min', 'smallest'), ('max', 'largest')):
        if dp.get(extreme) is not None:
            extreme_name = f'{name}_{extreme}'
            entries.append((extreme_name, 'gauge',
                            f'Graphsignal histogram {metric.name}, {word} observation',
                            [(extreme_name, labels, dp[extreme])]))
    return entries


def _profile_entries(metric, context):
    dp = metric.datapoint
    base = sanitize_name(metric.name)
    if base.endswith('_total'):
        base = base[:-len('_total')]
    value_name = base + '_total'
    samples_name = base + '_samples_total'
    frames = dp.get('frames') or {}
    frame_samples = dp.get('samples') or {}
    value_series = []
    sample_series = []
    for frame, value in sorted(frames.items(), key=lambda kv: kv[1], reverse=True):
        labels = _label_set(metric.tags, context, _frame_labels(metric.name, frame))
        value_series.append((value_name, labels, value))
        sample_series.append((samples_name, labels, frame_samples.get(frame, 0)))
    return [
        (value_name, 'counter', f'Graphsignal profile {metric.name}, cumulative value per frame',
         value_series),
        (samples_name, 'counter', f'Graphsignal profile {metric.name}, sample count per frame',
         sample_series),
    ]


_ENTRIES = {
    metrics_module.GAUGE: _gauge_entries,
    metrics_module.COUNTER: _counter_entries,
    metrics_module.HISTOGRAM: _histogram_entries,
    metrics_module.PROFILE: _profile_entries,
}


def render_metrics():
    """The current metric store as Prometheus text; empty without a watcher."""
    import graphsignal.watcher as gwatcher

    if not gwatcher.is_configured():
        return ''
    watcher = gwatcher.watcher()

    tags = watcher.tags()
    context = {key: value for key, value in tags.items()
               if key not in EXCLUDED_CONTEXT_TAGS}

    exposition = _Exposition()
    info_own = (('version', __version__), ('instance_id', tags.get('instance.id', '')))
    exposition.add([(INFO_METRIC, 'gauge', 'Graphsignal profiler version and instance',
                     [(INFO_METRIC, _label_set({}, context, info_own), 1)])])

    # The store's order is arbitrary; sorting makes which side of a name
    # collision wins stable from one scrape to the next.
    exported = sorted(
        watcher.metric_store().export(),
        key=lambda m: (m.name, m.type, sorted(m.tags.items())))
    for metric in exported:
        if metric.datapoint is None:
            continue
        try:
            entries = _ENTRIES[metric.type](metric, context)
            if not exposition.add(entries):
                logger.debug('Skipping metric %s (%s) in /metrics: name or series collision',
                             metric.name, metric.type)
        except Exception:
            logger.debug('Skipping metric %s (%s) in /metrics', metric.name, metric.type,
                         exc_info=True)

    return exposition.text()
