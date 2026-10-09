"""Frame layout for the top view.

`render_frame()` turns a `TopState` plus the UI state into a list of lines,
each a list of `(text, style)` spans, sized for a given terminal. It knows
nothing about curses: ui.py paints the spans with theme attributes and
`--once` joins the text. Every line is at most `width` cells wide and the
frame is exactly `height` lines, so the painter never has to clip.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from graphsignal.top import format as fmt
from graphsignal.top.state import TABS

Span = Tuple[str, str]
Line = List[Span]

MIN_TABLE_ROWS = 3
MIN_TABLE_ROWS_FULL = 10   # below this, sections collapse to make room
LABEL_COL_MAX = 40

TAB_TITLES = {'kernels': 'KERNELS', 'graphs': 'GRAPHS', 'memcpy': 'MEMCPY', 'sync': 'SYNC'}
# Key that selects each tab, and where that letter sits in the tab's name
# for the '[k]ernels' rendering.
TAB_KEYS = {'kernels': 'k', 'graphs': 'g', 'memcpy': 'm', 'sync': 's'}


@dataclass
class UiState:
    tab: str = 'kernels'
    selected: int = 0
    scroll: int = 0
    paused: bool = False
    overlay: Optional[str] = None           # 'help' | 'errors' | 'detail'
    overlay_scroll: int = 0
    once: bool = False
    rows_visible: int = 0                   # set by render for key handling
    row_count: int = 0

    def clamp(self):
        self.selected = max(0, min(self.selected, max(0, self.row_count - 1)))
        if self.rows_visible > 0:
            if self.selected < self.scroll:
                self.scroll = self.selected
            elif self.selected >= self.scroll + self.rows_visible:
                self.scroll = self.selected - self.rows_visible + 1
        self.scroll = max(0, min(self.scroll, max(0, self.row_count - self.rows_visible)))


def line_text(line):
    return ''.join(text for text, _ in line)


def frame_text(lines):
    return '\n'.join(line_text(line).rstrip() for line in lines)


def _clip(line, width):
    """Clip a line's spans to `width` cells."""
    out = []
    used = 0
    for text, style in line:
        if used >= width:
            break
        room = width - used
        if len(text) > room:
            text = text[:room]
        out.append((text, style))
        used += len(text)
    return out


def _join(*spans):
    return [s for s in spans if s[0]]


# ---------------------------------------------------------------- sections

def _header(state, ui, width):
    header = state.header()
    lines = []

    right = []
    if header.uptime_s is not None:
        right.append(('up ' + fmt.fmt_uptime(header.uptime_s), 'label'))
    if header.unavailable:
        right.append(('  ● ' + header.unavailable, 'error'))
    elif ui.paused:
        right.append(('  ● paused', 'warn'))
    else:
        right.append(('  ● live', 'accent'))
    right_text = ''.join(t for t, _ in right)

    left = [(' graphsignal-top', 'title'), ('  ' + (header.endpoint or ''), 'label')]
    left_len = sum(len(t) for t, _ in left)
    room = width - left_len - len(right_text) - 4
    if header.command_line and room > 8:
        left.append(('   ' + fmt.fit(header.command_line, room - 3), 'value'))
        left_len = sum(len(t) for t, _ in left)
    gap = max(1, width - left_len - len(right_text) - 1)
    lines.append(_clip(left + [(' ' * gap, 'value')] + right, width))

    parts = []
    if header.pid:
        parts.append(f'pid {header.pid}')
    if header.host:
        parts.append(f'host {header.host}')
    if header.profiler_version:
        parts.append(f'profiler {header.profiler_version}')
    second = [(' ' + '  '.join(parts), 'label')]
    if header.graph_trace_mode == 'graph':
        second.append(('  cuda graph trace: graph', 'label'))
        if ui.tab != 'graphs':
            used = sum(len(t) for t, _ in second)
            for note in ('  (kernels replayed from graphs are listed under [g]; '
                         '--cuda-graph-trace node lists them as kernels)',
                         '  (graph kernels under [g]; --cuda-graph-trace node lists them)',
                         '  (graph kernels under [g])'):
                if used + len(note) <= width:
                    second.append((note, 'dim'))
                    break
    elif header.graph_trace_mode == 'node':
        second.append(('  cuda graph trace: node', 'label'))
    elif header.rocm:
        second.append(('  rocm', 'label'))
    lines.append(_clip(second, width))
    return lines


def _device_lines(state, ui, device, width, full):
    lines = []
    name = device.name or device.uuid
    title = [(f' GPU {device.index}  ', 'section'), (name, 'value')]
    stats = []
    if device.power is not None:
        limit = f' / {device.power_limit:.0f}' if device.power_limit else ''
        stats.append((f'power {device.power:.0f}{limit} W', 'value'))
    if device.temperature is not None:
        stats.append((f'temp {device.temperature:.0f}°C', 'value'))
    if device.sm_clock_mhz is not None:
        stats.append((f'sm {device.sm_clock_mhz / 1000:.2f} GHz', 'value'))
    if device.throttle_reasons is not None:
        stats.append((f'throttle 0x{int(device.throttle_reasons):x}', 'warn') if device.throttled
                     else ('throttle none', 'value'))
    if device.xid_errors is not None:
        stats.append((f'xid {int(device.xid_errors)}', 'error' if device.xid_errors else 'value'))
    # Last, so a long device name squeezes PCIe out first rather than the
    # health indicators.
    if device.pcie_rx_bytes_per_s is not None or device.pcie_tx_bytes_per_s is not None:
        rx = fmt.fmt_bytes(device.pcie_rx_bytes_per_s).replace(' ', '') \
            if device.pcie_rx_bytes_per_s is not None else '-'
        tx = fmt.fmt_bytes(device.pcie_tx_bytes_per_s).replace(' ', '') \
            if device.pcie_tx_bytes_per_s is not None else '-'
        stats.append((f'pcie rx {rx}/s tx {tx}/s', 'value'))
    has_util = device.utilization is not None or any(v is not None for v in device.utilization_history)
    if not full or not has_util:
        util = fmt.fmt_percent(device.utilization) if device.utilization is not None else '-'
        mem = fmt.fmt_percent(device.memory_fraction * 100) if device.memory_fraction is not None else '-'
        stats.insert(0, (f'util {util}', 'accent'))
        if not full:
            stats.insert(1, (f'mem {mem}', 'accent'))
    # Drop trailing stats rather than clip one mid-word; a non-zero xid or
    # throttle always stays.
    title_len = sum(len(t) for t, _ in title)
    while stats and title_len + sum(len(t) + 3 for t, _ in stats) > width:
        keep = [(t, st) for t, st in stats if st in ('error', 'warn')]
        droppable = [i for i, (t, st) in enumerate(stats) if st not in ('error', 'warn')]
        if not droppable:
            break
        stats.pop(droppable[-1])
        if not stats and keep:
            stats = keep
    line = list(title)
    for text, style in stats:
        line.append(('   ', 'value'))
        line.append((text, style))
    lines.append(_clip(line, width))
    if not full:
        return lines

    # util row: '   util   94%   <spark>  2m' — only once utilization has
    # been measured (some platforms' NVML never reports it).
    if not has_util:
        return lines + _memory_line(device, width)
    util_text = fmt.fmt_percent(device.utilization) if device.utilization is not None else '-'
    prefix = [('   util   ', 'label'), (f'{util_text:>4}', 'value'), ('   ', 'value')]
    prefix_len = sum(len(t) for t, _ in prefix)
    spark_width = max(0, width - prefix_len - 6)
    span = state.series_span(spark_width)
    suffix = ('  ' + fmt.fmt_window(span)) if span else ''
    spark = fmt.sparkline(device.utilization_history, spark_width, vmax=100.0)
    throttled = device.throttle_history[-spark_width:] if spark_width else []
    throttled = [False] * (spark_width - len(throttled)) + list(throttled)
    spark_spans = []
    for ch, hot in zip(spark, throttled):
        style = 'warn' if hot else 'accent'
        if spark_spans and spark_spans[-1][1] == style:
            spark_spans[-1] = (spark_spans[-1][0] + ch, style)
        else:
            spark_spans.append((ch, style))
    lines.append(_clip(prefix + spark_spans + [(suffix, 'label')], width))

    return lines + _memory_line(device, width)


def _memory_line(device, width):
    # mem row: '   mem    98.1 / 119.7 GB  [bar]  82%'
    lines = []
    if device.memory_used is not None or device.memory_total is not None:
        used = fmt.fmt_bytes(device.memory_used) if device.memory_used is not None else '-'
        total = fmt.fmt_bytes(device.memory_total) if device.memory_total is not None else '-'
        if ' ' in used and ' ' in total and used.split(' ')[1] == total.split(' ')[1]:
            used = used.split(' ')[0]   # '98.1 / 119.7 GB'
        pct = device.memory_fraction
        mem_prefix = [('   mem    ', 'label'), (f'{used} / {total}', 'value'), ('  [', 'label')]
        pct_text = ('  ' + fmt.fmt_percent(pct * 100)) if pct is not None else '  -'
        bar_width = max(0, width - sum(len(t) for t, _ in mem_prefix) - 1 - len(pct_text))
        bar = fmt.bar(pct, bar_width)
        filled = bar.count(fmt.BAR_FILL)
        hot = pct is not None and pct > 0.95
        lines.append(_clip(mem_prefix + [
            (bar[:filled], 'warn' if hot else 'accent'),
            (bar[filled:], 'bar_empty'),
            (']', 'label'), (pct_text, 'warn' if hot else 'value')], width))
    return lines


def _devices(state, ui, width, full=True):
    """Device blocks: three rows each, or one row each when `full` is False
    (render_frame collapses them when the terminal is too short)."""
    devices = state.devices()
    if not devices:
        return [[(' GPU   ', 'section'), ('no device telemetry yet', 'dim')]]
    lines = []
    for device in devices:
        lines.extend(_device_lines(state, ui, device, width, full))
    return lines


def _engine(state, ui, width):
    engine = state.engine()
    if engine is None:
        return []
    lines = []
    title = [(' ENGINE  ', 'section'), (engine.engine, 'value')]
    stats = []
    if engine.running is not None:
        stats.append(f'running {engine.running:.0f}')
    if engine.waiting is not None:
        stats.append(f'waiting {engine.waiting:.0f}')
    if engine.kv_cache_pct is not None:
        stats.append(f'kv cache {engine.kv_cache_pct:.0f}%')
    if engine.prefix_hit_pct is not None:
        stats.append(f'prefix cache hit {engine.prefix_hit_pct:.0f}%')
    if stats:
        title.append(('   ' + '   '.join(stats), 'value'))
    lines.append(_clip(title, width))

    unit = 'tok/s'
    for label, value, history in (
            ('prompt ' + unit, engine.prompt_tokens_per_s, engine.prompt_history),
            ('gen ' + unit, engine.gen_tokens_per_s, engine.gen_history)):
        prefix = [(f'   {label:<13}', 'label'), (f'{fmt.fmt_rate(value):>7}', 'value'), ('  ', 'value')]
        prefix_len = sum(len(t) for t, _ in prefix)
        spark_width = max(0, width - prefix_len - 6)
        span = state.series_span(spark_width)
        suffix = ('  ' + fmt.fmt_window(span)) if span else ''
        spark = fmt.sparkline(history, spark_width)
        lines.append(_clip(prefix + [(spark, 'accent'), (suffix, 'label')], width))

    def q(view):
        return f'p50 {fmt.fmt_seconds(view.p50)}  p95 {fmt.fmt_seconds(view.p95)}'

    latency = [('   ttft  ', 'label'), (q(engine.ttft), 'value'),
               ('       tpot  ', 'label'), (q(engine.tpot), 'value'),
               ('       e2e  ', 'label'), (q(engine.e2e), 'value')]
    lines.append(_clip(latency, width))
    return lines


def _gpu_time(state, ui, width):
    view = state.gpu_time()
    scope = f'last {fmt.fmt_window(state.dt)}' if state.dt else 'warming up'
    title = [(' GPU TIME', 'section'), (f' ({scope})', 'label')]
    if not view.available:
        title.append(('   no GPU activity yet', 'dim'))
        return [_clip(title, width)]
    parts = [('kernels', view.kernels), ('graphs', view.graphs), ('memcpy', view.memcpy),
             ('memset', view.memset), ('sync', view.sync), ('idle', view.idle)]
    for name, value in parts:
        if value is None:
            continue
        title.append(('   ', 'value'))
        title.append((f'{name} ', 'label'))
        title.append((fmt.fmt_percent(value), 'value'))
    lines = [_clip(title, width)]

    # bar row: one segment per busy category plus idle, proportional, each a
    # plain label followed by its bar, separated by a gap. One color for GPU
    # time, a dim one for idle. Sync is host wait time, not GPU time, so it
    # is in the numbers but not in the bar.
    segments = [(name, value) for name, value in parts if value and name != 'sync']
    if not segments:
        return lines
    gap = '  '
    labels_width = sum(len(name) + 1 for name, _ in segments) + len(gap) * (len(segments) - 1)
    bar_width = max(len(segments), width - 3 - labels_width)
    total = sum(v for _, v in segments) or 1.0
    cells = [max(1, int(round(value / total * bar_width))) for _, value in segments]
    while sum(cells) > bar_width:
        cells[cells.index(max(cells))] -= 1
    spans = [('   ', 'value')]
    for i, ((name, _), count) in enumerate(zip(segments, cells)):
        if i:
            spans.append((gap, 'value'))
        spans.append((name + ' ', 'label'))
        if name == 'idle':
            spans.append((fmt.BAR_EMPTY * count, 'bar_empty'))
        else:
            spans.append((fmt.BAR_FULL * count, 'accent'))
    lines.append(_clip(spans, width))
    return lines


def _table(state, ui, width, rows_available):
    rows = state.table(ui.tab)
    ui.row_count = len(rows)
    lines = []

    scope = f'last {fmt.fmt_window(state.dt)}' if state.dt else 'warming up'
    title = [(f' {TAB_TITLES.get(ui.tab, ui.tab.upper())}', 'section'),
             (f'  sorted by time, {scope}', 'label')]
    tabs = []
    for tab in TABS:
        if not state.has_tab(tab) and tab != ui.tab:
            continue
        key = TAB_KEYS[tab]
        pos = tab.index(key)
        active = tab == ui.tab
        tabs.append(('  ' + tab[:pos] + '[', 'label' if not active else 'header'))
        tabs.append((key, 'key' if not active else 'selected'))
        tabs.append((']' + tab[pos + 1:], 'label' if not active else 'header'))
    title_len = sum(len(t) for t, _ in title)
    tabs_len = sum(len(t) for t, _ in tabs)
    gap = max(1, width - title_len - tabs_len - 1)
    lines.append(_clip(title + [(' ' * gap, 'value')] + tabs, width))

    bytes_tab = ui.tab == 'memcpy'
    cols = [('TIME%', 7, 'right'), ('ms/s', 9, 'right'), ('calls/s', 10, 'right'),
            ('avg µs', 9, 'right'), ('BYTES/s' if bytes_tab else 'CATEGORY', 13, 'left')]
    fixed = sum(w + 1 for _, w, _ in cols) + 1
    name_width = max(10, width - fixed)
    header = [(' ', 'value')]
    for name, w, align in cols:
        header.append((fmt.pad(name, w, align) + ' ', 'header'))
    name_header = ('KERNEL' if ui.tab == 'kernels' else
                   'GRAPH' if ui.tab == 'graphs' else 'KIND')
    header.append((fmt.pad(name_header, name_width), 'header'))
    lines.append(_clip(header, width))

    body_rows = max(0, rows_available - len(lines))
    ui.rows_visible = body_rows
    ui.clamp()
    visible = rows[ui.scroll:ui.scroll + body_rows]
    if not rows:
        reason = 'no activity yet' if state.dt else 'warming up — rates need two polls'
        lines.append([('   ', 'value'), (reason, 'dim')])
    for i, row in enumerate(visible):
        idx = ui.scroll + i
        selected = (idx == ui.selected) and not ui.once
        style = 'selected' if selected else 'value'
        cells = [(' ', style)]
        time_pct = f'{row.share_pct:.1f}' if row.share_pct is not None else '-'
        per_s = f'{row.ns_per_s / 1e6:.1f}' if row.ns_per_s is not None else '-'
        calls = fmt.fmt_count(row.calls_per_s) if row.calls_per_s is not None else '-'
        avg = f'{row.avg_ns / 1000:.1f}' if row.avg_ns is not None else '-'
        if bytes_tab:
            fifth = (fmt.fmt_bytes(row.bytes_per_s) + '/s') if row.bytes_per_s is not None else '-'
        else:
            fifth = row.category
        values = [time_pct, per_s, calls, avg, fifth]
        for (name, w, align), value in zip(cols, values):
            cells.append((fmt.pad(value, w, align) + ' ', style))
        label_width = min(LABEL_COL_MAX, name_width)
        short_width = name_width - label_width - 2
        if short_width >= 8 and row.short != row.label:
            cells.append((fmt.pad(row.label, label_width) + '  ', style))
            cells.append((fmt.pad(row.short, short_width), style if selected else 'dim'))
        else:
            cells.append((fmt.pad(row.label, name_width), style))
        lines.append(_clip(cells, width))
    hidden = len(rows) - (ui.scroll + len(visible))
    if hidden > 0 and body_rows > 0:
        if len(lines) - 2 >= body_rows:
            lines[-1] = [('   ', 'value'), (f'… {hidden + 1} more', 'dim')]
        else:
            lines.append([('   ', 'value'), (f'… {hidden} more', 'dim')])
    return lines


def _errors_line(state, width):
    errors = state.errors()
    if not errors:
        return [[(' ERRORS  ', 'section'), ('none', 'dim')]]
    newest = errors[-1]
    level = str(newest.get('level') or '').upper()
    ts = newest.get('log_ns')
    when = ''
    if ts:
        import time as _time
        when = _time.strftime('%H:%M:%S', _time.localtime(ts / 1e9)) + '  '
    message = (newest.get('message') or '').splitlines()[0] if newest.get('message') else ''
    count = f'  {len(errors)} total' if len(errors) > 1 else ''
    style = 'error' if level in ('ERROR', 'CRITICAL') else 'warn'
    prefix = [(' ERRORS  ', 'section'), (when, 'label'), (level + '  ', style)]
    room = width - sum(len(t) for t, _ in prefix) - len(count)
    return [_clip(prefix + [(fmt.fit(message, max(0, room)), 'value'), (count, 'label')], width)]


def _keys_line(ui, width):
    keys = [(' q', 'key'), (' quit  ', 'label'), ('↑↓', 'key'), (' select  ', 'label'),
            ('⏎', 'key'), (' details  ', 'label'), ('p', 'key'), (' pause  ', 'label'),
            ('e', 'key'), (' errors  ', 'label'),
            ('?', 'key'), (' help', 'label')]
    return [_clip(keys, width)]


# ---------------------------------------------------------------- overlays

def _box(title, body_lines, width, height):
    """Center `body_lines` in a bordered box; returns the box's lines and top row."""
    inner_w = min(width - 4, max(40, max((len(line_text(l)) for l in body_lines), default=40) + 2))
    inner_h = min(height - 4, len(body_lines))
    box_w = inner_w + 2
    left = max(0, (width - box_w) // 2)
    top = max(0, (height - inner_h - 2) // 2)
    pad = ' ' * left
    lines = [[(pad + '┌─ ', 'accent'), (title, 'title'), (' ' + '─' * max(0, inner_w - len(title) - 3) + '┐', 'accent')]]
    for body in body_lines[:inner_h]:
        text_len = len(line_text(body))
        lines.append(_clip([(pad + '│ ', 'accent')] + body + [(' ' * max(0, inner_w - text_len - 1) + '│', 'accent')],
                           width))
    lines.append([(pad + '└' + '─' * inner_w + '┘', 'accent')])
    return lines, top


HELP_LINES = [
    [('k g m s', 'key'), ('   kernels / graphs / memcpy / sync tabs (tab cycles)', 'value')],
    [('↑ ↓', 'key'), ('       select a row; pgup/pgdn page; home/end jump', 'value')],
    [('⏎', 'key'), ('         details of the selected row: full symbol, totals since start', 'value')],
    [('p', 'key'), ('         pause polling', 'value')],
    [('e', 'key'), ('         all recent errors from the watcher', 'value')],
    [('q', 'key'), ('         quit', 'value')],
]


def _overlay(state, ui, width, height):
    if ui.overlay == 'help':
        return _box('help', HELP_LINES, width, height)
    if ui.overlay == 'errors':
        body = []
        for entry in reversed(state.errors()):
            level = str(entry.get('level') or '').upper()
            style = 'error' if level in ('ERROR', 'CRITICAL') else 'warn'
            for i, text in enumerate((entry.get('message') or '').splitlines() or ['']):
                body.append([(f'{level:<8} ' if i == 0 else ' ' * 9, style), (fmt.fit(text, width - 16), 'value')])
            for text in (entry.get('exception') or '').splitlines():
                body.append([(' ' * 9, 'value'), (fmt.fit(text, width - 16), 'dim')])
        if not body:
            body = [[('no errors', 'dim')]]
        return _box('errors', body[ui.overlay_scroll:], width, height)
    if ui.overlay == 'detail':
        rows = state.table(ui.tab)
        if not rows:
            return None
        row = rows[min(ui.selected, len(rows) - 1)]
        wrap = max(20, width - 10)
        body = [[('label     ', 'label'), (row.label, 'value')],
                [('category  ', 'label'), (row.category, 'value')]]
        if row.kernel and row.kernel.library:
            body.append([('library   ', 'label'), (row.kernel.library, 'value')])
        raw = row.raw
        for i in range(0, max(1, len(raw)), wrap):
            body.append([('symbol    ' if i == 0 else ' ' * 10, 'label'), (raw[i:i + wrap], 'dim')])

        def stats(scope, r, per_s):
            text = (f'time {fmt.fmt_duration_ns(r.value_ns)}  calls {fmt.fmt_count(r.samples)}  '
                    f'avg {fmt.fmt_duration_ns(r.avg_ns)}  share {fmt.fmt_percent(r.share_pct, 1)}')
            if r.bytes_per_s is not None:
                text += f'  bytes {fmt.fmt_bytes(r.bytes_per_s)}' + ('/s' if per_s else '')
            return [(f'{scope:<10}', 'label'), (text, 'value')]

        body.append(stats(f'last {fmt.fmt_window(state.dt or 0)}', row, True))
        total = next((r for r in state.table(ui.tab, cumulative=True) if r.key == row.key), None)
        if total is not None:
            body.append(stats('since start', total, False))
        if row.per_process:
            body.append([('processes ', 'label'), ('', 'value')])
            for pid, (value, samples) in sorted(row.per_process.items(), key=lambda kv: -kv[1][0]):
                body.append([(' ' * 10, 'value'),
                             (f'pid {pid:<8} {fmt.fmt_duration_ns(value):>10}  {fmt.fmt_count(samples):>9} calls', 'value')])
        return _box('details', body, width, height)
    return None


# ---------------------------------------------------------------- frame

def render_frame(state, ui, width, height):
    width = max(40, int(width))
    height = max(8, int(height))

    header = _header(state, ui, width)
    devices = _devices(state, ui, width)
    engine = _engine(state, ui, width)
    gpu_time = _gpu_time(state, ui, width)
    errors = _errors_line(state, width)
    keys = [] if ui.once else _keys_line(ui, width)

    def assemble(devices, engine):
        top = list(header) + [[]] + list(devices) + [[]]
        if engine:
            top += list(engine) + [[]]
        top += list(gpu_time) + [[]]
        return top

    top = assemble(devices, engine)
    bottom = errors + keys
    rows_available = height - len(top) - len(bottom)
    # Too little room for the table: first collapse every device to its
    # title row, then drop the engine's sparklines. No key to press.
    if rows_available < MIN_TABLE_ROWS_FULL and state.device_count() > 1:
        devices = _devices(state, ui, width, full=False)
        top = assemble(devices, engine)
        rows_available = height - len(top) - len(bottom)
    if rows_available < MIN_TABLE_ROWS_FULL and engine:
        engine = engine[:1]
        top = assemble(devices, engine)
        rows_available = height - len(top) - len(bottom)
    rows_available = max(MIN_TABLE_ROWS, rows_available)

    if ui.once:
        # One printed frame: every row, no filler, no fixed height.
        table = _table(state, ui, width, 10 ** 6)
        return top + table + [[]] + bottom

    table = _table(state, ui, width, rows_available)
    lines = top + table
    lines = lines[:max(0, height - len(bottom))]
    while len(lines) < height - len(bottom):
        lines.append([])
    lines += bottom
    lines = lines[:height]

    if ui.overlay and not ui.once:
        box = _overlay(state, ui, width, height)
        if box:
            box_lines, top_row = box
            for i, box_line in enumerate(box_lines):
                if top_row + i < len(lines):
                    lines[top_row + i] = box_line
    return lines
