"""Humanizing helpers for the top view: units, sparklines, bars, truncation.

All functions return plain strings; styling happens in render.py.
"""

# Sparklines top out at the seven-eighths block, and the percent bar is drawn
# with it on a thin track: block glyphs grow from the bottom of the cell, so a
# full-cell block fuses with the row below. The blank strip this leaves at the
# top of every row keeps stacked rows apart without spending a line on it.
SPARK_BLOCKS = ' ▁▂▃▄▅▆▇'
BAR_FULL = '█'      # segment bars with text beside them (GPU time)
BAR_EMPTY = '░'
BAR_FILL = '▇'      # the percent bar
BAR_TRACK = '▁'
ELLIPSIS = '…'


def fmt_bytes(value, digits=1):
    """Decimal SI bytes: 98.1 GB, 512.0 MB, 1.2 TB. None → '-'."""
    if value is None:
        return '-'
    value = float(value)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB'):
        if abs(value) < 1000 or unit == 'PB':
            if unit == 'B':
                return f'{int(value)} B'
            return f'{value:.{digits}f} {unit}'
        value /= 1000.0
    return f'{value:.{digits}f} PB'


def fmt_rate(value, digits=None):
    """Compact count or rate: 980, 1.84k, 11.2k, 2.5M. None → '-'."""
    if value is None:
        return '-'
    value = float(value)
    if abs(value) < 1000:
        if digits is None:
            return f'{value:.0f}' if abs(value) >= 10 or value == int(value) else f'{value:.1f}'
        return f'{value:.{digits}f}'
    for suffix in ('k', 'M', 'G', 'T'):
        value /= 1000.0
        if abs(value) < 1000 or suffix == 'T':
            d = digits if digits is not None else (2 if abs(value) < 10 else 1)
            return f'{value:.{d}f}{suffix}'
    return f'{value:.1f}T'


def fmt_count(value):
    """Integer with thousands separators: 1,980. None → '-'."""
    if value is None:
        return '-'
    return f'{int(round(value)):,}'


def fmt_duration_ns(ns):
    """Nanoseconds as the nearest readable unit: 158.6 µs, 24.1 ms, 2.9 s."""
    if ns is None:
        return '-'
    ns = float(ns)
    if ns < 1_000:
        return f'{ns:.0f} ns'
    if ns < 1_000_000:
        return _scaled(ns / 1_000, 'µs')
    if ns < 1_000_000_000:
        return _scaled(ns / 1_000_000, 'ms')
    return _scaled(ns / 1_000_000_000, 's')


def _scaled(value, unit):
    """One decimal below 100, none above: 24.1 ms, 182 ms."""
    return f'{value:.1f} {unit}' if value < 100 else f'{value:.0f} {unit}'


def fmt_seconds(seconds):
    """Seconds as the nearest readable unit (latency quantiles)."""
    if seconds is None:
        return '-'
    return fmt_duration_ns(float(seconds) * 1_000_000_000)


def fmt_percent(value, digits=0):
    if value is None:
        return '-'
    return f'{float(value):.{digits}f}%'


def fmt_uptime(seconds):
    """02:14:37, or 1d 02:14 past a day."""
    if seconds is None:
        return '-'
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f'{days}d {hours:02d}:{minutes:02d}'
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def fmt_window(seconds):
    """Sparkline window length label: 45s, 2m, 1h."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m'
    return f'{seconds // 3600}h'


def sparkline(values, width, vmax=None):
    """Right-aligned one-row sparkline of the last `width` values.

    `values` may contain None (not measured → blank cell). The scale is
    0..vmax; vmax defaults to the max of the values shown, so a series that
    is all zero still renders as a flat floor rather than nothing.
    """
    if width <= 0:
        return ''
    values = list(values)[-width:]
    if vmax is None:
        present = [v for v in values if v is not None]
        vmax = max(present) if present else 0
    cells = []
    for v in values:
        if v is None:
            cells.append(' ')
            continue
        if vmax <= 0:
            level = 1 if v <= 0 else len(SPARK_BLOCKS) - 1
        else:
            level = 1 + int(round(min(max(v, 0), vmax) / vmax * (len(SPARK_BLOCKS) - 2)))
        cells.append(SPARK_BLOCKS[level])
    return ''.join(cells).rjust(width)


def bar(fraction, width):
    """Percent bar of `width` cells: filled cells on a thin track."""
    if width <= 0:
        return ''
    if fraction is None:
        return BAR_TRACK * width
    filled = int(round(min(max(float(fraction), 0.0), 1.0) * width))
    return BAR_FILL * filled + BAR_TRACK * (width - filled)


def fit(text, width):
    """Truncate to `width` cells with an ellipsis; never wider than width."""
    if width <= 0:
        return ''
    text = str(text)
    if len(text) <= width:
        return text
    if width == 1:
        return ELLIPSIS
    return text[:width - 1] + ELLIPSIS


def pad(text, width, align='left'):
    """fit() then pad to exactly `width` cells."""
    text = fit(text, width)
    if align == 'right':
        return text.rjust(width)
    return text.ljust(width)
