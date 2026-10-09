"""Colors for the top view: the Graphsignal brand palette on a terminal.

Style tokens are what render.py emits; this module maps them to curses
attributes at startup. Truecolor goes through `curses.init_color` when the
terminal lets us redefine colors, otherwise the nearest xterm-256 index,
otherwise the 8 ANSI colors. `NO_COLOR` (https://no-color.org) or
`--no-color` keeps only bold/dim/reverse.
"""

import os

# Brand palette (../website/src/styles/starlight.css).
PALETTE = {
    'accent': '#5376de',       # brand blue
    'accent_high': '#2c377c',  # dark accent (bar background, dim blue)
    'accent_low': '#dfe8fa',   # light accent
    'gray': '#8e8e8e',
    'dark_gray': '#454545',
    'amber': '#d9822b',
    'red': '#d64545',
    'white': '#ffffff',
}

# Nearest xterm-256 indexes for terminals that cannot redefine colors.
PALETTE_256 = {
    'accent': 68, 'accent_high': 60, 'accent_low': 189,
    'gray': 245, 'dark_gray': 238, 'amber': 172, 'red': 167, 'white': 231,
}

# 8-color fallback (curses.COLOR_* numbers).
PALETTE_8 = {
    'accent': 4, 'accent_high': 4, 'accent_low': 6,
    'gray': 7, 'dark_gray': 0, 'amber': 3, 'red': 1, 'white': 7,
}

# token -> (foreground color or None, background color or None, bold, dim, reverse)
STYLES = {
    'title': ('accent', None, True, False, False),
    'section': ('accent', None, True, False, False),
    'header': (None, None, True, False, False),
    'label': ('gray', None, False, False, False),
    'dim': ('gray', None, False, False, False),
    'value': (None, None, False, False, False),
    'accent': ('accent', None, False, False, False),
    'accent_dim': ('accent_high', None, False, False, False),
    'accent_low': ('accent_low', None, False, False, False),
    'bar_empty': ('dark_gray', None, False, False, False),
    'warn': ('amber', None, False, False, False),
    'error': ('red', None, True, False, False),
    # The selected table row: white on dark gray, quiet enough to leave on.
    'selected': ('white', 'dark_gray', True, False, False),
    'key': ('accent', None, True, False, False),
}


def color_enabled(flag=True):
    """Color is on unless the flag, NO_COLOR, or a dumb TERM says otherwise."""
    if not flag:
        return False
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('TERM', '') in ('dumb', ''):
        return False
    return True


def _hex_to_curses(hex_color):
    value = hex_color.lstrip('#')
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (r * 1000 // 255, g * 1000 // 255, b * 1000 // 255)


def init_colors(use_color=True):
    """Build the token -> curses attribute map. Call after curses.initscr()."""
    import curses

    attrs = {}
    color_numbers = {}
    if use_color and curses.has_colors():
        try:
            curses.start_color()
            curses.use_default_colors()
        except curses.error:
            use_color = False
    if use_color and curses.has_colors():
        colors = curses.COLORS
        if colors >= 256 and curses.can_change_color():
            # Redefine indexes above the standard 16 so the user's own
            # palette for 0..15 stays untouched.
            base = 200
            for i, (name, hex_color) in enumerate(PALETTE.items()):
                try:
                    curses.init_color(base + i, *_hex_to_curses(hex_color))
                    color_numbers[name] = base + i
                except curses.error:
                    color_numbers[name] = PALETTE_256[name]
        elif colors >= 256:
            color_numbers = dict(PALETTE_256)
        else:
            color_numbers = dict(PALETTE_8)
        pair = 1
        pair_for_colors = {}
        for token, (fg, bg, _, _, _) in STYLES.items():
            if fg is None and bg is None or (fg, bg) in pair_for_colors:
                continue
            try:
                curses.init_pair(pair, color_numbers.get(fg, -1) if fg else -1,
                                 color_numbers.get(bg, -1) if bg else -1)
                pair_for_colors[(fg, bg)] = curses.color_pair(pair)
                pair += 1
            except curses.error:
                pair_for_colors[(fg, bg)] = 0
    else:
        pair_for_colors = {}

    for token, (fg, bg, bold, dim, reverse) in STYLES.items():
        attr = pair_for_colors.get((fg, bg), 0)
        if attr == 0 and bg and not pair_for_colors:
            # No colors at all: keep the selection visible with reverse video.
            reverse = True
        if bold:
            attr |= curses.A_BOLD
        if dim:
            attr |= curses.A_DIM
        if reverse:
            attr |= curses.A_REVERSE
        attrs[token] = attr
    return attrs
