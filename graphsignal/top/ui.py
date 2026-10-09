"""The interactive loop: curses painting, key handling, and the poller thread.

This is the only module in the package that imports curses. Everything it
paints comes from `render.render_frame`; everything it changes is the
`UiState` the renderer reads, so `--once` and the tests exercise the same
layout code without a terminal.
"""

import curses
import locale
import threading
import time

from graphsignal.top import theme
from graphsignal.top.client import SignalsUnavailable
from graphsignal.top.render import TAB_KEYS, UiState, render_frame
from graphsignal.top.state import TABS

KEY_TIMEOUT_MS = 250
ENTER_KEYS = (10, 13, curses.KEY_ENTER)


class Poller(threading.Thread):
    """Fetches `/signals` every `interval` seconds into the shared state."""

    def __init__(self, client, state, interval, lock, ui):
        super().__init__(name='graphsignal-top-poller', daemon=True)
        self.client = client
        self.state = state
        self.interval = max(0.1, float(interval))
        self.lock = lock
        self.ui = ui
        self._stop = threading.Event()
        self.dirty = threading.Event()

    def run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            if not self.ui.paused:
                self.poll_once()
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.05, self.interval - elapsed))

    def poll_once(self):
        try:
            payload = self.client.fetch()
        except SignalsUnavailable as err:
            with self.lock:
                self.state.mark_unavailable(str(err))
        except Exception as err:  # never let the poller die on a surprise
            with self.lock:
                self.state.mark_unavailable(type(err).__name__)
        else:
            with self.lock:
                self.state.update(payload, received_ts=time.time())
        self.dirty.set()

    def stop(self):
        self._stop.set()


def paint(stdscr, lines, attrs):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    for y, line in enumerate(lines[:height]):
        x = 0
        for text, style in line:
            if x >= width:
                break
            text = text[:width - x]
            try:
                stdscr.addstr(y, x, text, attrs.get(style, 0))
            except curses.error:
                # Writing the bottom-right cell raises after the write succeeds.
                pass
            x += len(text)
    stdscr.refresh()


def _switch_tab(ui, state, tab):
    if tab == ui.tab:
        return
    ui.tab = tab
    ui.selected = 0
    ui.scroll = 0


def handle_key(key, ui, state):
    """Apply one key press to the UI state. Returns 'quit' to exit."""
    if key == curses.KEY_RESIZE:
        return None

    if ui.overlay:
        if key == curses.KEY_DOWN and ui.overlay == 'errors':
            ui.overlay_scroll += 1
        elif key == curses.KEY_UP and ui.overlay == 'errors':
            ui.overlay_scroll = max(0, ui.overlay_scroll - 1)
        elif key in (ord('q'), ord('Q')) and ui.overlay != 'detail':
            return 'quit'
        else:
            ui.overlay = None
            ui.overlay_scroll = 0
        return None

    if key in (ord('q'), ord('Q')):
        return 'quit'
    if key == ord('?'):
        ui.overlay = 'help'
    elif key == ord('e'):
        ui.overlay = 'errors'
    elif key in ENTER_KEYS:
        if ui.row_count:
            ui.overlay = 'detail'
    elif key == ord('p'):
        ui.paused = not ui.paused
    elif key == 9:  # tab: cycle the tabs that have data
        available = [t for t in TABS if state.has_tab(t)] or [ui.tab]
        if ui.tab in available:
            _switch_tab(ui, state, available[(available.index(ui.tab) + 1) % len(available)])
        else:
            _switch_tab(ui, state, available[0])
    elif 0 <= key < 256 and chr(key) in TAB_KEYS.values():
        tab = next(t for t, k in TAB_KEYS.items() if k == chr(key))
        if state.has_tab(tab) or tab == 'kernels':
            _switch_tab(ui, state, tab)
    elif key == curses.KEY_UP:
        ui.selected -= 1
    elif key == curses.KEY_DOWN:
        ui.selected += 1
    elif key == curses.KEY_PPAGE:
        ui.selected -= max(1, ui.rows_visible)
    elif key == curses.KEY_NPAGE:
        ui.selected += max(1, ui.rows_visible)
    elif key == curses.KEY_HOME:
        ui.selected = 0
    elif key == curses.KEY_END:
        ui.selected = max(0, ui.row_count - 1)
    ui.clamp()
    return None


def _loop(stdscr, client, state, interval, use_color):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    stdscr.timeout(KEY_TIMEOUT_MS)
    attrs = theme.init_colors(use_color)

    ui = UiState()
    lock = threading.Lock()
    poller = Poller(client, state, interval, lock, ui)
    poller.start()
    try:
        while True:
            height, width = stdscr.getmaxyx()
            with lock:
                lines = render_frame(state, ui, width, height)
            paint(stdscr, lines, attrs)
            poller.dirty.clear()
            key = stdscr.getch()
            if key == -1:
                continue
            with lock:
                if handle_key(key, ui, state) == 'quit':
                    break
    finally:
        poller.stop()


def run(client, state, interval=1.0, use_color=True):
    """Run the interactive view until the user quits. Returns the exit code."""
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        pass
    try:
        curses.wrapper(_loop, client, state, interval, use_color)
    except KeyboardInterrupt:
        pass
    return 0
