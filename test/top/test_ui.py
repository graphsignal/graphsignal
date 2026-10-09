import curses
import unittest

from graphsignal.top.render import UiState, render_frame
from graphsignal.top.state import TopState
from graphsignal.top.ui import handle_key
from test.top import fixtures as fx


def make_state(**kwargs):
    state = TopState(window=5.0)
    for t in range(10, 21):
        state.update(fx.build_payload(t, **kwargs))
    return state


class HandleKeyTest(unittest.TestCase):
    def setUp(self):
        self.state = make_state(graph_mode='graph')
        self.ui = UiState()
        render_frame(self.state, self.ui, 120, 30)   # sets row_count / rows_visible

    def key(self, ch):
        return handle_key(ch if isinstance(ch, int) else ord(ch), self.ui, self.state)

    def test_quit(self):
        self.assertEqual(self.key('q'), 'quit')
        self.assertEqual(self.key('Q'), 'quit')

    def test_tabs_by_letter_and_cycle(self):
        self.key('g')
        self.assertEqual(self.ui.tab, 'graphs')
        self.key('m')
        self.assertEqual(self.ui.tab, 'memcpy')
        self.key('t')   # no such tab: ignored
        self.assertEqual(self.ui.tab, 'memcpy')
        self.key('s')
        self.assertEqual(self.ui.tab, 'sync')
        self.key(9)     # tab cycles to the first available
        self.assertEqual(self.ui.tab, 'kernels')
        self.key(9)
        self.assertEqual(self.ui.tab, 'graphs')

    def test_switching_tab_resets_selection(self):
        self.ui.selected = 2
        self.key('g')
        self.assertEqual(self.ui.selected, 0)

    def test_toggles(self):
        for removed in 'cr/':
            self.assertIsNone(self.key(removed))
        self.key('p')
        self.assertTrue(self.ui.paused)
        self.key('p')
        self.assertFalse(self.ui.paused)

    def test_selection_clamps(self):
        self.assertEqual(self.ui.row_count, 3)
        self.key(curses.KEY_DOWN)
        self.key(curses.KEY_DOWN)
        self.key(curses.KEY_DOWN)
        self.assertEqual(self.ui.selected, 2)
        self.key(curses.KEY_UP)
        self.assertEqual(self.ui.selected, 1)
        self.key(curses.KEY_HOME)
        self.assertEqual(self.ui.selected, 0)
        self.key(curses.KEY_END)
        self.assertEqual(self.ui.selected, 2)
        self.key(curses.KEY_PPAGE)
        self.assertEqual(self.ui.selected, 0)

    def test_overlays(self):
        self.key('?')
        self.assertEqual(self.ui.overlay, 'help')
        self.key('x')
        self.assertIsNone(self.ui.overlay)
        self.key('e')
        self.assertEqual(self.ui.overlay, 'errors')
        self.key(curses.KEY_DOWN)
        self.assertEqual(self.ui.overlay_scroll, 1)
        self.key(27)
        self.assertIsNone(self.ui.overlay)
        self.assertEqual(self.ui.overlay_scroll, 0)
        self.key(10)
        self.assertEqual(self.ui.overlay, 'detail')
        self.key('q')   # q closes the detail overlay rather than quitting
        self.assertIsNone(self.ui.overlay)
        self.key('?')
        self.assertEqual(self.key('q'), 'quit')

    def test_detail_needs_rows(self):
        self.ui.row_count = 0
        self.key(10)
        self.assertIsNone(self.ui.overlay)

    def test_digits_do_nothing(self):
        self.assertIsNone(self.key('3'))
        self.assertEqual(self.ui.selected, 0)

    def test_resize_is_ignored(self):
        self.assertIsNone(self.key(curses.KEY_RESIZE))
