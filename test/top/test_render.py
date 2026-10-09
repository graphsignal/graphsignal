import unittest

from graphsignal.top.render import UiState, frame_text, line_text, render_frame
from graphsignal.top.state import TopState
from test.top import fixtures as fx


def make_state(ts=range(10, 21), **kwargs):
    state = TopState(window=5.0, endpoint='127.0.0.1:18259')
    for t in ts:
        state.update(fx.build_payload(t, **kwargs))
    return state


def render(state, ui=None, width=120, height=30):
    ui = ui or UiState()
    lines = render_frame(state, ui, width, height)
    return lines, [line_text(l) for l in lines]


class FrameGeometryTest(unittest.TestCase):
    def test_exact_height_and_max_width(self):
        for width, height in ((100, 24), (160, 40), (80, 12), (40, 8)):
            lines, texts = render(make_state(), width=width, height=height)
            self.assertEqual(len(lines), height, (width, height))
            for text in texts:
                self.assertLessEqual(len(text), width, (width, height, text))

    def test_once_frame_is_compact(self):
        lines, texts = render(make_state(), UiState(once=True), height=8)
        self.assertGreater(len(lines), 8)
        self.assertEqual(texts[-1].lstrip()[:6], 'ERRORS')
        self.assertEqual(texts[-2], '')
        self.assertIn('rms norm', texts[-3])   # last table row, no filler

    def test_tab_keys_rendered(self):
        _, texts = render(make_state(), UiState(tab='memcpy'))
        title = next(t for t in texts if 'sorted by time' in t)
        self.assertIn('[k]ernels', title)
        self.assertIn('[m]emcpy', title)
        self.assertIn('[s]ync', title)
        self.assertNotIn('umulative', title)
        self.assertNotIn('filter', title)

    def test_sections_present(self):
        _, texts = render(make_state())
        joined = '\n'.join(texts)
        self.assertIn('graphsignal-top  127.0.0.1:18259', texts[0])
        self.assertIn('vllm serve meta-llama', texts[0])
        self.assertIn('● live', texts[0])
        self.assertIn('pid 41822', texts[1])
        self.assertIn('cuda graph trace: node', texts[1])
        self.assertIn('GPU 0  NVIDIA GB10', joined)
        self.assertIn('util    94%', joined)
        self.assertIn('mem    98.1 / 119.7 GB', joined)
        self.assertIn('82%', joined)
        self.assertIn('ENGINE  vllm   running 48   waiting 3   kv cache 71%   prefix cache hit 42%', joined)
        self.assertIn('prompt tok/s   11.2k', joined)
        self.assertIn('gen tok/s      1.84k', joined)
        self.assertIn('ttft  p50 200 ms  p95 500 ms', joined)
        self.assertIn('GPU TIME (last 5s)   kernels 59%   graphs 0%   memcpy 2%   sync 6%   idle 39%', joined)
        self.assertIn('KERNELS  sorted by time, last 5s', joined)
        self.assertIn('flash attention prefill', joined)
        self.assertIn('ERRORS', joined)
        self.assertIn('WARNING  Profiler 1.1.0 is outdated', joined)
        self.assertIn('q quit', texts[-1])

    def test_table_row_values(self):
        _, texts = render(make_state())
        row = next(t for t in texts if 'flash attention prefill' in t)
        self.assertTrue(row.startswith(' '))   # no marker: the row is styled instead
        self.assertIn('31.4', row)
        self.assertIn('314.0', row)
        self.assertIn('1,980', row)
        self.assertIn('158.6', row)
        self.assertIn('attention', row)
        self.assertIn('flash_fwd_kernel', row)

    def test_selected_row_is_styled_from_column_zero(self):
        lines, _ = render(make_state())
        line = next(l for l in lines if 'flash attention prefill' in line_text(l))
        self.assertEqual(line[0], (' ', 'selected'))
        self.assertTrue(all(style in ('selected', 'dim') for _, style in line))

    def test_once_mode_has_no_selection_or_keys(self):
        lines, texts = render(make_state(), UiState(once=True))
        self.assertFalse(any(line and line[0][1] == 'selected' for line in lines))
        self.assertFalse(any('q quit' in t for t in texts))
        text = frame_text(render_frame(make_state(), UiState(once=True), 120, 30))
        self.assertNotIn('q quit', text)

    def test_engine_section_omitted_without_engine_metrics(self):
        _, texts = render(make_state(engine=None))
        self.assertFalse(any('ENGINE' in t for t in texts))
        self.assertTrue(any('KERNELS' in t for t in texts))

    def test_graph_mode_note_and_graph_tab(self):
        state = make_state(graph_mode='graph')
        _, texts = render(state)
        self.assertIn('cuda graph trace: graph', texts[1])
        self.assertIn('[g]', texts[1])
        _, texts = render(state, UiState(tab='graphs'))
        self.assertNotIn('(', texts[1])
        joined = '\n'.join(texts)
        self.assertIn('GRAPHS  sorted by time', joined)
        self.assertIn('graph 3f2a9c1e…', joined)
        self.assertIn('2,000', joined)

    def test_memcpy_tab_shows_bytes(self):
        _, texts = render(make_state(), UiState(tab='memcpy'))
        joined = '\n'.join(texts)
        self.assertIn('BYTES/s', joined)
        self.assertIn('5.0 MB/s', joined)
        self.assertIn('host to device', joined)

    def test_first_poll_shows_warming_up(self):
        _, texts = render(make_state(ts=[1]))
        joined = '\n'.join(texts)
        self.assertIn('warming up', joined)
        self.assertIn('no GPU activity yet', joined)

    def test_empty_state(self):
        state = TopState(endpoint='h:1')
        state.mark_unavailable('connection refused')
        lines, texts = render(state, height=20)
        self.assertEqual(len(lines), 20)
        self.assertIn('● connection refused', texts[0])
        self.assertTrue(any('no device telemetry yet' in t for t in texts))

    def test_paused_marker(self):
        _, texts = render(make_state(), UiState(paused=True))
        self.assertIn('● paused', texts[0])


class DevicesLayoutTest(unittest.TestCase):
    def test_long_device_title_drops_stats_instead_of_clipping(self):
        payload = fx.build_payload(1)
        for r in payload['resources']:
            if r['kind'] == 'device':
                r['attributes']['device.name'] = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
        for m in payload['metrics']:
            if m['name'] == 'gpu_clocks_throttle_reasons':
                m['stats']['value'] = 4
        state = TopState(window=5.0)
        state.update(payload)
        lines, texts = render(state, width=110)
        title = next(t for t in texts if t.startswith(' GPU 0'))
        self.assertLessEqual(len(title), 110)
        self.assertIn('throttle 0x4', title)       # health indicator kept
        self.assertNotIn('pcie', title)            # lowest priority dropped first
        self.assertFalse(title.endswith(' th'))


    def test_no_utilization_drops_the_sparkline_row(self):
        state = TopState(window=5.0)
        for t in (10, 11):
            payload = fx.build_payload(t)
            payload['metrics'] = [m for m in payload['metrics'] if m['name'] != 'gpu_utilization_percent']
            state.update(payload)
        _, texts = render(state)
        self.assertFalse(any(t.startswith('   util') for t in texts))
        title = next(t for t in texts if t.startswith(' GPU 0'))
        self.assertIn('util -', title)
        self.assertTrue(any(t.startswith('   mem') for t in texts))

    def test_many_devices_keep_full_rows_when_tall(self):
        state = make_state(devices=6)
        _, texts = render(state, height=60)
        gpu_lines = [t for t in texts if t.startswith(' GPU ') and t[5].isdigit()]
        self.assertEqual(len(gpu_lines), 6)
        self.assertEqual(len([t for t in texts if t.startswith('   util')]), 6)

    def test_many_devices_collapse_when_short(self):
        state = make_state(devices=6)
        _, texts = render(state, height=30)
        gpu_lines = [t for t in texts if t.startswith(' GPU ') and t[5].isdigit()]
        self.assertEqual(len(gpu_lines), 6)
        self.assertFalse(any(t.startswith('   util') for t in texts))
        self.assertIn('util 94%', gpu_lines[0])
        self.assertIn('mem 82%', gpu_lines[0])
        self.assertTrue(any('flash attention prefill' in t for t in texts))

    def test_short_terminal_keeps_table_rows(self):
        state = make_state(devices=3)
        _, texts = render(state, height=20)
        self.assertEqual(len(texts), 20)
        self.assertTrue(any('flash attention prefill' in t for t in texts))


class TableScrollTest(unittest.TestCase):
    def _many_kernels(self, n=40):
        frames = {f'_ZN3app{len(f"k{i:02d}_kernel")}k{i:02d}_kernelIfEvPf': (int((n - i) * 1e6 * 10), 100 * 10)
                  for i in range(n)}
        extra = []
        for t in (10, 15):
            extra.append({'name': 'cuda_kernels_nanoseconds', 'type': 'profile',
                          'tags': {'process.pid': '7'},
                          'stats': {'frames': [{'name': k, 'value': v * t // 10, 'samples': s * t // 10}
                                               for k, (v, s) in frames.items()]},
                          'updated_ns': 0})
        state = TopState(window=5.0)
        for t, m in zip((10, 15), extra):
            payload = fx.build_payload(t, profiles=False, engine=None)
            payload['metrics'].append(m)
            state.update(payload)
        return state

    def test_more_marker_and_scroll(self):
        state = self._many_kernels()
        ui = UiState()
        _, texts = render(state, ui, height=24)
        self.assertTrue(any('more' in t for t in texts), texts)
        self.assertEqual(ui.row_count, 40)
        self.assertGreater(ui.rows_visible, 0)
        ui.selected = 39
        lines, texts = render(state, ui, height=24)
        selected = [line_text(l) for l in lines if l and l[0][1] == 'selected']
        self.assertEqual(len(selected), 1)
        self.assertIn('k39', selected[0])
        self.assertFalse(any('more' in t for t in texts))


class OverlayTest(unittest.TestCase):
    def test_help_overlay(self):
        _, texts = render(make_state(), UiState(overlay='help'))
        joined = '\n'.join(texts)
        self.assertIn('┌─ help', joined)
        self.assertIn('totals since start', joined)
        self.assertNotIn('filter', joined)

    def test_detail_overlay(self):
        _, texts = render(make_state(), UiState(overlay='detail'))
        joined = '\n'.join(texts)
        self.assertIn('┌─ details', joined)
        self.assertIn('flash attention prefill', joined)
        self.assertIn('flash::flash_fwd_kernel'[:5], joined)
        self.assertIn('pid 41822', joined)
        self.assertIn('last 5s', joined)
        self.assertIn('since start', joined)
        _, texts = render(make_state(), UiState(overlay='detail', selected=1))
        joined = '\n'.join(texts)
        self.assertIn('pid 41900', joined)

    def test_errors_overlay(self):
        _, texts = render(make_state(), UiState(overlay='errors'))
        joined = '\n'.join(texts)
        self.assertIn('┌─ errors', joined)
        self.assertIn('WARNING', joined)

    def test_overlay_not_drawn_in_once_mode(self):
        _, texts = render(make_state(), UiState(overlay='help', once=True))
        self.assertNotIn('┌─ help', '\n'.join(texts))
