import sys
import types
import unittest
from unittest import mock

import graphsignal.watcher
from graphsignal.recorders import amdsmi_recorder
from graphsignal.recorders.amdsmi_recorder import AmdSmiRecorder
from test.test_utils import configure_test_watcher, find_metric


def has_amd_gpu() -> bool:
    try:
        import amdsmi
        amdsmi.amdsmi_init()
    except Exception:
        return False
    try:
        found = len(amdsmi.amdsmi_get_processor_handles()) > 0
    except Exception:
        found = False
    try:
        amdsmi.amdsmi_shut_down()
    except Exception:
        pass
    return found


class _FakeAmdSmiError(Exception):
    pass


def _fake_amdsmi():
    """A stand-in for the amdsmi module with one MI300X-like device, including
    the readings a virtual function reports as unavailable."""
    fake = types.SimpleNamespace()
    fake.AmdSmiMemoryType = types.SimpleNamespace(VRAM='VRAM')
    fake.AmdSmiClkType = types.SimpleNamespace(GFX='GFX', MEM='MEM')
    fake.AmdSmiTemperatureType = types.SimpleNamespace(
        EDGE='EDGE', HOTSPOT='HOTSPOT', VRAM='VRAM')
    fake.AmdSmiTemperatureMetric = types.SimpleNamespace(CURRENT='CURRENT')
    handle = object()
    fake.calls = []
    fake.amdsmi_init = lambda: fake.calls.append('init')
    fake.amdsmi_shut_down = lambda: fake.calls.append('shut_down')
    fake.amdsmi_get_processor_handles = lambda: [handle]
    fake.amdsmi_get_gpu_device_bdf = lambda h: '0000:83:00.0'
    fake.amdsmi_get_gpu_device_uuid = lambda h: 'ddff74b5-0000-1000-809c-e9b85dbbc6da'
    fake.amdsmi_get_gpu_asic_info = lambda h: {
        'market_name': 'AMD Instinct MI300X VF',
        'target_graphics_version': 'gfx942',
        'num_compute_units': 304}
    fake.amdsmi_get_gpu_memory_total = lambda h, t: 205822885888
    fake.amdsmi_get_gpu_memory_usage = lambda h, t: 157126492160
    fake.amdsmi_get_gpu_activity = lambda h: {
        'gfx_activity': 73, 'umc_activity': 41, 'mm_activity': 'N/A'}

    def temp(h, sensor, metric):
        if sensor == 'EDGE':
            raise _FakeAmdSmiError('AMDSMI_STATUS_NOT_SUPPORTED')
        return {'HOTSPOT': 41, 'VRAM': 33}[sensor]
    fake.amdsmi_get_temp_metric = temp
    fake.amdsmi_get_power_info = lambda h: {
        'socket_power': 192, 'current_socket_power': 192,
        'average_socket_power': 'N/A', 'power_limit': 750000000}
    fake.amdsmi_get_clock_info = lambda h, t: {
        'GFX': {'clk': 2097, 'min_clk': 500, 'max_clk': 2100},
        'MEM': {'clk': 900, 'min_clk': 900, 'max_clk': 1300}}[t]
    fake.amdsmi_get_pcie_info = lambda h: {
        'pcie_static': {'max_pcie_width': 16, 'pcie_interface_version': 5},
        'pcie_metric': {'pcie_width': 16, 'pcie_speed': 32000,
                        'pcie_replay_count': 7}}
    fake.amdsmi_get_gpu_total_ecc_count = lambda h: {
        'correctable_count': 2, 'uncorrectable_count': 0}
    fake.amdsmi_get_gpu_metrics_info = lambda h: {
        'throttle_status': 'N/A', 'indep_throttle_status': 'N/A'}
    return fake


class AmdSmiRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def test_setup_without_amd_gpu_is_noop(self):
        # AMD SMI present but no GPU enumerated (a CPU-only box with the
        # ROCm userspace installed): nothing is emitted, nothing raises.
        fake = _fake_amdsmi()
        fake.amdsmi_get_processor_handles = lambda: []
        with mock.patch.object(amdsmi_recorder, 'amdsmi', fake):
            recorder = AmdSmiRecorder()
            recorder.setup()
            recorder.on_tick()
            recorder.shutdown()
        self.assertEqual(fake.calls, ['init', 'shut_down'])
        self.assertEqual(self.watcher.metric_store().export(), [])

    def test_setup_without_module_is_noop(self):
        with mock.patch.object(amdsmi_recorder, 'amdsmi', None):
            recorder = AmdSmiRecorder()
            recorder.setup()
            recorder.on_tick()
            recorder.shutdown()
        self.assertEqual(self.watcher.metric_store().export(), [])

    def test_record_with_fake_amdsmi(self):
        fake = _fake_amdsmi()
        with mock.patch.object(amdsmi_recorder, 'amdsmi', fake):
            recorder = AmdSmiRecorder()
            recorder.setup()
            recorder.on_tick()
            recorder.shutdown()
        self.assertEqual(fake.calls, ['init', 'shut_down'])

        exported = self.watcher.metric_store().export()
        tags = {'device.uuid': 'ddff74b5-0000-1000-809c-e9b85dbbc6da',
                'device.bus_id': '0000:83:00.0'}

        def gauge(name):
            m = find_metric(exported, name, tags=tags, metric_type='gauge')
            self.assertIsNotNone(m, name)
            return m.datapoint['value']

        def counter(name):
            m = find_metric(exported, name, tags=tags, metric_type='counter')
            self.assertIsNotNone(m, name)
            return m.datapoint['total']

        self.assertEqual(gauge('gpu_memory_total_bytes'), 205822885888)
        self.assertEqual(gauge('gpu_memory_used_bytes'), 157126492160)
        self.assertEqual(gauge('gpu_memory_free_bytes'), 205822885888 - 157126492160)
        self.assertEqual(gauge('gpu_utilization_percent'), 73)
        self.assertEqual(gauge('gpu_memory_utilization_percent'), 41)
        # EDGE is not supported on this (virtual) device: hotspot stands in.
        self.assertEqual(gauge('gpu_temperature_celsius'), 41)
        self.assertEqual(gauge('gpu_memory_temperature_celsius'), 33)
        self.assertEqual(gauge('gpu_power_usage_watts'), 192)
        self.assertEqual(gauge('gpu_power_management_limit_watts'), 750)
        self.assertEqual(gauge('gpu_clock_sm_megahertz'), 2097)
        self.assertEqual(gauge('gpu_max_clock_sm_megahertz'), 2100)
        self.assertEqual(gauge('gpu_clock_mem_megahertz'), 900)
        self.assertEqual(gauge('gpu_pcie_link_generation'), 5)
        self.assertEqual(gauge('gpu_pcie_link_width'), 16)
        self.assertEqual(counter('gpu_pcie_replay_counter'), 7)
        self.assertEqual(counter('gpu_ecc_errors_corrected_aggregate'), 2)

        # Zero and "N/A" readings are not measured: no metric at all.
        self.assertIsNone(find_metric(exported, 'gpu_ecc_errors_uncorrected_aggregate'))
        self.assertIsNone(find_metric(exported, 'gpu_clocks_throttle_reasons'))

        resources = self.watcher.resource_store().export()
        devices = [r for r in resources if r['kind'] == 'device']
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]['tags'], tags)
        attrs = devices[0]['attributes']
        self.assertEqual(attrs['device.name'], 'AMD Instinct MI300X VF')
        self.assertEqual(attrs['architecture'], 'gfx942')
        self.assertEqual(attrs['mem_total'], '205822885888')

    def test_visible_devices_env(self):
        with mock.patch.dict('os.environ', {'HIP_VISIBLE_DEVICES': '1,3'}, clear=False):
            self.assertEqual(amdsmi_recorder._visible_device_idxs(4), [1, 3])
        with mock.patch.dict('os.environ', {'ROCR_VISIBLE_DEVICES': '2',
                                            'HIP_VISIBLE_DEVICES': '1,3'}, clear=False):
            self.assertEqual(amdsmi_recorder._visible_device_idxs(4), [2])
        with mock.patch.dict('os.environ', {'HIP_VISIBLE_DEVICES': '1,3',
                                            'LOCAL_RANK': '1'}, clear=False):
            self.assertEqual(amdsmi_recorder._current_device_idxs([1, 3]), [3])

    def test_record_on_amd_gpu(self):
        if not has_amd_gpu():
            self.skipTest("No AMD GPU available")
        recorder = AmdSmiRecorder()
        recorder.setup()
        recorder.on_tick()
        recorder.shutdown()

        exported = self.watcher.metric_store().export()
        emitted = {m.name for m in exported if m.name.startswith('gpu_')}
        self.assertTrue({'gpu_memory_total_bytes', 'gpu_memory_used_bytes',
                         'gpu_clock_sm_megahertz'} <= emitted, sorted(emitted))
        devices = [r for r in self.watcher.resource_store().export()
                   if r['kind'] == 'device']
        self.assertGreaterEqual(len(devices), 1)
        self.assertIn('device.name', devices[0]['attributes'])
        self.assertIn('architecture', devices[0]['attributes'])


if __name__ == '__main__':
    unittest.main()
