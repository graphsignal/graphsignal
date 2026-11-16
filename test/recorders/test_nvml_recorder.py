import unittest
from unittest.mock import patch

import pytest

from types import SimpleNamespace

import pynvml

import graphsignal.watcher
from graphsignal.recorders import nvml_recorder as nvml_mod
from graphsignal.recorders.nvml_recorder import NVMLRecorder, _record_nvlink
from test.test_utils import configure_test_watcher, find_metric


def has_nvidia_gpu() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except Exception:
        return False


def has_energy_counter() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class NVMLRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def test_setup_without_gpu_is_noop(self):
        if has_nvidia_gpu():
            self.skipTest("NVIDIA GPU available; no-op path not applicable")
        recorder = NVMLRecorder()
        recorder.setup()  # must not raise
        recorder.on_tick()  # must not raise

        self.assertEqual(self.watcher.metric_store().export(), [])
        recorder.shutdown()

    @pytest.mark.cuda
    def test_record(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")
        if not has_torch():
            self.skipTest("torch not installed in the active Python environment")

        recorder = NVMLRecorder()
        recorder.setup()

        import torch
        model = torch.nn.Conv2d(1, 1, kernel_size=(1, 1))
        if torch.cuda.is_available():
            model = model.cuda()

        x = torch.arange(-5, 5, 0.1).view(1, 1, -1, 1)
        if torch.cuda.is_available():
            x = x.cuda()
        _ = model(x)

        recorder.on_tick()

        exported = self.watcher.metric_store().export()
        gpu_metrics = [m for m in exported if m.name.startswith('gpu_')]
        self.assertGreater(len(gpu_metrics), 0)
        for metric in gpu_metrics:
            self.assertIn(metric.type, ('gauge', 'counter'))

        expected_gauges = [
            'gpu_utilization_percent', 'gpu_memory_used_bytes', 'gpu_memory_total_bytes',
            'gpu_temperature_celsius', 'gpu_power_usage_watts',
        ]
        emitted = {m.name for m in gpu_metrics}
        self.assertTrue(emitted & set(expected_gauges),
                        f'no expected gpu gauges among {sorted(emitted)}')

        if has_energy_counter():
            energy = find_metric(exported, 'gpu_energy_joules')
            self.assertIsNotNone(energy)
            self.assertEqual(energy.type, 'counter')
            self.assertGreater(energy.datapoint['total'], 0)

        resources = self.watcher.resource_store().export()
        device_resources = [r for r in resources if r['kind'] == 'device']
        self.assertGreaterEqual(len(device_resources), 1)
        attrs = device_resources[0]['attributes']
        self.assertIn('device.name', attrs)
        self.assertIn('architecture', attrs)
        self.assertIn('compute_capability', attrs)

    def test_energy_emitted_as_joules_total_counter(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")

        recorder = NVMLRecorder()
        recorder.setup()

        # NVML reports millijoules since driver load; the metric is joules,
        # passed through as the cumulative total, never diffed.
        with patch('graphsignal.recorders.nvml_recorder.nvmlDeviceGetTotalEnergyConsumption',
                   return_value=1500000):
            recorder.on_tick()
        metric = find_metric(self.watcher.metric_store().export(), 'gpu_energy_joules')
        self.assertIsNotNone(metric)
        self.assertEqual(metric.type, 'counter')
        self.assertEqual(metric.datapoint['total'], 1500.0)

        with patch('graphsignal.recorders.nvml_recorder.nvmlDeviceGetTotalEnergyConsumption',
                   return_value=2500000):
            recorder.on_tick()
        metric = find_metric(self.watcher.metric_store().export(), 'gpu_energy_joules')
        self.assertEqual(metric.datapoint['total'], 2500.0)
        recorder.shutdown()

    def test_energy_not_emitted_when_unsupported(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")

        recorder = NVMLRecorder()
        recorder.setup()

        with patch('graphsignal.recorders.nvml_recorder.nvmlDeviceGetTotalEnergyConsumption',
                   side_effect=RuntimeError('not supported')):
            recorder.on_tick()  # must not raise
        self.assertIsNone(find_metric(self.watcher.metric_store().export(),
                                      'gpu_energy_joules'))
        recorder.shutdown()

    def test_xid_errors_emitted_as_running_total_counter(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")

        recorder = NVMLRecorder()
        recorder.setup()

        self.watcher.log_store().clear()

        recorder._pending_xid_error_codes[0] = [1, 2, 3]
        recorder.on_tick()

        metric = find_metric(self.watcher.metric_store().export(),
                             'gpu_xid_critical_errors')
        self.assertIsNotNone(metric)
        self.assertEqual(metric.type, 'counter')
        self.assertEqual(metric.datapoint['total'], 3)

        # A tick with no new XID errors must not re-count the drained ones.
        recorder.on_tick()
        metric = find_metric(self.watcher.metric_store().export(),
                             'gpu_xid_critical_errors')
        self.assertEqual(metric.datapoint['total'], 3)

        # Recorder-held running total: further errors add to the total.
        recorder._pending_xid_error_codes[0] = [4]
        recorder.on_tick()
        metric = find_metric(self.watcher.metric_store().export(),
                             'gpu_xid_critical_errors')
        self.assertEqual(metric.datapoint['total'], 4)

        xid_entries = [e for e in self.watcher.log_store().export()
                       if 'XID error' in (e['message'] or '')]
        self.assertEqual(len(xid_entries), 4)
        for entry in xid_entries:
            self.assertEqual(entry['level'], 'error')
        codes = {int(e['message'].split('XID error')[1].strip())
                 for e in xid_entries}
        self.assertEqual(codes, {1, 2, 3, 4})


ALL = nvml_mod.NVLINK_ALL_LINKS_SCOPE
DATA_TX = pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX
DATA_RX = pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX
RAW_TX = pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX
RAW_RX = pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX
SPEED = pynvml.NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON
LINK_SPEED = pynvml.NVML_FI_DEV_NVLINK_GET_SPEED


def _field(value=None):
    """A nvmlFieldValue_t stand-in: unsupported when value is None."""
    if value is None:
        return SimpleNamespace(nvmlReturn=pynvml.NVML_ERROR_NOT_SUPPORTED,
                               valueType=pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG,
                               value=SimpleNamespace(ullVal=0))
    return SimpleNamespace(nvmlReturn=pynvml.NVML_SUCCESS,
                           valueType=pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG,
                           value=SimpleNamespace(ullVal=value))


class FakeNvLink:
    """Patches the NVML calls _record_nvlink makes. `links` maps link id →
    active; ids past the last one raise like NVML does. `fields` maps
    (field id, scope id) → value, absent = unsupported."""

    def __init__(self, links, fields=None, field_error=None):
        self.links = links
        self.fields = fields or {}
        self.field_error = field_error
        self.error_counts = {}
        self.requests = None

    def state(self, handle, link):
        if link not in self.links:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_INVALID_ARGUMENT)
        return 1 if self.links[link] else 0

    def errors(self, handle, link, counter_type):
        return self.error_counts.get((link, counter_type), 0)

    def field_values(self, handle, requests):
        if self.field_error is not None:
            raise self.field_error
        self.requests = list(requests)
        return [_field(self.fields.get(r)) for r in requests]

    def run(self):
        self.gauges, self.counters, self.link_counters = {}, {}, {}
        with patch.object(nvml_mod, 'nvmlDeviceGetNvLinkState', side_effect=self.state), \
             patch.object(nvml_mod, 'nvmlDeviceGetNvLinkErrorCounter', side_effect=self.errors), \
             patch.object(nvml_mod, 'nvmlDeviceGetFieldValues', side_effect=self.field_values):
            _record_nvlink(
                'h',
                lambda name, value: self.gauges.__setitem__(name, value),
                lambda name, total: self.counters.__setitem__(name, total),
                lambda name, total, link: self.link_counters.__setitem__((name, link), total))
        return self


class RecordNvLinkTest(unittest.TestCase):
    def test_device_without_nvlink_emits_nothing(self):
        fake = FakeNvLink(links={}).run()
        self.assertEqual((fake.gauges, fake.counters, fake.link_counters), ({}, {}, {}))
        self.assertIsNone(fake.requests)

    def test_all_18_links_enumerated(self):
        fake = FakeNvLink(links={i: True for i in range(18)}).run()
        self.assertEqual(fake.gauges['gpu_nvlink_link_count'], 18)
        self.assertEqual(fake.gauges['gpu_nvlink_active_links'], 18)
        # Four all-links reads, four per active link, the common speed, then
        # each active link's speed.
        self.assertEqual(len(fake.requests), 4 + 18 * 4 + 1 + 18)

    def test_link_count_includes_inactive_links(self):
        fake = FakeNvLink(links={0: True, 1: True, 2: False, 3: True}).run()
        self.assertEqual(fake.gauges['gpu_nvlink_link_count'], 4)
        self.assertEqual(fake.gauges['gpu_nvlink_active_links'], 3)
        scopes = {scope for _, scope in fake.requests}
        self.assertNotIn(2, scopes)

    def test_aggregate_scope_and_per_link_counters(self):
        fields = {
            (DATA_TX, ALL): 3000, (DATA_RX, ALL): 600,
            (RAW_TX, ALL): 3300, (RAW_RX, ALL): 660,
            (DATA_TX, 0): 1000, (DATA_TX, 1): 2000,
            (DATA_RX, 0): 600, (DATA_RX, 1): 0,
            (SPEED, 0): 53125,
        }
        fake = FakeNvLink(links={0: True, 1: True}, fields=fields).run()
        self.assertEqual(fake.requests[:4], [(DATA_TX, ALL), (DATA_RX, ALL),
                                             (RAW_TX, ALL), (RAW_RX, ALL)])
        self.assertEqual(fake.counters['gpu_nvlink_throughput_data_tx_kibibytes'], 3000)
        self.assertEqual(fake.counters['gpu_nvlink_throughput_data_rx_kibibytes'], 600)
        self.assertEqual(fake.counters['gpu_nvlink_throughput_raw_tx_kibibytes'], 3300)
        self.assertEqual(fake.counters['gpu_nvlink_throughput_raw_rx_kibibytes'], 660)
        self.assertEqual(fake.link_counters, {
            ('gpu_nvlink_link_throughput_data_tx_kibibytes', 0): 1000,
            ('gpu_nvlink_link_throughput_data_tx_kibibytes', 1): 2000,
            # Link 1 RX is zero: not emitted until it moves.
            ('gpu_nvlink_link_throughput_data_rx_kibibytes', 0): 600,
        })
        self.assertEqual(fake.gauges['gpu_nvlink_speed_mbps'], 53125)

    def test_speed_from_links_without_common_speed(self):
        # H200 on driver 595: the common-speed field is unsupported, the
        # per-link speed field answers (MB/s per link and direction).
        fields = {(LINK_SPEED, 0): 26562, (LINK_SPEED, 1): 26562,
                  (LINK_SPEED, 2): 25000}
        fake = FakeNvLink(links={0: True, 1: True, 2: True}, fields=fields).run()
        self.assertIn((LINK_SPEED, 2), fake.requests)
        # The slowest active link, so speed x links never overstates the peak.
        self.assertEqual(fake.gauges['gpu_nvlink_speed_mbps'], 25000)

    def test_common_speed_wins_over_link_speeds(self):
        fields = {(SPEED, 0): 53125, (LINK_SPEED, 0): 26562}
        fake = FakeNvLink(links={0: True}, fields=fields).run()
        self.assertEqual(fake.gauges['gpu_nvlink_speed_mbps'], 53125)

    def test_falls_back_to_sum_of_links_without_aggregate_scope(self):
        fields = {(DATA_TX, 0): 1000, (DATA_TX, 1): 2000,
                  (RAW_TX, 0): 1100}  # RAW_TX link 1 unsupported: no total
        fake = FakeNvLink(links={0: True, 1: True}, fields=fields).run()
        self.assertEqual(fake.counters['gpu_nvlink_throughput_data_tx_kibibytes'], 3000)
        self.assertNotIn('gpu_nvlink_throughput_raw_tx_kibibytes', fake.counters)
        self.assertNotIn('gpu_nvlink_speed_mbps', fake.gauges)

    def test_error_counters_summed_over_active_links(self):
        fake = FakeNvLink(links={0: True, 1: True, 2: False})
        fake.error_counts = {
            (0, pynvml.NVML_NVLINK_ERROR_DL_REPLAY): 3,
            (1, pynvml.NVML_NVLINK_ERROR_DL_REPLAY): 4,
            (1, pynvml.NVML_NVLINK_ERROR_DL_CRC_FLIT): 2,
            # Inactive link: not read.
            (2, pynvml.NVML_NVLINK_ERROR_DL_RECOVERY): 9,
        }
        fake.run()
        self.assertEqual(fake.counters['gpu_nvlink_errors_replay'], 7)
        self.assertEqual(fake.counters['gpu_nvlink_errors_crc_flit'], 2)
        # Zero totals are not emitted.
        self.assertNotIn('gpu_nvlink_errors_recovery', fake.counters)
        self.assertNotIn('gpu_nvlink_errors_crc_data', fake.counters)

    def test_field_read_failure_keeps_link_gauges(self):
        fake = FakeNvLink(links={0: True},
                          field_error=pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)).run()
        self.assertEqual(fake.gauges['gpu_nvlink_link_count'], 1)
        self.assertEqual(fake.counters, {})
        self.assertEqual(fake.link_counters, {})



if __name__ == '__main__':
    unittest.main()
