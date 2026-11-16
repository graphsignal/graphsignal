import unittest
from unittest.mock import patch

import graphsignal.watcher
from graphsignal.recorders.dcgm_recorder import (
    SCRAPE_INTERVAL_SEC, STALENESS_METRIC, DcgmRecorder)
from graphsignal.recorders.prometheus_scraper import INITIAL_DETECT_DELAY_SEC
from test.test_utils import configure_test_watcher, find_metric

OURS = 'GPU-076474df-d02d-da62-c410-0387e73e0847'
OTHER = 'GPU-11111111-2222-3333-4444-555555555555'
URL = 'http://10.0.0.7:9400/metrics'

# dcgm-exporter text with the default `UUID` and `Hostname` labels,
# Kubernetes pod mapping labels, and DEV and PROF fields side by side.
_EXPORTER_BODY = """# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature (in C).
# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="%(ours)s",pci_bus_id="00000000:0F:00.0",device="nvidia0",modelName="NVIDIA H100 80GB HBM3",Hostname="node-a",DCGM_FI_DRIVER_VERSION="580.95.05",container="vllm",namespace="ml",pod="vllm-0"} 41
DCGM_FI_DEV_GPU_TEMP{gpu="1",UUID="%(other)s",pci_bus_id="00000000:10:00.0",device="nvidia1",modelName="NVIDIA H100 80GB HBM3",Hostname="node-a",DCGM_FI_DRIVER_VERSION="580.95.05"} 38
# HELP DCGM_FI_PROF_GR_ENGINE_ACTIVE Ratio of time the graphics engine is active.
# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="%(ours)s",pci_bus_id="00000000:0F:00.0",device="nvidia0",modelName="NVIDIA H100 80GB HBM3",Hostname="node-a",DCGM_FI_DRIVER_VERSION="580.95.05",container="vllm",namespace="ml",pod="vllm-0"} %(gr)s
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="1",UUID="%(other)s",pci_bus_id="00000000:10:00.0",device="nvidia1",modelName="NVIDIA H100 80GB HBM3",Hostname="node-a",DCGM_FI_DRIVER_VERSION="580.95.05"} 0.1
# HELP DCGM_FI_PROF_DRAM_ACTIVE Ratio of cycles the device memory interface is active sending or receiving data.
# TYPE DCGM_FI_PROF_DRAM_ACTIVE gauge
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0",UUID="%(ours)s",pci_bus_id="00000000:0F:00.0",device="nvidia0",modelName="NVIDIA H100 80GB HBM3",Hostname="node-a",DCGM_FI_DRIVER_VERSION="580.95.05",container="vllm",namespace="ml",pod="vllm-0"} 0.62
# HELP DCGM_FI_PROF_PCIE_RX_BYTES The rate of data received over the PCIe bus in bytes per second.
# TYPE DCGM_FI_PROF_PCIE_RX_BYTES gauge
DCGM_FI_PROF_PCIE_RX_BYTES{gpu="0",UUID="%(ours)s",pci_bus_id="00000000:0F:00.0",device="nvidia0",modelName="NVIDIA H100 80GB HBM3",Hostname="node-a",DCGM_FI_DRIVER_VERSION="580.95.05",container="vllm",namespace="ml",pod="vllm-0"} 1.5e+08
"""


def _body(gr='0.75', ours=OURS, other=OTHER):
    return _EXPORTER_BODY % {'ours': ours, 'other': other, 'gr': gr}


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def ns(self):
        return int(self.now * 1e9)


class DcgmRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.clock = FakeClock()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _add_device(self, uuid=OURS):
        self.watcher.update_resource('device', tags={'device.uuid': uuid},
                                     attributes={'device.name': 'NVIDIA H100'})

    def _recorder(self, url=URL):
        recorder = DcgmRecorder(root_pid=1, pid=1, exporter_url=url,
                                clock_ns=self.clock.ns)
        return recorder

    def _exported(self):
        return self.watcher.metric_store().export()

    def _dcgm_metrics(self):
        return [m for m in self._exported() if m.name.startswith('DCGM_')]

    def test_keeps_prof_fields_as_gauges_for_workload_gpus(self):
        self._add_device()
        self._recorder()._parse_and_emit(_body())

        metrics = self._dcgm_metrics()
        self.assertEqual(sorted(m.name for m in metrics), [
            'DCGM_FI_PROF_DRAM_ACTIVE', 'DCGM_FI_PROF_GR_ENGINE_ACTIVE',
            'DCGM_FI_PROF_PCIE_RX_BYTES'])
        for metric in metrics:
            self.assertEqual(metric.type, 'gauge')
            # Every exporter label is dropped; the GPU joins on device.uuid.
            self.assertEqual(metric.tags, {'device.uuid': OURS})
        dram = find_metric(self._exported(), 'DCGM_FI_PROF_DRAM_ACTIVE')
        self.assertEqual(dram.datapoint['value'], 0.62)
        # DEV fields are NVMLRecorder's job.
        self.assertIsNone(find_metric(self._exported(), 'DCGM_FI_DEV_GPU_TEMP'))

    def test_keeps_all_gpus_without_device_resources(self):
        self._recorder()._parse_and_emit(_body())
        gr = [m for m in self._dcgm_metrics()
              if m.name == 'DCGM_FI_PROF_GR_ENGINE_ACTIVE']
        self.assertEqual({m.tags['device.uuid'] for m in gr}, {OURS, OTHER})

    def test_wrong_node_warns_once_and_emits_nothing(self):
        self._add_device('GPU-99999999-8888-7777-6666-555555555555')
        recorder = self._recorder()
        with self.assertLogs('graphsignal', level='WARNING') as logs:
            recorder._parse_and_emit(_body())
        self.assertEqual(len(logs.records), 1)
        self.assertIn('none of this workload', logs.records[0].getMessage())
        self.assertIn('10.0.0.7:9400', logs.records[0].getMessage())
        self.assertEqual(self._dcgm_metrics(), [])
        with self.assertNoLogs('graphsignal', level='WARNING'):
            recorder._parse_and_emit(_body())

    def test_exporter_without_prof_fields_warns_once(self):
        self._add_device()
        body = """# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="%s"} 41
""" % OURS
        recorder = self._recorder()
        with self.assertLogs('graphsignal', level='WARNING') as logs:
            recorder._parse_and_emit(body)
        self.assertIn('no DCGM_FI_PROF_* fields', logs.records[0].getMessage())
        with self.assertNoLogs('graphsignal', level='WARNING'):
            recorder._parse_and_emit(body)

    def test_metrics_without_uuid_warn(self):
        self._add_device()
        body = """# TYPE DCGM_FI_PROF_DRAM_ACTIVE gauge
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0"} 0.5
"""
        with self.assertLogs('graphsignal', level='WARNING') as logs:
            self._recorder()._parse_and_emit(body)
        self.assertIn('no GPU UUID', logs.records[0].getMessage())
        self.assertEqual(self._dcgm_metrics(), [])

    def test_current_naming_and_configurable_uuid_label(self):
        # Current dcgm-exporter main: *_UTIL_RATIO field names, lowercase
        # `hostname`, and a UUID label whose name is configurable.
        self._add_device()
        body = """# TYPE DCGM_FI_PROF_TENSOR_UTIL_RATIO gauge
DCGM_FI_PROF_TENSOR_UTIL_RATIO{gpu="0",gpu_uuid="%s",hostname="node-a"} 0.4
# TYPE DCGM_FI_PROF_DRAM_UTIL_RATIO gauge
DCGM_FI_PROF_DRAM_UTIL_RATIO{gpu="0",custom_gpu_id="%s",hostname="node-a"} 0.7
""" % (OURS, OURS)
        self._recorder()._parse_and_emit(body)
        tensor = find_metric(self._exported(), 'DCGM_FI_PROF_TENSOR_UTIL_RATIO')
        dram = find_metric(self._exported(), 'DCGM_FI_PROF_DRAM_UTIL_RATIO')
        self.assertEqual(tensor.tags, {'device.uuid': OURS})
        self.assertEqual(dram.tags, {'device.uuid': OURS})
        self.assertEqual(dram.datapoint['value'], 0.7)

    def test_mig_instance_tag(self):
        self._add_device()
        body = """# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="%s",GPU_I_PROFILE="1g.10gb",GPU_I_ID="7"} 0.3
""" % OURS
        self._recorder()._parse_and_emit(body)
        metric = find_metric(self._exported(), 'DCGM_FI_PROF_GR_ENGINE_ACTIVE')
        self.assertEqual(metric.tags, {'device.uuid': OURS, 'mig.instance': '7'})

    def test_counter_fields_stay_counters(self):
        self._add_device()
        # dcgm-exporter's counter fields end in an uppercase _TOTAL, which
        # the TYPE line names in full.
        body = """# TYPE DCGM_FI_PROF_SM_CYCLES_ELAPSED_TOTAL counter
DCGM_FI_PROF_SM_CYCLES_ELAPSED_TOTAL{gpu="0",UUID="%s"} 123456
""" % OURS
        self._recorder()._parse_and_emit(body)
        metric = find_metric(self._exported(), 'DCGM_FI_PROF_SM_CYCLES_ELAPSED_TOTAL')
        self.assertEqual(metric.type, 'counter')
        self.assertEqual(metric.datapoint['total'], 123456)

    def test_non_finite_values_skipped(self):
        self._add_device()
        self._recorder()._parse_and_emit(_body(gr='NaN'))
        self.assertIsNone(find_metric(self._exported(), 'DCGM_FI_PROF_GR_ENGINE_ACTIVE'))
        self.assertIsNotNone(find_metric(self._exported(), 'DCGM_FI_PROF_DRAM_ACTIVE'))

    def test_update_interval_from_activity_changes(self):
        self._add_device()
        recorder = self._recorder()
        # Scraped every 5 s; the exporter refreshes every 30 s, so the value
        # changes on every sixth scrape.
        value = 0.1
        for i in range(60):
            if i % 6 == 0:
                value = round(value + 0.01, 2)
            recorder._parse_and_emit(_body(gr=str(value)))
            self.clock.now += 5.0
        gauge = find_metric(self._exported(), STALENESS_METRIC)
        self.assertEqual(gauge.type, 'gauge')
        self.assertEqual(gauge.datapoint['value'], 30 * 1e9)

    def test_update_interval_absent_while_idle(self):
        self._add_device()
        recorder = self._recorder()
        for _ in range(10):
            recorder._parse_and_emit(_body(gr='0'))
            self.clock.now += 5.0
        self.assertIsNone(find_metric(self._exported(), STALENESS_METRIC))

    def test_no_url_never_scrapes(self):
        recorder = DcgmRecorder(root_pid=1, pid=1, exporter_url=None)
        recorder.setup()
        recorder.on_tick()  # must not raise
        self.assertIsNone(recorder._scraper)

    def test_on_tick_scrapes_named_url_every_interval(self):
        self._add_device()
        recorder = self._recorder()
        recorder._scraper._clock = self.clock
        recorder.setup()
        self.clock.now += INITIAL_DETECT_DELAY_SEC
        with patch.object(recorder._scraper, '_fetch', return_value=_body()) as fetch_m:
            recorder.on_tick()
            self.clock.now += 1.0
            recorder.on_tick()  # within the interval: no fetch
            self.clock.now += SCRAPE_INTERVAL_SEC
            recorder.on_tick()
        self.assertEqual(fetch_m.call_count, 2)
        fetch_m.assert_called_with(URL)
        self.assertIsNotNone(find_metric(self._exported(), 'DCGM_FI_PROF_DRAM_ACTIVE'))


if __name__ == '__main__':
    unittest.main()
