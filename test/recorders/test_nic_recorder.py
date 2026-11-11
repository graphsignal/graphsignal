import os
import shutil
import tempfile
import unittest

import graphsignal.watcher
from graphsignal.recorders import nic_recorder
from graphsignal.recorders.nic_recorder import NicRecorder
from test.test_utils import configure_test_watcher, find_metric

# PCI layout of the fake host: one PCIe switch under root port 00:01.0 with
# the NIC and GPU0 on two of its downstream ports; GPU1 on another socket.
NIC_PCI = 'pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:00.0/0000:03:00.0'
GPU0_PCI = 'pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:04.0/0000:05:00.0'
GPU1_PCI = 'pci0000:80/0000:80:01.0/0000:81:00.0'

MLX5_COUNTERS = {
    'port_xmit_data': 0, 'port_rcv_data': 0,
    'port_xmit_packets': 0, 'port_rcv_packets': 0,
    'port_xmit_wait': 0, 'port_xmit_discards': 0, 'port_rcv_errors': 0,
    'symbol_error': 0, 'link_downed': 0, 'link_error_recovery': 0,
    'VL15_dropped': 0,
}
MLX5_HW_COUNTERS = {
    'out_of_buffer': 0, 'out_of_sequence': 0, 'packet_seq_err': 0,
    'local_ack_timeout_err': 0, 'rnr_nak_retry_err': 0,
    'req_transport_retries_exceeded': 0, 'np_cnp_sent': 0,
    'rp_cnp_handled': 0, 'np_ecn_marked_roce_packets': 0,
    'roce_adp_retrans': 0, 'lifespan': 10,
}


class FakeSysfs:
    def __init__(self, root):
        self.root = root

    def _write(self, path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(f'{text}\n')

    def add_gpu(self, pci_path, vendor='0x10de', pci_class='0x030200'):
        dev_dir = os.path.join(self.root, 'devices', pci_path)
        self._write(os.path.join(dev_dir, 'class'), pci_class)
        self._write(os.path.join(dev_dir, 'vendor'), vendor)
        link_dir = os.path.join(self.root, 'bus', 'pci', 'devices')
        os.makedirs(link_dir, exist_ok=True)
        os.symlink(dev_dir, os.path.join(link_dir, os.path.basename(pci_path)))

    def add_device(self, name, pci_path=None, counters=None, hw_counters=None,
                   state='4: ACTIVE', rate='400 Gb/sec (4X NDR)',
                   link_layer='InfiniBand', netdev=None, fw_ver='28.45.4028',
                   numa_node='0'):
        if pci_path:
            pci_dir = os.path.join(self.root, 'devices', pci_path)
            dev_dir = os.path.join(pci_dir, 'infiniband', name)
            os.makedirs(dev_dir)
            os.symlink(pci_dir, os.path.join(dev_dir, 'device'))
            self._write(os.path.join(pci_dir, 'numa_node'), numa_node)
        else:
            dev_dir = os.path.join(self.root, 'devices', 'virtual', 'infiniband', name)
            os.makedirs(dev_dir)
        class_dir = os.path.join(self.root, 'class', 'infiniband')
        os.makedirs(class_dir, exist_ok=True)
        os.symlink(dev_dir, os.path.join(class_dir, name))
        if fw_ver:
            self._write(os.path.join(dev_dir, 'fw_ver'), fw_ver)
        port_dir = os.path.join(dev_dir, 'ports', '1')
        self._write(os.path.join(port_dir, 'state'), state)
        self._write(os.path.join(port_dir, 'rate'), rate)
        self._write(os.path.join(port_dir, 'link_layer'), link_layer)
        if netdev:
            self._write(os.path.join(port_dir, 'gid_attrs', 'ndevs', '0'), netdev)
        for name_, value in (counters or {}).items():
            self._write(os.path.join(port_dir, 'counters', name_), value)
        for name_, value in (hw_counters or {}).items():
            self._write(os.path.join(port_dir, 'hw_counters', name_), value)
        return port_dir

    def remove_device(self, name):
        os.remove(os.path.join(self.root, 'class', 'infiniband', name))

    def set(self, port_dir, rel, value):
        self._write(os.path.join(port_dir, rel), value)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class NicRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.root = tempfile.mkdtemp()
        self.sysfs = FakeSysfs(self.root)
        self.clock = FakeClock()

    def tearDown(self):
        graphsignal.watcher.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def _recorder(self, **kwargs):
        recorder = NicRecorder(sysfs_root=self.root, clock=self.clock, **kwargs)
        recorder.setup()
        return recorder

    def _tick(self, recorder, advance=0.0):
        self.clock.now += advance
        recorder.on_tick()

    def _metric(self, name, device='mlx5_0', metric_type=None):
        return find_metric(self.watcher.metric_store().export(), name,
                           tags={'nic.device': device, 'nic.port': '1'},
                           metric_type=metric_type)

    def _value(self, name, device='mlx5_0'):
        metric = self._metric(name, device)
        if metric is None:
            return None
        return metric.datapoint.get('total', metric.datapoint.get('value'))

    def _resources(self):
        return [r for r in self.watcher.resource_store().export()
                if r['kind'] == 'nic']

    def _add_mlx5(self, **kwargs):
        kwargs.setdefault('pci_path', NIC_PCI)
        kwargs.setdefault('counters', dict(MLX5_COUNTERS))
        kwargs.setdefault('hw_counters', dict(MLX5_HW_COUNTERS))
        return self.sysfs.add_device('mlx5_0', **kwargs)

    def test_no_rdma_devices_emits_nothing(self):
        recorder = self._recorder()
        self._tick(recorder)
        self._tick(recorder, 1)
        names = [m.name for m in self.watcher.metric_store().export()]
        self.assertFalse([n for n in names if n.startswith('nic_')])
        self.assertEqual(self._resources(), [])

    def test_traffic_bytes_from_pma_words(self):
        port = self._add_mlx5()
        recorder = self._recorder()
        self._tick(recorder)
        self.assertEqual(self._value('nic_transmit_bytes'), 0)
        self.assertEqual(self._value('nic_receive_bytes'), 0)
        self.assertEqual(self._metric('nic_transmit_bytes').type, 'counter')
        self.assertEqual(self._value('nic_port_active'), 1)
        self.assertEqual(self._value('nic_link_rate_gigabits_per_second'), 400)

        self.sysfs.set(port, 'counters/port_xmit_data', 1000)
        self.sysfs.set(port, 'counters/port_rcv_data', 250)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 4000)
        self.assertEqual(self._value('nic_receive_bytes'), 1000)

    def test_totals_count_from_first_read(self):
        port = self._add_mlx5(counters=dict(MLX5_COUNTERS, port_xmit_data=10 ** 9))
        recorder = self._recorder()
        self._tick(recorder)
        self.assertEqual(self._value('nic_transmit_bytes'), 0)
        self.sysfs.set(port, 'counters/port_xmit_data', 10 ** 9 + 5)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 20)

    def test_counter_reset_never_decreases_total(self):
        port = self._add_mlx5()
        recorder = self._recorder()
        self._tick(recorder)
        self.sysfs.set(port, 'counters/port_xmit_data', 1000)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 4000)
        self.sysfs.set(port, 'counters/port_xmit_data', 10)  # perfquery -R
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 4040)

    def test_saturated_pma_counter_holds_total(self):
        port = self._add_mlx5()
        recorder = self._recorder()
        self._tick(recorder)
        self.sysfs.set(port, 'counters/port_xmit_data', 100)
        self._tick(recorder, 1)
        self.sysfs.set(port, 'counters/port_xmit_data', 0xFFFFFFFF)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 400)
        # A 64-bit counter passing through the value just resumes.
        self.sysfs.set(port, 'counters/port_xmit_data', 0xFFFFFFFF + 10)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), (0xFFFFFFFF + 10) * 4)

    def test_packets_and_errors_on_slow_cadence(self):
        port = self._add_mlx5()
        recorder = self._recorder()
        self._tick(recorder)
        self.assertEqual(self._value('nic_transmit_packets'), 0)
        # Error counters appear only once non-zero.
        self.assertIsNone(self._metric('nic_symbol_error'))
        self.assertIsNone(self._metric('nic_out_of_buffer'))

        self.sysfs.set(port, 'counters/port_xmit_packets', 7)
        self.sysfs.set(port, 'counters/symbol_error', 3)
        self.sysfs.set(port, 'hw_counters/out_of_buffer', 2)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_packets'), 0)
        self.assertIsNone(self._metric('nic_symbol_error'))

        self._tick(recorder, nic_recorder.SLOW_INTERVAL_SEC)
        self.assertEqual(self._value('nic_transmit_packets'), 7)
        self.assertEqual(self._value('nic_symbol_error'), 3)
        self.assertEqual(self._value('nic_out_of_buffer'), 2)
        self.assertEqual(self._metric('nic_symbol_error').type, 'counter')

    def test_roce_congestion_counters_only_on_ethernet(self):
        ib_port = self._add_mlx5(link_layer='InfiniBand')
        roce_port = self.sysfs.add_device(
            'mlx5_1', pci_path='pci0000:00/0000:00:02.0/0000:06:00.0',
            counters=dict(MLX5_COUNTERS), hw_counters=dict(MLX5_HW_COUNTERS),
            link_layer='Ethernet', netdev='enp6s0np0')
        recorder = self._recorder()
        self._tick(recorder)
        self.sysfs.set(ib_port, 'hw_counters/np_cnp_sent', 5)
        self.sysfs.set(roce_port, 'hw_counters/np_cnp_sent', 5)
        self._tick(recorder, nic_recorder.SLOW_INTERVAL_SEC)
        self.assertIsNone(self._metric('nic_np_cnp_sent', 'mlx5_0'))
        self.assertEqual(self._value('nic_np_cnp_sent', 'mlx5_1'), 5)

    def test_down_port_emits_traffic_once_nonzero(self):
        port = self._add_mlx5(state='1: DOWN')
        recorder = self._recorder()
        self._tick(recorder)
        self.assertEqual(self._value('nic_port_active'), 0)
        self.assertIsNone(self._metric('nic_transmit_bytes'))
        self.assertIsNone(self._metric('nic_receive_packets'))
        self.assertIsNone(self._metric('nic_link_rate_gigabits_per_second'))
        resource, = self._resources()
        self.assertEqual(resource['attributes']['state'], 'DOWN')

        # mlx5 internal loopback: counted on a down port.
        self.sysfs.set(port, 'counters/port_rcv_data', 100)
        self.sysfs.set(port, 'counters/port_rcv_packets', 2)
        self.sysfs.set(port, 'counters/symbol_error', 1)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_receive_bytes'), 400)
        self.assertIsNone(self._metric('nic_transmit_bytes'))
        self._tick(recorder, nic_recorder.SLOW_INTERVAL_SEC)
        self.assertEqual(self._value('nic_receive_packets'), 2)
        # Error counters only on an active link.
        self.assertIsNone(self._metric('nic_symbol_error'))

        self.sysfs.set(port, 'state', '4: ACTIVE')
        self._tick(recorder, nic_recorder.SLOW_INTERVAL_SEC)
        self.assertEqual(self._value('nic_port_active'), 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 0)
        self.assertEqual(self._value('nic_link_rate_gigabits_per_second'), 400)
        # First read on an active link is the baseline for error counters.
        self.assertIsNone(self._metric('nic_symbol_error'))
        self.sysfs.set(port, 'counters/symbol_error', 3)
        self._tick(recorder, nic_recorder.SLOW_INTERVAL_SEC)
        self.assertEqual(self._value('nic_symbol_error'), 2)

    def test_efa_bytes_from_hw_counters(self):
        port = self.sysfs.add_device(
            'efa_0', pci_path='pci0000:00/0000:00:1e.0', link_layer='Unspecified',
            rate='100 Gb/sec (4X EDR)', fw_ver=None,
            hw_counters={'tx_bytes': 0, 'rx_bytes': 0, 'tx_pkts': 0,
                         'rx_pkts': 0, 'rx_drops': 0, 'retrans_pkts': 0})
        recorder = self._recorder()
        self._tick(recorder)
        self.sysfs.set(port, 'hw_counters/tx_bytes', 1000)
        self.sysfs.set(port, 'hw_counters/rx_pkts', 4)
        self.sysfs.set(port, 'hw_counters/retrans_pkts', 1)
        self._tick(recorder, nic_recorder.SLOW_INTERVAL_SEC)
        self.assertEqual(self._value('nic_transmit_bytes', 'efa_0'), 1000)
        self.assertEqual(self._value('nic_receive_packets', 'efa_0'), 4)
        self.assertEqual(self._value('nic_retrans_pkts', 'efa_0'), 1)

    def test_rxe_packets_on_fast_cadence(self):
        port = self.sysfs.add_device(
            'rxe0', link_layer='Ethernet', rate='10 Gb/sec (1X QDR)',
            fw_ver=None, netdev='eth0',
            hw_counters={'sent_pkts': 0, 'rcvd_pkts': 0, 'retry_exceeded_err': 0})
        recorder = self._recorder()
        self._tick(recorder)
        self.assertIsNone(self._metric('nic_transmit_bytes', 'rxe0'))
        self.sysfs.set(port, 'hw_counters/sent_pkts', 12)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_packets', 'rxe0'), 12)
        resource, = self._resources()
        self.assertEqual(resource['attributes']['netdev'], 'eth0')
        self.assertNotIn('bus_id', resource['attributes'])

    def test_resource_attributes_and_nearest_gpu(self):
        self.sysfs.add_gpu(GPU0_PCI)
        self.sysfs.add_gpu(GPU1_PCI)
        self.sysfs.add_gpu('pci0000:00/0000:00:1f.0', vendor='0x1a03',
                           pci_class='0x030000')  # BMC VGA
        self._add_mlx5(netdev='ibp3s0')
        recorder = self._recorder()
        self._tick(recorder)
        resource, = self._resources()
        self.assertEqual(resource['tags'], {'nic.device': 'mlx5_0', 'nic.port': '1'})
        attrs = resource['attributes']
        self.assertEqual(attrs['link_layer'], 'InfiniBand')
        self.assertEqual(attrs['state'], 'ACTIVE')
        self.assertEqual(attrs['fw_ver'], '28.45.4028')
        self.assertEqual(attrs['netdev'], 'ibp3s0')
        self.assertEqual(attrs['bus_id'], '0000:03:00.0')
        self.assertEqual(attrs['numa_node'], '0')
        self.assertEqual(attrs['nearest_gpu_bus_ids'], '0000:05:00.0')

    def test_no_nearest_gpu_through_root_complex_only(self):
        # A flat topology, as in most VMs: everything on the root bus.
        self.sysfs.add_gpu('pci0000:00/0000:00:05.0')
        self._add_mlx5(pci_path='pci0000:00/0000:00:06.0', numa_node='-1')
        recorder = self._recorder()
        self._tick(recorder)
        resource, = self._resources()
        self.assertNotIn('nearest_gpu_bus_ids', resource['attributes'])
        self.assertNotIn('numa_node', resource['attributes'])

    def test_rediscovery_adds_and_removes_devices(self):
        self._add_mlx5()
        recorder = self._recorder()
        self._tick(recorder)
        self.sysfs.add_device('rxe0', link_layer='Ethernet', fw_ver=None,
                              hw_counters={'sent_pkts': 0, 'rcvd_pkts': 0})
        self._tick(recorder, 1)
        self.assertIsNone(self._metric('nic_transmit_packets', 'rxe0'))
        self._tick(recorder, nic_recorder.DISCOVERY_INTERVAL_SEC)
        self.assertEqual(self._value('nic_port_active', 'rxe0'), 1)
        self.assertEqual(self._value('nic_transmit_packets', 'rxe0'), 0)

        self.sysfs.remove_device('rxe0')
        self._tick(recorder, nic_recorder.DISCOVERY_INTERVAL_SEC)
        self.assertEqual(sorted(recorder._ports), [('mlx5_0', '1')])

    def test_slow_reads_stretch_the_interval(self):
        port = self._add_mlx5()
        # Every pass "takes" 200 ms: 20x that is a 4 s fast interval.
        ticks = iter(range(10 ** 6))
        recorder = self._recorder(timer=lambda: next(ticks) * 0.2)
        self._tick(recorder)
        self.sysfs.set(port, 'counters/port_xmit_data', 10)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 0)
        self._tick(recorder, 3)
        self.assertEqual(self._value('nic_transmit_bytes'), 40)
        self.assertAlmostEqual(recorder._fast_interval, 4.0)
        self.assertEqual(recorder._slow_interval, nic_recorder.SLOW_INTERVAL_SEC)

    def test_unreadable_counter_is_skipped(self):
        port = self._add_mlx5()
        recorder = self._recorder()
        self._tick(recorder)
        self.sysfs.set(port, 'counters/port_xmit_data', 'N/A')
        self.sysfs.set(port, 'counters/port_rcv_data', 5)
        self._tick(recorder, 1)
        self.assertEqual(self._value('nic_transmit_bytes'), 0)
        self.assertEqual(self._value('nic_receive_bytes'), 20)


if __name__ == '__main__':
    unittest.main()
