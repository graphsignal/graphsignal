import logging
import os
import time

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

# RDMA NIC counters from sysfs (/sys/class/infiniband/<dev>/ports/<port>/):
# InfiniBand and RoCE HCAs (mlx5, …), AWS EFA, and Soft-RoCE (rxe). Read from
# the watcher only — nothing is injected into the workload — so this is the
# one source that also sees traffic the workload's libraries never post from
# the host: GPU-initiated RDMA (NVSHMEM IBGDA, DeepEP) and KV-cache transfer
# engines alike.
#
# The counters are per port, so per host: every process sharing a NIC shows
# up in the same totals. The recorder runs for the root target only.
#
# Traffic counters are read on every port; error counters and the link rate
# only on ACTIVE ones.
#
# Reads are not free. On mlx5 every counter file is a firmware command
# (measured on ConnectX-7: ~0.4 ms per counters/ file, ~2.4 ms per
# hw_counters/ file, ~1.7 ms for state), so only a curated set is read, and
# each group's interval stretches to keep its read time under 1/DUTY_FACTOR
# of wall time if the firmware is slower than that.
#
# Counters are totals since the recorder's first read, kept by the recorder
# from per-read increases: a NIC counter reset (perfquery -R, driver reload)
# never makes a total go backwards.

FAST_INTERVAL_SEC = 1.0
SLOW_INTERVAL_SEC = 30.0
DISCOVERY_INTERVAL_SEC = 60.0
DUTY_FACTOR = 20
# A tick landing slightly before a group's due time still runs it, so a 1s
# group on a 1s tick loop does not slip to every other tick.
DUE_SLACK_SEC = 0.1

# A basic (non-extended) PMA counter saturates at 2^32-1 instead of wrapping.
PMA_SATURATED = 0xFFFFFFFF

# metric name -> candidate sources (path under the port dir, scale), first
# existing wins: IB/RoCE PMA counters, then EFA, then rxe.
# port_xmit_data/port_rcv_data count 4-byte words.
_BYTES_SOURCES = {
    'nic_transmit_bytes': (('counters/port_xmit_data', 4), ('hw_counters/tx_bytes', 1)),
    'nic_receive_bytes': (('counters/port_rcv_data', 4), ('hw_counters/rx_bytes', 1)),
}
_PACKETS_SOURCES = {
    'nic_transmit_packets': (('counters/port_xmit_packets', 1), ('hw_counters/tx_pkts', 1),
                             ('hw_counters/sent_pkts', 1)),
    'nic_receive_packets': (('counters/port_rcv_packets', 1), ('hw_counters/rx_pkts', 1),
                            ('hw_counters/rcvd_pkts', 1)),
}

_TRAFFIC_NAMES = frozenset(_BYTES_SOURCES) | frozenset(_PACKETS_SOURCES)

# Error, retransmission and congestion counters, emitted as nic_<file name>
# once non-zero. Read only where the file exists.
_ERROR_FILES = (
    # IB PMA (all IB/RoCE HCAs)
    'counters/port_xmit_wait',
    'counters/port_xmit_discards',
    'counters/port_rcv_errors',
    'counters/symbol_error',
    'counters/link_downed',
    'counters/link_error_recovery',
    # mlx5
    'hw_counters/out_of_buffer',
    'hw_counters/out_of_sequence',
    'hw_counters/packet_seq_err',
    'hw_counters/local_ack_timeout_err',
    'hw_counters/rnr_nak_retry_err',
    'hw_counters/req_transport_retries_exceeded',
    # EFA
    'hw_counters/rx_drops',
    'hw_counters/retrans_pkts',
    'hw_counters/retrans_timeout_events',
    'hw_counters/unresponsive_remote_events',
    'hw_counters/impaired_remote_conn_events',
    # rxe
    'hw_counters/retry_exceeded_err',
)
# mlx5 RoCE congestion control and adaptive retransmission: meaningless on an
# InfiniBand link, so not read there.
_ROCE_ERROR_FILES = (
    'hw_counters/np_cnp_sent',
    'hw_counters/rp_cnp_handled',
    'hw_counters/np_ecn_marked_roce_packets',
    'hw_counters/roce_adp_retrans',
)

_GPU_VENDORS = ('0x10de', '0x1002')


class _Counter:
    __slots__ = ('path', 'scale', 'last', 'total', 'saturated_logged')

    def __init__(self, path, scale):
        self.path = path
        self.scale = scale
        self.last = None
        self.total = 0
        self.saturated_logged = False


class _Port:
    def __init__(self, device, port, port_dir):
        self.device = device
        self.port = port
        self.port_dir = port_dir
        self.tags = {'nic.device': device, 'nic.port': port}
        self.active = False
        self.link_layer = None
        self.fast = {}    # metric name -> _Counter
        self.slow = {}    # metric name -> _Counter
        self.attributes = {}


class NicRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None,
                 sysfs_root='/sys', clock=time.monotonic,
                 timer=time.perf_counter):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._sysfs_root = sysfs_root
        self._clock = clock
        self._timer = timer
        self._ports = {}  # (device, port) -> _Port
        self._first_seen_ts = None
        self._next_discovery = None
        self._next_slow = None
        self._next_fast = None
        self._slow_interval = SLOW_INTERVAL_SEC
        self._fast_interval = FAST_INTERVAL_SEC

    def setup(self):
        self._first_seen_ts = time.time_ns()
        now = self._clock()
        self._discover()
        self._next_discovery = now + DISCOVERY_INTERVAL_SEC
        self._next_slow = now
        self._next_fast = now
        if self._ports:
            logger.debug('NIC recorder: %d RDMA port(s): %s', len(self._ports),
                         ', '.join(f'{d}/{p}' for d, p in sorted(self._ports)))

    def on_tick(self):
        if self._next_discovery is None:
            return
        now = self._clock()
        if now + DUE_SLACK_SEC >= self._next_discovery:
            if self._discover():
                # Read a new port's state now rather than up to a slow
                # interval later.
                self._next_slow = now
            self._next_discovery = now + DISCOVERY_INTERVAL_SEC
        if not self._ports:
            return

        # Slow before fast: the slow pass reads link state, which decides
        # whether a port's zero traffic totals are emitted.
        if now + DUE_SLACK_SEC >= self._next_slow:
            started = self._timer()
            self._read_slow()
            elapsed = self._timer() - started
            self._slow_interval = max(SLOW_INTERVAL_SEC, elapsed * DUTY_FACTOR)
            self._next_slow = now + self._slow_interval

        if now + DUE_SLACK_SEC >= self._next_fast:
            started = self._timer()
            self._read_fast()
            elapsed = self._timer() - started
            interval = max(FAST_INTERVAL_SEC, elapsed * DUTY_FACTOR)
            if interval > FAST_INTERVAL_SEC and self._fast_interval == FAST_INTERVAL_SEC:
                logger.debug('NIC recorder: counter reads took %.1f ms, '
                             'reading every %.1f s', elapsed * 1e3, interval)
            self._fast_interval = interval
            self._next_fast = now + self._fast_interval

    # ------------------------------------------------------------ discovery

    def _discover(self):
        class_dir = os.path.join(self._sysfs_root, 'class', 'infiniband')
        try:
            devices = sorted(os.listdir(class_dir))
        except OSError:
            devices = []

        gpus = None
        seen = set()
        added = False
        for device in devices:
            ports_dir = os.path.join(class_dir, device, 'ports')
            try:
                port_names = sorted(os.listdir(ports_dir))
            except OSError:
                continue
            for port_name in port_names:
                key = (device, port_name)
                seen.add(key)
                if key in self._ports:
                    continue
                if gpus is None:
                    gpus = self._gpu_pci_paths()
                port = self._new_port(device, port_name,
                                      os.path.join(ports_dir, port_name), gpus)
                self._ports[key] = port
                added = True

        for key in list(self._ports):
            if key not in seen:
                del self._ports[key]
        return added

    def _new_port(self, device, port_name, port_dir, gpus):
        port = _Port(device, port_name, port_dir)
        port.link_layer = _read_text(os.path.join(port_dir, 'link_layer'))

        for name, sources in _BYTES_SOURCES.items():
            counter = _first_existing(port_dir, sources)
            if counter:
                port.fast[name] = counter
        # Packets ride the fast pass only where there are no byte counters
        # (rxe), so a port always has a traffic signal at full resolution.
        packets_target = port.slow if port.fast else port.fast
        for name, sources in _PACKETS_SOURCES.items():
            counter = _first_existing(port_dir, sources)
            if counter:
                packets_target[name] = counter

        error_files = _ERROR_FILES
        if port.link_layer != 'InfiniBand':
            error_files = error_files + _ROCE_ERROR_FILES
        for rel in error_files:
            path = os.path.join(port_dir, rel)
            if os.path.exists(path):
                port.slow['nic_' + os.path.basename(rel)] = _Counter(path, 1)

        attrs = {}
        if port.link_layer:
            attrs['link_layer'] = port.link_layer
        device_dir = os.path.join(self._sysfs_root, 'class', 'infiniband', device)
        fw_ver = _read_text(os.path.join(device_dir, 'fw_ver'))
        if fw_ver:
            attrs['fw_ver'] = fw_ver
        netdev = _read_text(os.path.join(port_dir, 'gid_attrs', 'ndevs', '0'))
        if not netdev:
            try:
                netdevs = os.listdir(os.path.join(device_dir, 'device', 'net'))
                if len(netdevs) == 1:
                    netdev = netdevs[0]
            except OSError:
                pass
        if netdev:
            attrs['netdev'] = netdev
        pci_path = _pci_path(device_dir)
        if pci_path:
            attrs['bus_id'] = pci_path[-1]
            numa = _read_text(os.path.join(device_dir, 'device', 'numa_node'))
            if numa and numa != '-1':
                attrs['numa_node'] = numa
            nearest = _nearest_gpus(pci_path, gpus or {})
            if nearest:
                attrs['nearest_gpu_bus_ids'] = ','.join(nearest)
        port.attributes = attrs
        return port

    def _gpu_pci_paths(self):
        """GPU bus id -> PCI path components, from sysfs, vendor-neutral."""
        gpus = {}
        pci_dir = os.path.join(self._sysfs_root, 'bus', 'pci', 'devices')
        try:
            entries = os.listdir(pci_dir)
        except OSError:
            return gpus
        for entry in entries:
            dev_dir = os.path.join(pci_dir, entry)
            pci_class = _read_text(os.path.join(dev_dir, 'class')) or ''
            vendor = _read_text(os.path.join(dev_dir, 'vendor')) or ''
            if not pci_class.startswith('0x03') or vendor not in _GPU_VENDORS:
                continue
            path = _path_components(dev_dir)
            if path:
                gpus[entry.lower()] = path
        return gpus

    # ------------------------------------------------------------ reads

    def _read_slow(self):
        watcher = graphsignal.watcher.watcher()
        for port in list(self._ports.values()):
            now_ns = time.time_ns()
            state = _read_text(os.path.join(port.port_dir, 'state'))
            if state is None:
                continue
            # "4: ACTIVE"
            port.active = state.split(':')[0].strip() == '4'
            watcher.set_gauge(name='nic_port_active', value=1 if port.active else 0,
                              measurement_ts=now_ns, tags=port.tags)
            attrs = dict(port.attributes)
            attrs['state'] = state.split(':')[-1].strip()
            if port.active:
                rate = _parse_rate(_read_text(os.path.join(port.port_dir, 'rate')))
                if rate is not None:
                    watcher.set_gauge(name='nic_link_rate_gigabits_per_second',
                                      value=rate, measurement_ts=now_ns,
                                      tags=port.tags)
            for name, counter in port.slow.items():
                if name in _TRAFFIC_NAMES:
                    self._update(watcher, port, name, counter, now_ns,
                                 emit_zero=port.active)
                elif port.active:
                    self._update(watcher, port, name, counter, now_ns,
                                 emit_zero=False)
            watcher.update_resource('nic', tags=port.tags, attributes=attrs,
                                    first_seen_ts=self._first_seen_ts)

    def _read_fast(self):
        watcher = graphsignal.watcher.watcher()
        # Every port, not only ACTIVE ones: mlx5 loops RDMA between QPs of
        # one function inside the NIC even with the link down, and that
        # traffic is counted (port_rcv_data). A down port's counters appear
        # once non-zero, so idle unused ports add no series.
        for port in list(self._ports.values()):
            now_ns = time.time_ns()
            for name, counter in port.fast.items():
                self._update(watcher, port, name, counter, now_ns,
                             emit_zero=port.active)

    def _update(self, watcher, port, name, counter, now_ns, emit_zero):
        raw = _read_int(counter.path)
        if raw is None:
            return
        if raw == PMA_SATURATED and '/counters/' in counter.path:
            # A basic PMA counter stuck at its maximum. Hold the last total
            # rather than report a flat line as "no traffic"; a 64-bit
            # counter passing through this value resumes on the next read.
            if not counter.saturated_logged:
                counter.saturated_logged = True
                logger.debug('NIC recorder: %s is saturated (no extended counters)',
                             counter.path)
            return
        if counter.last is not None:
            if raw >= counter.last:
                counter.total += (raw - counter.last) * counter.scale
            else:
                # Reset since the last read: everything it counted since is new.
                counter.total += raw * counter.scale
        counter.last = raw
        if counter.total > 0 or emit_zero:
            watcher.set_counter(name=name, total=counter.total,
                                measurement_ts=now_ns, tags=port.tags)


def _first_existing(port_dir, sources):
    for rel, scale in sources:
        path = os.path.join(port_dir, rel)
        if os.path.exists(path):
            return _Counter(path, scale)
    return None


def _read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path):
    text = _read_text(path)
    if not text:
        return None
    try:
        return int(text.split()[0])
    except ValueError:
        return None


def _parse_rate(text):
    # "400 Gb/sec (4X NDR)", "100 Gb/sec (2X HDR)", "2.5 Gb/sec (1X SDR)"
    if not text:
        return None
    try:
        value = float(text.split()[0])
    except (ValueError, IndexError):
        return None
    return int(value) if value.is_integer() else value


def _path_components(dev_dir):
    """['pci0000:00', '0000:00:01.0', '0000:05:00.0'] for a PCI device dir."""
    if not os.path.exists(dev_dir):
        # rxe and other software devices have no PCI parent.
        return None
    try:
        real = os.path.realpath(dev_dir)
    except OSError:
        return None
    parts = real.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith('pci') and ':' in part:
            return [p.lower() for p in parts[i:]]
    return None


def _pci_path(ib_device_dir):
    return _path_components(os.path.join(ib_device_dir, 'device'))


def _nearest_gpus(nic_path, gpus):
    """GPUs behind the same PCIe switch as the NIC (nvidia-smi topo PIX/PXB):
    the deepest common ancestor shared with any GPU, if it is a bridge below
    the root complex. Empty when GPUs are only reachable through the root
    complex or across sockets (or the topology is flattened, as in most VMs)."""
    best, best_depth = [], 0
    for bus_id, gpu_path in gpus.items():
        depth = 0
        for a, b in zip(nic_path[:-1], gpu_path[:-1]):
            if a != b:
                break
            depth += 1
        if depth > best_depth:
            best, best_depth = [bus_id], depth
        elif depth == best_depth and depth > 0:
            best.append(bus_id)
    # Depth counts shared ancestors: 1 = the root complex only, 2 = also a
    # root port (only functions of one multi-function device share one, a
    # root port being point-to-point), 3+ = a PCIe switch in common.
    if best_depth < 3:
        return []
    return sorted(best)
