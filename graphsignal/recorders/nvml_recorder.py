import logging
import os
import time
import socket

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder

# nvidia-ml-py is a declared dependency, but the profiler is routinely
# installed with --no-deps into a container somebody else built, and an AMD
# image has no reason to carry it. A missing module disables this recorder
# only; the watcher wires the others regardless.
try:
    from pynvml import *
    _NVML_IMPORT_ERROR = None
except Exception as _exc:
    _NVML_IMPORT_ERROR = _exc

# after pynvml import to avoid import conflicts
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger('graphsignal')

# Every metric is named after the NVML API it is read from and carries the
# value as NVML reports it: gauges are instantaneous readings, counters are
# NVML's own cumulative totals. Nothing is diffed or derived here — rates and
# utilization ratios are computed by whoever reads consecutive snapshots.

MAX_NVLINK_LINKS = 6


class NVMLRecorder(BaseRecorder):
    MIN_SAMPLE_READ_INTERVAL_US = int(10 * 1e6)

    def __init__(self, root_pid=None, pid=None, args=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._is_initialized: bool = False
        self._setup_us: Optional[int] = None
        # Cumulative XID error totals per device index (counters carry totals).
        self._xid_error_totals: Dict[int, int] = {}

        # XID event monitoring
        self._event_sets: Dict[int, Any] = {}  # device_idx -> event_set
        self._pending_xid_error_codes: Dict[int, List[int]] = {}

        self._gpu_first_seen_ts: Optional[int] = None

        self._hostname: Optional[str] = None
        try:
            self._hostname = socket.gethostname()
        except BaseException:
            logger.debug('Error reading hostname', exc_info=True)

        self._visible_device_idxs: list[int] = []
        self._current_device_idxs: list[int] = []

    def setup(self):
        if _NVML_IMPORT_ERROR is not None:
            logger.debug('pynvml not importable (%s), skipping GPU usage',
                         _NVML_IMPORT_ERROR)
            return
        try:
            nvmlInit()
            self._is_initialized = True
            logger.debug('Initialized NVML')
        except BaseException:
            logger.debug('Error initializing NVML, skipping GPU usage')
            return

        self._setup_us = int(time.time() * 1e6)
        self._gpu_first_seen_ts = time.time_ns()

        device_count = nvmlDeviceGetCount()
        if device_count == 0:
            return

        # Get visible device idxs
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            try:
                self._visible_device_idxs = [int(x) for x in cuda_visible_devices.split(',')]
            except Exception:
                self._visible_device_idxs = list(range(device_count))
        else:
            self._visible_device_idxs = list(range(device_count))

        # Get current device idxs via local rank
        local_rank = None
        for env_var in ["LOCAL_RANK", "SLURM_LOCALID", "NCCL_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"]:
            if env_var in os.environ:
                try:
                    local_rank = int(os.environ[env_var])
                except ValueError:
                    pass
                break

        if local_rank is not None:
            if local_rank < len(self._visible_device_idxs):
                self._current_device_idxs = [self._visible_device_idxs[local_rank]]
            else:
                logger.debug(f'Local rank {local_rank} is out of visible device idxs {self._visible_device_idxs}')
        # Fallback to all visible devices if no current devices are set
        if len(self._current_device_idxs) == 0:
            self._current_device_idxs = self._visible_device_idxs.copy()

        self._setup_error_monitoring()

    def _setup_error_monitoring(self):
        if not self._is_initialized:
            return

        try:
            for idx in self._current_device_idxs:
                try:
                    handle = nvmlDeviceGetHandleByIndex(idx)

                    # Create event set for this device
                    event_set = nvmlEventSetCreate()
                    self._event_sets[idx] = event_set
                    self._pending_xid_error_codes[idx] = []

                    # Register for XID critical errors
                    try:
                        nvmlDeviceRegisterEvents(handle, nvmlEventTypeXidCriticalError, event_set)
                        logger.debug(f'Registered for XID events on device {idx}')
                    except Exception as err:
                        if str(err) == "Not Supported":
                            logger.debug(f'XID event monitoring not supported on device {idx}')
                        else:
                            logger.warning(f'Failed to register for XID events on device {idx}: {err}')

                except Exception as err:
                    logger.warning(f'Failed to setup error monitoring for device {idx}: {err}')

        except Exception as e:
            logger.warning(f'Error setting up error monitoring: {e}')

    def _check_for_errors(self):
        for device_idx, event_set in self._event_sets.items():
            try:
                # Non-blocking check for events (timeout=0)
                try:
                    event = nvmlEventSetWait_v2(event_set, 0)
                except Exception as err:
                    event = nvmlEventSetWait(event_set, 0)

                if event.eventType & nvmlEventTypeXidCriticalError:
                    error_code = event.eventData
                    self._pending_xid_error_codes[device_idx].append(error_code)
            except Exception as err:
                if hasattr(err, 'value') and err.value == NVML_ERROR_TIMEOUT:
                    pass
                else:
                    _log_nvml_error(err)

    def shutdown(self):
        if not self._is_initialized:
            return

        # Clean up event sets
        for device_idx, event_set in self._event_sets.items():
            try:
                nvmlEventSetFree(event_set)
                logger.debug(f'Freed event set for device {device_idx}')
            except Exception as err:
                _log_nvml_error(err)

        self._event_sets.clear()
        self._pending_xid_error_codes.clear()

        try:
            nvmlShutdown()
            self._is_initialized = False
        except BaseException:
            logger.error('Error shutting down NVML', exc_info=True)

    def on_tick(self):
        if not self._is_initialized:
            return

        now_ns = time.time_ns()
        now_us = int(time.time() * 1e6)

        self._check_for_errors()

        watcher = graphsignal.watcher.watcher()

        for idx in self._current_device_idxs:
            try:
                handle = nvmlDeviceGetHandleByIndex(idx)
            except Exception as err:
                _log_nvml_error(err)
                continue

            device_tags = {}
            try:
                pci_info = nvmlDeviceGetPciInfo_v3(handle)
                device_tags['device.bus_id'] = pci_info.busId
            except Exception:
                try:
                    pci_info = nvmlDeviceGetPciInfo(handle)
                    device_tags['device.bus_id'] = pci_info.busId
                except Exception as err:
                    _log_nvml_error(err)
            try:
                device_tags['device.uuid'] = nvmlDeviceGetUUID(handle)
            except Exception as err:
                _log_nvml_error(err)

            def gauge(name, value):
                watcher.set_gauge(name=name, value=value,
                                  measurement_ts=now_ns, tags=device_tags)

            def counter(name, total):
                watcher.set_counter(name=name, total=total,
                                    measurement_ts=now_ns, tags=device_tags)

            # nvmlDeviceGetMemoryInfo
            mem_total = 0
            try:
                try:
                    mem_info = nvmlDeviceGetMemoryInfo_v2(handle)
                    mem_reserved = mem_info.reserved
                except Exception:
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    mem_reserved = 0  # Not available in v1
                mem_total = mem_info.total
                if mem_info.used > 0:
                    gauge('gpu_memory_used_bytes', mem_info.used)
                if mem_info.free > 0:
                    gauge('gpu_memory_free_bytes', mem_info.free)
                if mem_info.total > 0:
                    gauge('gpu_memory_total_bytes', mem_info.total)
                if mem_reserved > 0:
                    gauge('gpu_memory_reserved_bytes', mem_reserved)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetSamples: average over the interval since the last
            # read, floored at setup so a fresh run never reads another's window.
            try:
                last_read_us = max(
                    int(self._setup_us or 0),
                    now_us - NVMLRecorder.MIN_SAMPLE_READ_INTERVAL_US)

                sample_value_type, gpu_samples = nvmlDeviceGetSamples(
                    handle, NVML_GPU_UTILIZATION_SAMPLES, last_read_us)
                gpu_utilization = _avg_sample_value(sample_value_type, gpu_samples)
                if gpu_utilization > 0:
                    gauge('gpu_utilization_percent', gpu_utilization)

                sample_value_type, mem_samples = nvmlDeviceGetSamples(
                    handle, NVML_MEMORY_UTILIZATION_SAMPLES, last_read_us)
                memory_utilization = _avg_sample_value(sample_value_type, mem_samples)
                if memory_utilization > 0:
                    gauge('gpu_memory_utilization_percent', memory_utilization)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetTemperature
            try:
                temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                if temperature > 0:
                    gauge('gpu_temperature_celsius', temperature)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetPowerUsage / nvmlDeviceGetPowerManagementLimit
            try:
                power_usage = nvmlDeviceGetPowerUsage(handle) / 1000.0
                if power_usage > 0:
                    gauge('gpu_power_usage_watts', power_usage)
            except Exception as err:
                _log_nvml_error(err)

            try:
                power_limit = nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                if power_limit > 0:
                    gauge('gpu_power_management_limit_watts', power_limit)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetFanSpeed
            try:
                fan_speed = nvmlDeviceGetFanSpeed(handle)
                if fan_speed > 0:
                    gauge('gpu_fan_speed_percent', fan_speed)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetClockInfo / nvmlDeviceGetMaxClockInfo
            try:
                clock_sm = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
                if clock_sm > 0:
                    gauge('gpu_clock_sm_megahertz', clock_sm)
            except Exception as err:
                _log_nvml_error(err)

            try:
                clock_mem = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
                if clock_mem > 0:
                    gauge('gpu_clock_mem_megahertz', clock_mem)
            except Exception as err:
                _log_nvml_error(err)

            try:
                max_clock_sm = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)
                if max_clock_sm > 0:
                    gauge('gpu_max_clock_sm_megahertz', max_clock_sm)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetCurrentClocksThrottleReasons (bitmask)
            try:
                throttle_reasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                if throttle_reasons > 0:
                    gauge('gpu_clocks_throttle_reasons', throttle_reasons)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetPerformanceState (0 is P0, the highest state)
            try:
                gauge('gpu_performance_state', nvmlDeviceGetPerformanceState(handle))
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetPcieThroughput reports KB/s sampled by NVML itself.
            try:
                pcie_tx = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)
                if pcie_tx > 0:
                    gauge('gpu_pcie_throughput_tx_kilobytes_per_second', pcie_tx)
                pcie_rx = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)
                if pcie_rx > 0:
                    gauge('gpu_pcie_throughput_rx_kilobytes_per_second', pcie_rx)
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetCurrPcieLinkGeneration / Width: the link capability,
            # for deriving utilization from the throughput gauges.
            try:
                gauge('gpu_pcie_link_generation', nvmlDeviceGetCurrPcieLinkGeneration(handle))
                gauge('gpu_pcie_link_width', nvmlDeviceGetCurrPcieLinkWidth(handle))
            except Exception as err:
                _log_nvml_error(err)

            # nvmlDeviceGetPcieReplayCounter (cumulative)
            try:
                pcie_replay = nvmlDeviceGetPcieReplayCounter(handle)
                if pcie_replay > 0:
                    counter('gpu_pcie_replay_counter', pcie_replay)
            except Exception as err:
                _log_nvml_error(err)

            # NVML_FI_DEV_NVLINK_THROUGHPUT_*: cumulative KiB totals.
            for field_id, name in [
                (NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, 'gpu_nvlink_throughput_data_tx_kibibytes'),
                (NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, 'gpu_nvlink_throughput_data_rx_kibibytes'),
                (NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX, 'gpu_nvlink_throughput_raw_tx_kibibytes'),
                (NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX, 'gpu_nvlink_throughput_raw_rx_kibibytes'),
            ]:
                try:
                    field_value = nvmlDeviceGetFieldValues(handle, [field_id])[0]
                    if field_value.nvmlReturn == NVML_SUCCESS:
                        value = _nvml_value(field_value.valueType, field_value.value)
                        if value is not None and value > 0:
                            counter(name, value)
                except Exception as err:
                    _log_nvml_error(err)

            # nvmlDeviceGetNvLinkState / nvmlDeviceGetNvLinkErrorCounter
            try:
                link_count = 0
                active_links = 0
                error_totals = {
                    'gpu_nvlink_errors_replay': [NVML_NVLINK_ERROR_DL_REPLAY, 0],
                    'gpu_nvlink_errors_recovery': [NVML_NVLINK_ERROR_DL_RECOVERY, 0],
                    'gpu_nvlink_errors_crc': [NVML_NVLINK_ERROR_DL_CRC, 0],
                    'gpu_nvlink_errors_minor': [NVML_NVLINK_ERROR_DL_MINOR, 0],
                    'gpu_nvlink_errors_major': [NVML_NVLINK_ERROR_DL_MAJOR, 0],
                    'gpu_nvlink_errors_fatal': [NVML_NVLINK_ERROR_DL_FATAL, 0],
                }
                for link in range(MAX_NVLINK_LINKS):
                    try:
                        state = nvmlDeviceGetNvLinkState(handle, link)
                    except Exception:
                        break
                    if not state:
                        continue
                    link_count += 1
                    active_links += 1
                    for name, entry in error_totals.items():
                        try:
                            entry[1] += nvmlDeviceGetNvLinkErrorCounter(handle, link, entry[0])
                        except Exception as err:
                            _log_nvml_error(err)

                if link_count > 0:
                    gauge('gpu_nvlink_link_count', link_count)
                    gauge('gpu_nvlink_active_links', active_links)
                    for name, entry in error_totals.items():
                        if entry[1] > 0:
                            counter(name, entry[1])
            except Exception:
                # NVLINK not available on this device
                pass

            # nvmlDeviceGetTotalEccErrors (cumulative)
            for name, error_type, counter_type in [
                ('gpu_ecc_errors_corrected_volatile', NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC),
                ('gpu_ecc_errors_uncorrected_volatile', NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_VOLATILE_ECC),
                ('gpu_ecc_errors_corrected_aggregate', NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_AGGREGATE_ECC),
                ('gpu_ecc_errors_uncorrected_aggregate', NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_AGGREGATE_ECC),
            ]:
                try:
                    total = nvmlDeviceGetTotalEccErrors(handle, error_type, counter_type)
                    if total > 0:
                        counter(name, total)
                except Exception as err:
                    _log_nvml_error(err)

            # XID events, drained from the event sets: the recorder keeps the
            # running total (counters carry totals) and each code is logged once.
            new_xid_error_codes = self._pending_xid_error_codes.get(idx, [])
            if new_xid_error_codes:
                self._pending_xid_error_codes[idx] = []
                self._xid_error_totals[idx] = (
                    self._xid_error_totals.get(idx, 0) + len(new_xid_error_codes))
                for xid_error_code in new_xid_error_codes:
                    watcher.log_message(
                        message=f'XID error {xid_error_code}',
                        tags=device_tags,
                        level='error')
            if self._xid_error_totals.get(idx, 0) > 0:
                counter('gpu_xid_critical_errors', self._xid_error_totals[idx])

            # Device resource
            resource_attrs = {}
            try:
                device_name = nvmlDeviceGetName(handle)
                if device_name:
                    resource_attrs['device.name'] = device_name
            except Exception as err:
                _log_nvml_error(err)
            try:
                resource_attrs['architecture'] = _arch_name(nvmlDeviceGetArchitecture(handle))
            except Exception as err:
                _log_nvml_error(err)
            try:
                cc_major, cc_minor = nvmlDeviceGetCudaComputeCapability(handle)
                resource_attrs['compute_capability'] = f'{cc_major}.{cc_minor}'
            except Exception as err:
                _log_nvml_error(err)
            if mem_total > 0:
                resource_attrs['mem_total'] = str(mem_total)
            watcher.update_resource(
                'device',
                tags=device_tags,
                attributes=resource_attrs,
                first_seen_ts=self._gpu_first_seen_ts)


def _arch_name(arch) -> str:
    names = {
        NVML_DEVICE_ARCH_KEPLER: 'Kepler',
        NVML_DEVICE_ARCH_MAXWELL: 'Maxwell',
        NVML_DEVICE_ARCH_PASCAL: 'Pascal',
        NVML_DEVICE_ARCH_VOLTA: 'Volta',
        NVML_DEVICE_ARCH_TURING: 'Turing',
        NVML_DEVICE_ARCH_AMPERE: 'Ampere',
        NVML_DEVICE_ARCH_ADA: 'Ada',
        NVML_DEVICE_ARCH_HOPPER: 'Hopper',
        NVML_DEVICE_ARCH_BLACKWELL: 'Blackwell',
    }
    return names.get(arch, f'Unknown({arch})')


def _avg_sample_value(sample_value_type, samples):
    if not samples:
        return 0.0

    sample_values = []

    if sample_value_type == NVML_VALUE_TYPE_DOUBLE:
        sample_values = [sample.sampleValue.dVal for sample in samples]
    if sample_value_type == NVML_VALUE_TYPE_UNSIGNED_INT:
        sample_values = [sample.sampleValue.uiVal for sample in samples]
    if sample_value_type == NVML_VALUE_TYPE_UNSIGNED_LONG:
        sample_values = [sample.sampleValue.ulVal for sample in samples]
    if sample_value_type == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        sample_values = [sample.sampleValue.ullVal for sample in samples]

    if len(sample_values) > 0:
        return sum(sample_values) / len(sample_values)

    return 0.0


def _nvml_value(value_type, value) -> Optional[Union[int, float]]:
    if value_type == NVML_VALUE_TYPE_DOUBLE:
        return value.dVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_INT:
        return value.uiVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG:
        return value.ulVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        return value.ullVal
    return None


def _log_nvml_error(err):
    if hasattr(err, 'value'):
        if (err.value == NVML_ERROR_NOT_FOUND):
            pass
        elif (err.value == NVML_ERROR_NOT_SUPPORTED):
            pass
        elif (err.value == NVML_ERROR_INVALID_ARGUMENT):
            logger.debug(f'NVML call invalid argument', exc_info=True)
        else:
            logger.debug('Error calling NVML', exc_info=True)
    else:
        logger.debug('Exception calling NVML', exc_info=True)
