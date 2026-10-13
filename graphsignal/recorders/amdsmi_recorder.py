import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

# The ROCm analog of NVMLRecorder: AMD GPU telemetry read through AMD SMI,
# whose Python bindings ship with ROCm (/opt/rocm/share/amd_smi, preinstalled
# in the rocm/* images; `pip install amdsmi` otherwise). A missing module
# disables this recorder only. Metrics keep the NVML names wherever the
# reading means the same thing, so a consumer ranks GPUs the same way on
# either vendor; a reading AMD SMI reports as unavailable ("N/A") is simply
# not emitted (null = not measured).
try:
    import amdsmi
    _AMDSMI_IMPORT_ERROR = None
except Exception as _exc:
    amdsmi = None
    _AMDSMI_IMPORT_ERROR = _exc


class AmdSmiRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._is_initialized: bool = False
        self._gpu_first_seen_ts: Optional[int] = None
        # (device index, processor handle, static tags, static attributes)
        self._devices: List[Tuple[int, Any, Dict[str, str], Dict[str, str]]] = []
        self._visible_device_idxs: List[int] = []
        self._current_device_idxs: List[int] = []

    def setup(self):
        if amdsmi is None:
            logger.debug('amdsmi not importable (%s), skipping AMD GPU usage',
                         _AMDSMI_IMPORT_ERROR)
            return
        try:
            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
        except Exception:
            logger.debug('Error initializing AMD SMI, skipping AMD GPU usage',
                         exc_info=True)
            return
        self._is_initialized = True
        logger.debug('Initialized AMD SMI')
        self._gpu_first_seen_ts = time.time_ns()

        device_count = len(handles)
        if device_count == 0:
            return

        self._visible_device_idxs = _visible_device_idxs(device_count)
        self._current_device_idxs = _current_device_idxs(self._visible_device_idxs)

        for idx in self._current_device_idxs:
            if idx >= device_count:
                logger.debug('Visible AMD device index %d out of range (%d devices)',
                             idx, device_count)
                continue
            handle = handles[idx]
            self._devices.append((idx, handle, _device_tags(handle),
                                  _device_attributes(handle)))

    def shutdown(self):
        if not self._is_initialized:
            return
        self._devices = []
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            logger.debug('Error shutting down AMD SMI', exc_info=True)
        self._is_initialized = False

    def on_tick(self):
        if not self._is_initialized:
            return

        now_ns = time.time_ns()
        watcher = graphsignal.watcher.watcher()

        for idx, handle, device_tags, attrs in self._devices:
            def gauge(name, value):
                if value is not None and value > 0:
                    watcher.set_gauge(name=name, value=value,
                                      measurement_ts=now_ns, tags=device_tags)

            def counter(name, total):
                if total is not None and total > 0:
                    watcher.set_counter(name=name, total=total,
                                        measurement_ts=now_ns, tags=device_tags)

            # amdsmi_get_gpu_memory_total / amdsmi_get_gpu_memory_usage (bytes)
            mem_total = _call(amdsmi.amdsmi_get_gpu_memory_total, handle,
                              amdsmi.AmdSmiMemoryType.VRAM)
            mem_used = _call(amdsmi.amdsmi_get_gpu_memory_usage, handle,
                             amdsmi.AmdSmiMemoryType.VRAM)
            gauge('gpu_memory_total_bytes', _num(mem_total))
            gauge('gpu_memory_used_bytes', _num(mem_used))
            if _num(mem_total) is not None and _num(mem_used) is not None:
                gauge('gpu_memory_free_bytes', mem_total - mem_used)

            # amdsmi_get_gpu_activity: gfx busy and memory-controller busy percent
            activity = _call(amdsmi.amdsmi_get_gpu_activity, handle) or {}
            gauge('gpu_utilization_percent', _num(activity.get('gfx_activity')))
            gauge('gpu_memory_utilization_percent', _num(activity.get('umc_activity')))

            # amdsmi_get_temp_metric: edge is the die temperature (NVML's GPU
            # temperature); a virtual function reports only hotspot, so fall
            # back to it. VRAM temperature has no NVML counterpart.
            temp = _temperature(handle, 'EDGE')
            if temp is None:
                temp = _temperature(handle, 'HOTSPOT')
            gauge('gpu_temperature_celsius', temp)
            gauge('gpu_memory_temperature_celsius', _temperature(handle, 'VRAM'))

            # amdsmi_get_power_info: socket power in W, limit in microwatts
            power = _call(amdsmi.amdsmi_get_power_info, handle) or {}
            power_usage = _num(power.get('socket_power'))
            if power_usage is None:
                power_usage = _num(power.get('current_socket_power'))
            gauge('gpu_power_usage_watts', power_usage)
            power_limit = _num(power.get('power_limit'))
            if power_limit is not None:
                gauge('gpu_power_management_limit_watts', power_limit / 1e6)

            # amdsmi_get_clock_info (MHz)
            gfx_clock = _call(amdsmi.amdsmi_get_clock_info, handle,
                              amdsmi.AmdSmiClkType.GFX) or {}
            gauge('gpu_clock_sm_megahertz', _num(gfx_clock.get('clk')))
            gauge('gpu_max_clock_sm_megahertz', _num(gfx_clock.get('max_clk')))
            mem_clock = _call(amdsmi.amdsmi_get_clock_info, handle,
                              amdsmi.AmdSmiClkType.MEM) or {}
            gauge('gpu_clock_mem_megahertz', _num(mem_clock.get('clk')))

            # amdsmi_get_pcie_info: link capability and the replay counter
            pcie = _call(amdsmi.amdsmi_get_pcie_info, handle) or {}
            pcie_static = pcie.get('pcie_static') or {}
            pcie_metric = pcie.get('pcie_metric') or {}
            gauge('gpu_pcie_link_generation', _num(pcie_static.get('pcie_interface_version')))
            gauge('gpu_pcie_link_width', _num(pcie_metric.get('pcie_width')))
            counter('gpu_pcie_replay_counter', _num(pcie_metric.get('pcie_replay_count')))

            # amdsmi_get_gpu_total_ecc_count (cumulative)
            ecc = _call(amdsmi.amdsmi_get_gpu_total_ecc_count, handle) or {}
            counter('gpu_ecc_errors_corrected_aggregate', _num(ecc.get('correctable_count')))
            counter('gpu_ecc_errors_uncorrected_aggregate', _num(ecc.get('uncorrectable_count')))

            # amdsmi_get_gpu_metrics_info: the throttle bitmask (the
            # independent one where the firmware reports it)
            metrics = _call(amdsmi.amdsmi_get_gpu_metrics_info, handle) or {}
            throttle = _num(metrics.get('indep_throttle_status'))
            if throttle is None:
                throttle = _num(metrics.get('throttle_status'))
            gauge('gpu_clocks_throttle_reasons', throttle)

            resource_attrs = dict(attrs)
            if _num(mem_total) is not None and mem_total > 0:
                resource_attrs['mem_total'] = str(mem_total)
            watcher.update_resource(
                'device',
                tags=device_tags,
                attributes=resource_attrs,
                first_seen_ts=self._gpu_first_seen_ts)


def _visible_device_idxs(device_count: int) -> List[int]:
    # HIP honors ROCR_VISIBLE_DEVICES (runtime level) and HIP_VISIBLE_DEVICES,
    # and accepts CUDA_VISIBLE_DEVICES as an alias; the first one set wins.
    for env_var in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        value = os.environ.get(env_var)
        if value:
            try:
                return [int(x) for x in value.split(',')]
            except ValueError:
                # UUID form: not resolvable to an index here.
                return list(range(device_count))
    return list(range(device_count))


def _current_device_idxs(visible_device_idxs: List[int]) -> List[int]:
    local_rank = None
    for env_var in ('LOCAL_RANK', 'SLURM_LOCALID', 'NCCL_LOCAL_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if env_var in os.environ:
            try:
                local_rank = int(os.environ[env_var])
            except ValueError:
                pass
            break
    if local_rank is not None:
        if local_rank < len(visible_device_idxs):
            return [visible_device_idxs[local_rank]]
        logger.debug('Local rank %s is out of visible device idxs %s',
                     local_rank, visible_device_idxs)
    return list(visible_device_idxs)


def _device_tags(handle) -> Dict[str, str]:
    tags = {}
    bdf = _call(amdsmi.amdsmi_get_gpu_device_bdf, handle)
    if isinstance(bdf, str) and bdf:
        tags['device.bus_id'] = bdf
    device_uuid = _call(amdsmi.amdsmi_get_gpu_device_uuid, handle)
    if isinstance(device_uuid, str) and device_uuid:
        tags['device.uuid'] = device_uuid
    return tags


def _device_attributes(handle) -> Dict[str, str]:
    attrs = {}
    asic = _call(amdsmi.amdsmi_get_gpu_asic_info, handle) or {}
    name = asic.get('market_name')
    if isinstance(name, str) and name and name != 'N/A':
        attrs['device.name'] = name
    arch = asic.get('target_graphics_version')
    if isinstance(arch, str) and arch and arch != 'N/A':
        attrs['architecture'] = arch
    return attrs


def _temperature(handle, sensor: str) -> Optional[float]:
    try:
        sensor_type = getattr(amdsmi.AmdSmiTemperatureType, sensor)
    except AttributeError:
        return None
    return _num(_call(amdsmi.amdsmi_get_temp_metric, handle, sensor_type,
                      amdsmi.AmdSmiTemperatureMetric.CURRENT))


def _call(fn, *args):
    # Every AMD SMI query may raise AmdSmiLibraryException — NOT_SUPPORTED is
    # routine (a virtual function exposes a subset of sensors) — and never
    # costs more than that one reading.
    try:
        return fn(*args)
    except Exception:
        logger.debug('AMD SMI call %s failed', getattr(fn, '__name__', fn), exc_info=True)
        return None


def _num(value) -> Optional[float]:
    # AMD SMI reports an unavailable reading as the string "N/A".
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value
