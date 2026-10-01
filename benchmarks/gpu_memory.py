"""Observe this process's CUDA footprint outside the timed inference region.

Callers synchronize CUDA before taking a snapshot. NVML-backed nvidia-smi
snapshots include native allocations that PyTorch's allocator cannot see;
they do not establish a hard physical-memory cap or capture every transient.
"""
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import io
import os
import subprocess
import uuid


def _uuid_key(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        if len(value) == 16:
            value = str(uuid.UUID(bytes=value))
        else:
            value = value.decode('ascii')
    value = str(value).strip().lower()
    if not value or value in ('n/a', 'none'):
        return None
    return value.removeprefix('gpu-')


def _nonnegative_bytes(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer byte count')
    return value


def memory_snapshot(torch_module=None, device=None):
    """Return live allocator and own-PID process bytes on the selected GPU.

Telemetry errors raise: a missing process row must never be interpreted as
zero native overhead. If PyTorch cannot expose a device UUID, exactly one
row for this PID is required to identify the physical GPU unambiguously.
"""
    if torch_module is None:
        import torch as torch_module
    cuda = torch_module.cuda
    if device is None:
        device = cuda.current_device()
    selected_uuid = _uuid_key(getattr(cuda.get_device_properties(device), 'uuid', None))
    allocated = int(cuda.memory_allocated(device))
    reserved = int(cuda.memory_reserved(device))
    pid = os.getpid()
    command = ['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_uuid',
               '--format=csv,noheader,nounits']
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('Process GPU-memory telemetry is required: nvidia-smi failed') from exc

    rows = []
    try:
        for row in csv.reader(io.StringIO(output)):
            if not row or not any(field.strip() for field in row):
                continue
            if len(row) != 3:
                raise ValueError('Expected pid,used_memory,gpu_uuid columns')
            if int(row[0].strip()) == pid:
                rows.append(tuple(field.strip() for field in row))
    except (ValueError, csv.Error) as exc:
        raise RuntimeError('Malformed nvidia-smi process GPU-memory telemetry') from exc

    if selected_uuid is not None:
        rows = [row for row in rows if _uuid_key(row[2]) == selected_uuid]
    if len(rows) != 1:
        detail = 'selected GPU' if selected_uuid is not None else 'unknown GPU UUID'
        raise RuntimeError(f'Expected one GPU-memory row for PID {pid} ({detail}); found {len(rows)}')
    _, used_mib, gpu_uuid = rows[0]
    if _uuid_key(gpu_uuid) is None:
        raise RuntimeError('GPU-memory telemetry did not identify a GPU UUID')
    try:
        memory_mib = Decimal(used_mib)
        if not memory_mib.is_finite() or memory_mib < 0:
            raise ValueError('Invalid process memory')
        process = int((memory_mib * (2 ** 20)).to_integral_value(rounding=ROUND_CEILING))
    except (ValueError, InvalidOperation) as exc:
        raise RuntimeError('Process GPU-memory usage is unavailable or invalid') from exc
    _nonnegative_bytes(allocated, 'allocated_bytes')
    _nonnegative_bytes(reserved, 'reserved_bytes')
    return {'allocated_bytes': allocated, 'reserved_bytes': reserved,
            'process_bytes': process, 'native_overhead_bytes': max(0, process - reserved),
            'gpu_uuid': gpu_uuid, 'pid': pid}


def memory_footprint(before, after, peak_reserved_bytes, peak_allocated_bytes):
    """Combine allocator peaks with observed native overhead without double counting."""
    if before['pid'] != after['pid'] or _uuid_key(before['gpu_uuid']) != _uuid_key(after['gpu_uuid']):
        raise ValueError('Memory snapshots must refer to the same process and GPU')
    if _uuid_key(before['gpu_uuid']) is None:
        raise ValueError('Memory snapshots must identify a GPU UUID')
    for name, snapshot in (('before', before), ('after', after)):
        for field in ('allocated_bytes', 'reserved_bytes', 'process_bytes', 'native_overhead_bytes'):
            _nonnegative_bytes(snapshot[field], f'{name}.{field}')
        if snapshot['native_overhead_bytes'] != max(0, snapshot['process_bytes'] - snapshot['reserved_bytes']):
            raise ValueError(f'{name}.native_overhead_bytes is inconsistent with its snapshot')
    peak_reserved = _nonnegative_bytes(peak_reserved_bytes, 'peak_reserved_bytes')
    peak_allocated = _nonnegative_bytes(peak_allocated_bytes, 'peak_allocated_bytes')
    native = max(before['native_overhead_bytes'], after['native_overhead_bytes'])
    return {'runtime_gpu_bytes': max(peak_reserved + native, before['process_bytes'], after['process_bytes']),
            'peak_reserved_bytes': peak_reserved, 'peak_allocated_bytes': peak_allocated,
            'native_overhead_bytes': native, 'before': dict(before), 'after': dict(after)}
