"""Optional NUMA validation and best-effort metadata collection."""
import os
from pathlib import Path
import subprocess


def probe_numa():
    try:
        return subprocess.check_output(
            ['numactl', '--show'], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f'unavailable: {exc}'


def check_numa(placement):
    cpu_node = placement.get('cpu_node', 0)
    memory_node = placement.get('memory_node', 0)
    try:
        cpu_list = Path(f'/sys/devices/system/node/node{cpu_node}/cpulist').read_text().strip()
        node_cpus = set()
        for item in cpu_list.split(','):
            bounds = [int(value) for value in item.split('-')]
            node_cpus.update(range(bounds[0], bounds[-1] + 1))
        affinity = set(os.sched_getaffinity(0))
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f'Cannot validate NUMA CPU placement: {exc}') from exc
    if not affinity.issubset(node_cpus):
        raise RuntimeError(f'Bind CPUs with numactl --cpunodebind={cpu_node}')
    policy = dict(line.split(':', 1) for line in probe_numa().splitlines() if ':' in line)
    if policy.get('policy', '').strip() != 'preferred' or policy.get('preferred node', '').strip() != str(memory_node):
        raise RuntimeError(f'Use numactl --preferred={memory_node}; numactl must be available')
