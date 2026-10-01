"""Compare completed measurements only when their workloads and timing match."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from baselines.protocol import METHODS, TPOT_DEFINITION


def read_run(directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    inputs = json.loads((directory / 'inputs.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (directory / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    expected = config['expected_records']
    if len(records) != expected:
        raise ValueError(f'Incomplete run {directory}: {len(records)}/{expected}')
    args = config['args']
    expected_keys = {(dataset, index, repeat) for dataset in args['datasets']
                     for index in range(args['num_data']) for repeat in range(args['repeats'])}
    keyed = {(record['dataset'], record['prompt_index'], record['repeat']): record for record in records}
    if len(keyed) != len(records) or set(keyed) != expected_keys:
        raise ValueError(f'Duplicate or missing measurement keys: {directory}')
    if config.get('tpot_definition') != TPOT_DEFINITION:
        raise ValueError(f'TPOT definition mismatch: {directory}')
    for record in records:
        if (record['generated_tokens'] != args['tokens'] or record['strategy'] != 'greedy'
                or len(record['token_ids']) != record['prefix_tokens'] + args['tokens']):
            raise ValueError(f'Output length or strategy mismatch: {directory}')
        value = record['tpot_ms']
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid TPOT: {directory}')
    return config, inputs, keyed


def compare_pair(specter_directory, baseline_directory):
    specter_config, specter_inputs, specter = read_run(specter_directory)
    baseline_config, baseline_inputs, baseline = read_run(baseline_directory)
    if specter_inputs != baseline_inputs:
        raise ValueError('Recorded input text/token IDs differ; rerun with the same input selection')
    for field in ('model', 'tokens', 'datasets', 'num_data', 'repeats', 'prefix_tokens'):
        if specter_config['args'][field] != baseline_config['args'][field]:
            raise ValueError(f'Workload mismatch: {field}')
    for field in ('hardware', 'cuda_visible_devices', 'resource_configuration_toml'):
        if specter_config.get(field) != baseline_config.get(field):
            raise ValueError(f'Environment/configuration mismatch: {field}')
    for field in ('torch_cuda', 'cpu_affinity', 'numa_policy'):
        if specter_config['runtime_environment'].get(field) != baseline_config['runtime_environment'].get(field):
            raise ValueError(f'Runtime mismatch: {field}')
    specter_packages = specter_config['runtime_environment'].get('packages', {})
    baseline_packages = baseline_config['runtime_environment'].get('packages', {})
    for name in set(specter_packages) & set(baseline_packages):
        if specter_packages[name] != baseline_packages[name]:
            raise ValueError(f'Package version mismatch: {name}')
    if set(specter) != set(baseline):
        raise ValueError('Measurement key sets differ')
    for key in specter:
        if specter[key]['seed'] != baseline[key]['seed']:
            raise ValueError(f'Seed mismatch: {key}')
    if (specter_config['args'].get('require_greedy_match') or baseline_config['args'].get('require_greedy_match')):
        if any(specter[key]['token_ids'] != baseline[key]['token_ids'] for key in specter):
            raise ValueError('Cross-method greedy output agreement gate failed')
    method = baseline_config['args']['method']
    memory_budget = baseline_config.get('memory_budget')
    memory_result = None
    has_memory_telemetry = all('gpu_memory' in record for record in (*specter.values(), *baseline.values()))
    if memory_budget:
        if not all('gpu_memory' in record for record in (*specter.values(), *baseline.values())):
            raise ValueError('Matched-memory comparison requires process telemetry for both methods')
        specter_peak = max(record['gpu_memory']['runtime_gpu_bytes'] for record in specter.values())
        baseline_peak = max(record['gpu_memory']['runtime_gpu_bytes'] for record in baseline.values())
        if memory_budget['budget_bytes'] != specter_peak or baseline_peak > specter_peak:
            raise ValueError('Baseline GPU footprint exceeds the Specter budget, or the recorded budget differs')
        memory_result = {'metric': memory_budget['metric'], 'specter_budget_bytes': specter_peak,
                         'baseline_peak_bytes': baseline_peak, 'within_budget': True}
    rows = []
    for dataset in specter_config['args']['datasets']:
        keys = [key for key in specter if key[0] == dataset]
        specter_ms = statistics.mean(specter[key]['tpot_ms'] for key in keys)
        baseline_ms = statistics.mean(baseline[key]['tpot_ms'] for key in keys)
        rows.append({
            'dataset': dataset, 'baseline': method, 'records_per_method': len(keys),
            'specter_tpot_ms': specter_ms, 'baseline_tpot_ms': baseline_ms,
            'speedup_baseline_over_specter': baseline_ms / specter_ms,
            'identical_output_records': sum(specter[key]['token_ids'] == baseline[key]['token_ids'] for key in keys),
            'specter_peak_allocated_gib': max(specter[key]['peak_allocated_bytes'] for key in keys) / 2**30,
            'baseline_peak_allocated_gib': max(baseline[key]['peak_allocated_bytes'] for key in keys) / 2**30,
            'specter_runtime_gpu_gib': max(specter[key]['gpu_memory']['runtime_gpu_bytes'] for key in keys) / 2**30 if has_memory_telemetry else None,
            'baseline_runtime_gpu_gib': max(baseline[key]['gpu_memory']['runtime_gpu_bytes'] for key in keys) / 2**30 if has_memory_telemetry else None,
        })
    specter_ms = statistics.mean(row['specter_tpot_ms'] for row in rows)
    baseline_ms = statistics.mean(row['baseline_tpot_ms'] for row in rows)
    return {
        'baseline': method, 'datasets': rows,
        'macro_specter_tpot_ms': specter_ms, 'macro_baseline_tpot_ms': baseline_ms,
        'ratio_of_macro_tpot': baseline_ms / specter_ms,
        'mean_dataset_speedup': statistics.mean(row['speedup_baseline_over_specter'] for row in rows),
        'specter_resident_per_layer': specter_config['model_case']['num_experts'] - specter_config['model_case']['offload_per_layer'],
        'baseline_resident_per_layer': baseline_config['model_case']['num_experts'] - baseline_config['model_case']['offload_per_layer'],
        'memory_matching': memory_result,
        'memory_accounting': 'Matched runtime GPU footprint' if memory_result else 'Fixed residency; GPU footprint recorded per method',
    }


def compare_root(root, reference_root=None):
    root = Path(root)
    comparisons = []
    rows = []
    for memory in ('high', 'low'):
        specter = Path(reference_root or root) / memory / 'specter'
        if not specter.is_dir():
            continue
        for method in METHODS:
            if method == 'specter':
                continue
            baseline = root / memory / method
            if not baseline.is_dir():
                continue
            result = compare_pair(specter, baseline)
            result['memory'] = memory
            comparisons.append(result)
            rows.extend(dict(memory=memory, **row) for row in result['datasets'])
    if not comparisons:
        raise ValueError('No completed Specter/baseline pair found')
    payload = {'tpot_definition': TPOT_DEFINITION, 'comparisons': comparisons}
    (root / 'comparison.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with (root / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for result in comparisons:
        print(f'[comparison] {result["memory"]} {result["baseline"]}: '
              f'{result["macro_baseline_tpot_ms"]:.3f} / {result["macro_specter_tpot_ms"]:.3f} ms/token '
              f'= {result["ratio_of_macro_tpot"]:.3f}x', flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--reference-root', type=Path)
    args = parser.parse_args()
    from configuration import external_path
    compare_root(external_path(args.input), external_path(args.reference_root) if args.reference_root else None)


if __name__ == '__main__':
    main()
