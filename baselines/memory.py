"""Calibrate baseline residency against a completed Specter GPU footprint."""
import gc
import json
from pathlib import Path

METRIC = 'peak CUDA reserved bytes plus observed native process overhead'


def read_reference(path, args):
    from baselines.compare import read_run
    from baselines.protocol import settings
    config, inputs, records = read_run(path)
    for name in ('model', 'datasets', 'num_data', 'repeats', 'tokens', 'prefix_tokens'):
        if config['args'][name] != getattr(args, name):
            raise ValueError(f'Memory reference workload mismatch: {name}')
    expected = settings(args.model, args.memory, 'specter')
    for name in ('offload_per_layer', 'gamma', 'buffer_size'):
        if config['model_case'].get(name) != expected[name]:
            raise ValueError(f'Memory reference {args.memory} tier mismatch: {name}')
    values = []
    for record in records.values():
        if record['kind'] != 'specter' or 'gpu_memory' not in record:
            raise ValueError('Rerun Specter with --record-process-memory to obtain a measured reference')
        values.append(record['gpu_memory']['runtime_gpu_bytes'])
    return {'reference': str(Path(path).resolve()), 'metric': METRIC,
            'budget_bytes': max(values), 'config': config, 'inputs': inputs}


def calibrate(backend, items, tokens, budget, output, measure, method, max_resident):
    """Bracket the largest fitting capacity; byte estimates only choose probes.

    Every selected capacity is checked again if restoring it after a failed
    larger probe. Only measured budget overshoots are recoverable here; CUDA
    OOM or backend/telemetry failures propagate without reusing partial state.
    """
    import torch
    from baselines.run_tpot import append
    metadata = backend.metadata
    slot_step = metadata['expert_slot_bytes'] * metadata['cache_group_count']
    if not isinstance(slot_step, int) or isinstance(slot_step, bool) or slot_step <= 0:
        raise ValueError('Backend must report positive per-layer residency storage')
    if not isinstance(max_resident, int) or isinstance(max_resident, bool) or max_resident < 1:
        raise ValueError('Maximum resident capacity must be a positive integer')
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1 or not items:
        raise ValueError('A positive integer memory budget and nonempty calibration inputs are required')
    low, high, resident = 0, max_resident, 1
    observations = []
    while len(observations) <= 2 * max_resident:
        backend.resize_residency(resident)
        gc.collect()
        torch.cuda.empty_cache()
        peak = 0
        print(f'[memory-calibration] {method} resident={resident} budget={budget / 2**30:.3f} GiB', flush=True)
        for global_index, (dataset, index, _, ids) in enumerate(items):
            result = measure(backend, ids, tokens, 42 + global_index * 1009, method, record_process_memory=True)
            result.update(dataset=dataset, prompt_index=index, resident_per_layer=resident)
            append(output / 'memory_calibration.jsonl', result)
            peak = max(peak, result['gpu_memory']['runtime_gpu_bytes'])
            print(f'[memory-calibration] {global_index + 1}/{len(items)} {dataset} p{index} '
                  f'resident={resident} peak={peak / 2**30:.3f} GiB', flush=True)
        observations.append({'resident_per_layer': resident, 'runtime_gpu_bytes': peak,
                             'within_budget': peak <= budget})
        print(f'[memory-calibration] resident={resident} peak={peak / 2**30:.3f} GiB '
              f'fits={peak <= budget}', flush=True)
        if peak <= budget:
            low = max(low, resident)
            if low == high:
                break
            additional = (budget - peak) // slot_step
            candidate = min(high, resident + max(1, additional))
        else:
            high = min(high, resident - 1)
            if low > high:
                low = 0
            if high < 1:
                report = {'budget_bytes': budget, 'metric': METRIC, 'observations': observations,
                          'feasible': False, 'reason': 'Minimum residency exceeds the measured Specter budget'}
                (output / 'memory_budget.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                raise RuntimeError(f'{method} cannot fit the measured budget even with one resident expert per layer; '
                                   'gamma, transfer buffers, and weight sharing were not altered')
            if high == low:
                # Restore and recheck: allocator fragmentation can differ.
                candidate = low
            else:
                decrease = max(1, (peak - budget + slot_step - 1) // slot_step)
                candidate = min(high, resident - decrease)
                if candidate <= low:
                    # The byte estimate can overshoot when workspace grows
                    # nonlinearly. Keep searching the untested open bracket.
                    candidate = (low + high + 1) // 2
        resident = candidate
    else:
        raise RuntimeError('Memory calibration did not converge; inspect memory_calibration.jsonl')
    return {'budget_bytes': budget, 'metric': METRIC, 'resident_per_layer': resident,
            'feasible': True, 'observations': observations,
            'validation': 'Full configured inputs, output length, and greedy decoding; final measurements are checked again.'}
