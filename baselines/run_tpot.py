"""Measure an independent baseline using the current committed-token clock."""
import argparse
import contextlib
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

from baselines.protocol import DATASETS, ROOT, TPOT_DEFINITION, settings, validate_workload, workload_arguments


def append(path, record):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')


def measure(backend, ids, tokens, seed, method, target_only=False, record_process_memory=False):
    import torch
    from benchmarks.token_timing import TokenCommitClock
    backend.reset()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if record_process_memory:
        from benchmarks.gpu_memory import memory_snapshot, memory_footprint
        memory_before = memory_snapshot()
    clock = TokenCommitClock(ids.shape[1])
    captured = io.StringIO()
    started = time.perf_counter()
    clock.begin()
    with torch.inference_mode(), contextlib.redirect_stdout(captured):
        result = backend.generate(ids.clone(), tokens, seed=seed, clock=clock, target_only=target_only)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    token_ids = result['token_ids']
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().reshape(-1).tolist()
    if len(token_ids) != ids.shape[1] + tokens or token_ids[:ids.shape[1]] != ids[0].tolist():
        raise ValueError('Baseline output must preserve the prefix and contain exactly the requested tokens')
    record = {
        'kind': 'target' if target_only else method, 'strategy': 'greedy', 'seed': seed,
        'token_ids': token_ids, 'generated_tokens': tokens,
        'text': backend.tokenizer.decode(token_ids[ids.shape[1]:], skip_special_tokens=True),
        'digest': hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
        'wall_ms': wall_ms, 'e2e_ms_per_token': wall_ms / tokens,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'stats': result.get('stats', {}), 'log': captured.getvalue(), **clock.result(tokens),
    }
    if record_process_memory:
        record['gpu_memory'] = memory_footprint(memory_before, memory_snapshot(),
            record['peak_reserved_bytes'], record['peak_allocated_bytes'])
    return record


def summarize(records, output, expected, datasets, method):
    rows = []
    for dataset in datasets:
        values = [record for record in records if record['dataset'] == dataset]
        if values:
            row = {'dataset': dataset, 'method': method, 'n': len(values)}
            for metric in ('tpot_ms', 'ttft_ms', 'e2e_ms_per_token', 'post_first_batch_ms_per_token', 'peak_allocated_bytes'):
                row[metric] = statistics.mean(value[metric] for value in values)
            rows.append(row)
    payload = {'completed_records': len(records), 'expected_records': expected,
               'complete': len(records) == expected and len(rows) == len(datasets), 'datasets': rows}
    if len(rows) == len(datasets):
        payload['macro_tpot_ms'] = statistics.mean(row['tpot_ms'] for row in rows)
    temporary = output / 'summary.tmp'
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(output / 'summary.json')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    workload_arguments(parser)
    parser.add_argument('--method', choices=['mixtral-offloading', 'specmoeoff'], required=True)
    parser.add_argument('--memory', choices=['high', 'low'], default='high')
    parser.add_argument('--resident-per-layer', type=int, help='Explicit custom residency; recorded as a custom run.')
    parser.add_argument('--buffer-size', type=int, help='Explicit transfer-buffer count.')
    parser.add_argument('--memory-reference', type=Path,
        help='Completed matching Specter run with process telemetry; calibrate residency to its measured budget.')
    parser.add_argument('--record-process-memory', action='store_true',
        help='Record process GPU memory without changing an explicitly selected residency.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_workload(args)
    except ValueError as error:
        parser.error(str(error))
    if args.method == 'mixtral-offloading':
        if args.model != 'dsv2lite':
            parser.error('The original Mixtral-Offloading runtime currently has a DeepSeek-V2-Lite adapter only')
        if args.memory_reference:
            parser.error('Original Mixtral-Offloading uses a fixed cache; select --resident-per-layer instead')
        if args.memory == 'low' and args.resident_per_layer is None:
            parser.error('Low-memory Mixtral-Offloading requires an explicit --resident-per-layer')
    from configuration import configure
    config = configure(args.config)
    output = config.output_path(args.output, f'baselines/{args.model}/{args.memory}/{args.method}')
    if output.exists():
        raise FileExistsError(f'Use a fresh output path: {output}')
    from data.loader import prepare_datasets
    prompts = prepare_datasets(config, args.datasets, args.num_data)
    reference = None
    if args.memory_reference:
        if args.resident_per_layer is not None:
            parser.error('--resident-per-layer cannot override automatic matched-memory calibration')
        from baselines.memory import read_reference
        reference = read_reference(args.memory_reference, args)
        if reference['config']['resource_configuration_toml'] != config.file.read_text(encoding='utf-8'):
            raise ValueError('Memory reference uses a different resource configuration')
    record_process_memory = bool(reference) or args.record_process_memory
    from benchmarks.numa import check_numa, probe_numa
    if args.check_numa:
        check_numa(config.values.get('runtime', {}))
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    import torch
    from initialization.model_loader import model_cases
    selected = settings(args.model, args.memory, args.method)
    case = model_cases(config)[args.model]
    resident = selected['resident_per_layer'] if args.resident_per_layer is None else args.resident_per_layer
    if reference:
        resident = 1
    buffer_size = selected['buffer_size'] if args.buffer_size is None else args.buffer_size
    minimum_buffers = 2 if args.method == 'mixtral-offloading' else 0
    if not 1 <= resident <= case.num_experts or buffer_size < minimum_buffers:
        parser.error(f'Valid resident capacity and at least {minimum_buffers} transfer buffers required')
    case = replace(case, offload_per_layer=case.num_experts - resident,
                   buffer_size=buffer_size, gamma=selected['gamma'])
    torch.cuda.set_device(0)
    torch.set_num_threads(config.values.get('runtime', {}).get('threads', 8))
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('Select an A100')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output.mkdir(parents=True)
    expected = len(args.datasets) * args.num_data * args.repeats
    metadata = {
        'args': vars(args), 'model_case': asdict(case), 'prefix_tokens': args.prefix_tokens,
        'resource_configuration_file': str(config.file),
        'resource_configuration_toml': config.file.read_text(encoding='utf-8'),
        'reference_case': selected, 'expected_records': expected,
        'sampling_profile': 'greedy argmax', 'tpot_definition': TPOT_DEFINITION,
        'cache_reset': 'initial resident set and LRU restored before every input outside timing',
        'hardware': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'runtime_environment': {
            'executable': sys.executable, 'python_version': sys.version, 'torch_cuda': torch.version.cuda,
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'numa_policy': probe_numa(),
            'packages': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'gptqmodel']},
        },
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'sources': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in ROOT.rglob('*.py') if '__pycache__' not in path.parts},
    }
    (output / 'config.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    module = importlib.import_module('baselines.' + args.method.replace('-', '_') + '.backend')
    print(f'[load] {args.method} {args.model} resident={resident} buffers={buffer_size} gamma={case.gamma}', flush=True)
    backend = module.load_backend(case, torch.device('cuda:0'))
    try:
        (output / 'backend.json').write_text(json.dumps(backend.metadata, indent=2), encoding='utf-8')
        items = []
        for dataset in args.datasets:
            for index, prompt in enumerate(prompts[dataset]):
                ids = backend.tokenizer.encode(prompt, return_tensors='pt')[:, :args.prefix_tokens].cuda()
                if ids.shape[1] == 0:
                    raise ValueError(f'Empty tokenized prompt: {dataset}:{index}')
                items.append((dataset, index, prompt, ids))
        recorded_inputs = [
            {'dataset': dataset, 'index': index, 'text': prompt, 'input_ids': ids[0].tolist()}
            for dataset, index, prompt, ids in items]
        (output / 'inputs.json').write_text(json.dumps(recorded_inputs, indent=2), encoding='utf-8')
        if reference:
            if recorded_inputs != reference['inputs']:
                raise ValueError('Memory reference input text/token IDs differ')
            for key in ('hardware', 'cuda_visible_devices'):
                if metadata[key] != reference['config'][key]:
                    raise ValueError(f'Memory reference environment differs: {key}')
            from baselines.memory import calibrate
            report = calibrate(backend, items, args.tokens, reference['budget_bytes'], output,
                               measure, args.method, case.num_experts)
            report['reference'] = reference['reference']
            (output / 'memory_budget.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            metadata['model_case']['offload_per_layer'] = case.num_experts - report['resident_per_layer']
            metadata['memory_budget'] = report
            (output / 'config.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
            (output / 'backend.json').write_text(json.dumps(backend.metadata, indent=2), encoding='utf-8')
            print(f'[memory-selected] resident={report["resident_per_layer"]} '
                  f'budget={report["budget_bytes"] / 2**30:.3f} GiB', flush=True)
        for target_only in ([False, True] if args.method == 'specmoeoff' else [False]):
            record = measure(backend, items[0][3], 64, 42, args.method, target_only, record_process_memory)
            append(output / 'warmup.jsonl', record)
            print(f'[warmup] {record["kind"]} TPOT={record["tpot_ms"]:.3f}', flush=True)
        checks = []
        if args.method == 'specmoeoff':
            for dataset, index, _, ids in items:
                if index:
                    continue
                results = []
                for target_only in (True, False):
                    record = measure(backend, ids, args.tokens, 42, args.method, target_only, record_process_memory)
                    record.update(dataset=dataset, prompt_index=index)
                    append(output / 'greedy_checks.jsonl', record)
                    results.append(record)
                equal = results[0]['token_ids'] == results[1]['token_ids']
                check = {'dataset': dataset, 'baseline_matches_target': equal}
                checks.append(check)
                print('[greedy-check]', check, flush=True)
            (output / 'greedy_summary.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
            if args.require_greedy_match and not all(check['baseline_matches_target'] for check in checks):
                raise RuntimeError('Greedy correctness gate failed; measurements not started')
        records = []
        for repeat in range(args.repeats):
            for global_index, (dataset, index, _, ids) in enumerate(items):
                seed = 42 + global_index * 1009 + repeat * 100003
                record = measure(backend, ids, args.tokens, seed, args.method, record_process_memory=record_process_memory)
                record.update(model=case.name, dataset=dataset, prompt_index=index,
                              repeat=repeat, prefix_tokens=ids.shape[1])
                append(output / 'records.jsonl', record)
                if reference and record['gpu_memory']['runtime_gpu_bytes'] > reference['budget_bytes']:
                    raise RuntimeError('A final measurement exceeded the matched GPU-memory budget; '
                                       'no speedup comparison is accepted. Inspect memory_calibration.jsonl and records.jsonl.')
                records.append(record)
                summarize(records, output, expected, args.datasets, args.method)
                print(f'[result] {len(records)}/{expected} {dataset} p{index} r{repeat} '
                      f'{args.method} TPOT={record["tpot_ms"]:.3f} TTFT={record["ttft_ms"]:.1f}', flush=True)
        print(f'[done] {output}', flush=True)
    finally:
        if hasattr(backend, 'close'):
            backend.close()


if __name__ == '__main__':
    main()
