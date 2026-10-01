"""Run optional measurements against the current Specter implementation."""
import argparse
import contextlib
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time

from benchmarks.experiment_plan import DATASETS, EXPERIMENTS, MODELS, build_cases, select_inputs, seed_for


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--model', choices=MODELS, default='dsv2lite')
    parser.add_argument('--config')
    parser.add_argument('--output', help='External output directory, or a name under paths.output.')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--prompts-json', help='External JSON array of prompt strings.')
    parser.add_argument('--num-data', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--prefix-tokens', type=int, default=16)
    parser.add_argument('--prefix-lengths', nargs='+', type=int, default=[8, 16, 32])
    parser.add_argument('--allow-short-prefix', action='store_true')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--depths', nargs='+', type=int)
    parser.add_argument('--resident', type=int)
    parser.add_argument('--residents', nargs='+', type=int)
    parser.add_argument('--policies', nargs='+', choices=['none', 'early', 'late', 'adaptive'],
                        default=['none', 'early', 'late', 'adaptive'])
    parser.add_argument('--error-rates', nargs='+', type=float, default=[0, 0.25, 0.5, 0.75])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-numa', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def append_json(path, data):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(data, allow_nan=False) + '\n')


def load_candidate_prompts(config, args, cases):
    """Validate candidate pools before importing models or creating results."""
    from configuration import external_path
    from data.loader import prepare_data

    raw_prompts = {}
    for dataset in dict.fromkeys(case.dataset for case in cases):
        if dataset == 'CUSTOM':
            path = external_path(args.prompts_json)
            context = f"Dataset 'CUSTOM' at {path}"
            try:
                prompts = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
                    raise ValueError('prompts-json must contain an array of strings')
                if not any(prompt.strip() for prompt in prompts):
                    raise ValueError('Need at least 1 nonempty prompt; found 0')
            except FileNotFoundError as error:
                raise FileNotFoundError(f'{context}: {error}') from error
            except (OSError, ValueError) as error:
                raise ValueError(f'{context}: {error}') from error
        else:
            path = config.path('datasets', dataset)
            prompts = prepare_data(path,
                                   max(args.num_data * 8, 1024),
                                   dataset_name=dataset, allow_fewer=True)
        available = sum(bool(prompt.strip()) for prompt in prompts)
        if available < args.num_data:
            raise ValueError(f"Dataset {dataset!r} at {path}: Need {args.num_data} "
                             f'nonempty prompts; found {available}')
        raw_prompts[dataset] = prompts
    return raw_prompts


def summarize(records, expected, output):
    rows = []
    for label in dict.fromkeys(record['case']['label'] for record in records):
        values = [record for record in records if record['case']['label'] == label]
        row = {**values[0]['case'], 'samples': len(values)}
        for field in ('tpot_ms', 'ttft_ms', 'e2e_ms_per_token', 'peak_allocated_bytes',
                      'target_recall', 'draft_distinct_mean', 'target_distinct_mean',
                      'cache_hit_rate', 'prefetch_accuracy', 'fidelity_recall', 'exact_set_match_rate'):
            numbers = [record[field] for record in values if record.get(field) is not None]
            if numbers:
                row[field] = statistics.mean(numbers)
                if field == 'tpot_ms':
                    row['tpot_stdev_ms'] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        rows.append(row)
    payload = {'expected_records': expected, 'completed_records': len(records),
               'complete': len(records) == expected, 'rows': rows}
    write_json(output / 'summary.json', payload)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return payload


def trace_metrics(trace):
    if any(window['status'] != 'complete' for window in trace['windows']):
        raise ValueError('Cannot summarize aborted diagnostic windows')
    layers = [layer for window in trace['windows'] for layer in window['layers']]
    recalls, draft_sizes, target_sizes = [], [], []
    for layer in layers:
        draft = {value for row in layer['draft_ids'] for value in row}
        target = {value for row in layer['target_ids'] for value in row}
        if target:
            recalls.append(len(draft & target) / len(target))
        draft_sizes.append(len(draft))
        target_sizes.append(len(target))
    window_hits = []
    for window in trace['windows']:
        requests = sum(layer['io'].get('demand_requests', 0) for layer in window['layers'])
        hits = sum(layer['io'].get('demand_hits', 0) for layer in window['layers'])
        if requests:
            window_hits.append(hits / requests)
    return {'target_recall': statistics.mean(recalls) if recalls else None,
            'draft_distinct_mean': statistics.mean(draft_sizes) if draft_sizes else None,
            'target_distinct_mean': statistics.mean(target_sizes) if target_sizes else None,
            'cache_hit_rate': statistics.mean(window_hits) if window_hits else None,
            'cache_hit_rate_scope': 'Mean per-verification-window demand hit rate; prefill and tail excluded; in-flight residents count as hits'}


def run_decode(model_case, setting, tokenizer, draft, target, reset, ids, tokens, seed,
               diagnostic=None, oracle_plans=None):
    import torch
    from initialization.model_loader import install_hooks
    from benchmarks.run_tpot import controller_for
    from benchmarks.token_timing import TokenCommitClock, observe_spec_inf
    from benchmarks.cache_policies import PrefetchPolicy
    from benchmarks.variants import ScopedVariant
    from speculative_inference_controller.state import spec_inf

    model_case = replace(model_case, gamma=setting.depth)
    controller = controller_for(model_case, target, ids.device)
    manager = controller.expert_manager
    reset.restore()
    manager.reset_async_loading_stats()
    hooks = install_hooks(model_case, draft, target, controller)
    clock = TokenCommitClock(ids.shape[1])
    decode = observe_spec_inf(spec_inf, clock)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trace = latency = memory = oracle = None
    log = io.StringIO()
    try:
        with contextlib.ExitStack() as stack:
            variant = stack.enter_context(ScopedVariant(manager, kv_mode=setting.kv_mode,
                                                       expert_mode=setting.expert_mode))
            stack.enter_context(PrefetchPolicy(controller, policy=setting.policy))
            state_factory = variant.state_factory
            if oracle_plans is not None:
                from benchmarks.oracle import OraclePrefetch
                oracle = stack.enter_context(OraclePrefetch(controller, oracle_plans))
            if diagnostic == 'routing':
                from benchmarks.routing import RouteCapture
                trace = stack.enter_context(RouteCapture(model_case, draft, target, controller))
            elif diagnostic == 'latency':
                from benchmarks.latency import LatencyTrace
                latency = stack.enter_context(LatencyTrace(draft, target, controller, manager))
            elif diagnostic == 'memory':
                from benchmarks.memory import MemorySnapshots
                memory = MemorySnapshots(draft, target)
                memory.capture('loaded')
                from speculative_inference_controller.state import TargetConsistentDecodingStateManager
                base_state = state_factory or TargetConsistentDecodingStateManager

                class MemoryState:
                    def __init__(self, *args, **kwargs):
                        self.inner = base_state(*args, **kwargs)
                        self.captured_commit = False

                    def prefill(self, input_ids):
                        self.inner.prefill(input_ids)
                        memory.capture('prefill', caches={'draft': self.inner.draft._past_key_values,
                                                         'target': self.inner.target._past_key_values},
                                       metadata={'prefix_tokens': input_ids.shape[1]})

                    def commit(self, length):
                        self.inner.commit(length)
                        if not self.captured_commit:
                            memory.capture('first_commit', caches={'draft': self.inner.draft._past_key_values,
                                                                  'target': self.inner.target._past_key_values},
                                           metadata={'committed_length': length})
                            self.captured_commit = True

                state_factory = MemoryState
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            clock.begin()
            with torch.inference_mode(), contextlib.redirect_stdout(log):
                result = decode(draft, target, ids.clone(), tokens, setting.depth, tokenizer,
                                prefetch_controller=controller, sampling_strategy=setting.strategy,
                                model_name=model_case.name,
                                sampling_profile='custom' if setting.strategy == 'sampling' else 'untruncated',
                                sampling_config={'temperature': setting.temperature, 'top_k': 0, 'top_p': 1.0}
                                if setting.strategy == 'sampling' else None,
                                state_factory=state_factory)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - started) * 1000
            timing = clock.result(tokens)
            token_ids = result[0].tolist()
            peak = torch.cuda.max_memory_allocated()
            io_stats = manager.get_async_loading_stats()
            policy_stats = controller.get_prefetch_policy_stats()
            variant_metadata = variant.metadata
    finally:
        for hook in hooks:
            hook.remove()
    record = {'role': 'diagnostic' if diagnostic else 'timed', 'diagnostic': diagnostic,
              'token_ids': token_ids, 'generated_tokens': len(token_ids) - ids.shape[1],
              'output_sha256': hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
              'prefix_tokens': ids.shape[1], 'seed': seed, 'wall_ms': wall_ms,
              'e2e_ms_per_token': wall_ms / tokens, 'peak_allocated_bytes': peak,
              'io': io_stats, 'pio': policy_stats, 'variant': variant_metadata,
              'log': log.getvalue(), **timing}
    requests = io_stats.get('demand_requests', 0)
    record['cache_hit_rate'] = io_stats.get('demand_hits', 0) / requests if requests else None
    record['cache_hit_rate_scope'] = 'All target demand requests, including prefill and tail'
    if trace is not None:
        record['routing'] = trace.result()
        record.update(trace_metrics(record['routing']))
    if latency is not None:
        record['latency'] = latency.result()
    if memory is not None:
        record['memory'] = memory.snapshots
    if oracle is not None:
        record['oracle'] = oracle.result()
        if record['oracle']['visited_windows'] != record['oracle']['planned_windows']:
            raise RuntimeError('Oracle replay ended before all recorded windows were visited')
        if not record['oracle']['used_windows']:
            raise RuntimeError('Oracle replay submitted no predictive plans; use depth >= 2 '
                               'with an enabled prefetch policy')
        record['prefetch_accuracy'] = record['oracle']['prefetch_accuracy']
    return record


def run_fidelity(model_case, tokenizer, draft, target, reset, ids, seed):
    import torch
    from transformers import DynamicCache
    from benchmarks.routing import ForwardRouteCapture, matched_route_fidelity
    torch.manual_seed(seed)
    reset.restore()
    captures = {}
    log = io.StringIO()
    for role, model in (('draft', draft), ('target', target)):
        with ForwardRouteCapture(model_case, model, role=role) as capture, \
                torch.inference_mode(), contextlib.redirect_stdout(log):
            result = model(ids, past_key_values=DynamicCache(), use_cache=True)
        del result
        torch.cuda.synchronize()
        captures[role] = capture.result()
    if any(len(captures[role]['calls']) != 1 for role in ('draft', 'target')):
        raise RuntimeError('Expected one complete forward per model for matched-input fidelity')
    routes = {role: {layer['layer_id']: layer['ids'] for layer in captures[role]['calls'][0]['layers']}
              for role in ('draft', 'target')}
    fidelity = matched_route_fidelity(routes['draft'], routes['target'])
    return {'role': 'fidelity', 'seed': seed, 'prefix_tokens': ids.shape[1],
            'log': log.getvalue(),
            'captures': captures, 'fidelity': fidelity,
            'fidelity_recall': fidelity['mean_target_recall'],
            'exact_set_match_rate': fidelity['exact_set_match_rate'],
            'cache_context': 'Independent empty DynamicCache for each model; identical full input IDs'}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cases = build_cases(args)
    except ValueError as error:
        parser.error(str(error))
    from configuration import configure
    config = configure(args.config)
    output = config.output_path(args.output, f'experiments/{args.experiment}/{args.model}')
    plan = {'experiment': args.experiment, 'model': args.model,
            'cases': [case.to_dict() for case in cases],
            'expected_records': len(cases) * args.num_data * args.repeats,
            'output': str(output), 'trace_timing': 'Diagnostics run separately from timed decode',
            'input_selection': 'First qualifying inputs shared across cases; source indices are recorded',
            'source_index_scope': 'Index in the loader-filtered prompt list, or in the supplied JSON array'}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists():
        raise FileExistsError(f'Use a fresh output path: {output}')
    raw_prompts = load_candidate_prompts(config, args, cases)
    from benchmarks.numa import check_numa, probe_numa
    if args.check_numa:
        check_numa(config.values.get('runtime', {}))
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    import torch
    from initialization.model_loader import model_cases
    from speculative_inference_controller.model_init import HybridPrecisionModelInitializer
    from benchmarks.cache_reset import ResidentReset
    from benchmarks.run_tpot import controller_for
    torch.cuda.set_device(0)
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('Select an A100 using CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(config.values.get('runtime', {}).get('threads', 8))
    output.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    metadata = {**plan, 'args': vars(args), 'hardware': torch.cuda.get_device_name(),
                'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'configuration_file': str(config.file),
                'configuration_toml': config.file.read_text(encoding='utf-8'),
                'numa': probe_numa(), 'torch_version': torch.__version__,
                'source_sha256': {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source_root.rglob('*.py') if '__pycache__' not in path.parts}}
    write_json(output / 'config.json', metadata)
    records, inputs = [], {}
    base_case = model_cases(config)[args.model]
    print(f'Output: {output}; planned measurements: {plan["expected_records"]}', flush=True)
    for resident in dict.fromkeys(case.resident for case in cases):
        model_case = replace(base_case, offload_per_layer=base_case.num_experts - resident)
        tokenizer, draft, target, sharing = HybridPrecisionModelInitializer().load(model_case, torch.device('cuda:0'))
        manager = controller_for(model_case, target, torch.device('cuda:0')).expert_manager
        reset = ResidentReset(manager)
        write_json(output / f'sharing_resident{resident}.json', sharing)
        if not inputs:
            for dataset, prompts in raw_prompts.items():
                length = max(case.prefix_tokens for case in cases if case.dataset == dataset)
                inputs[dataset] = select_inputs(prompts, tokenizer, args.num_data, length, args.allow_short_prefix)
            write_json(output / 'inputs.json', inputs)
        group = [case for case in cases if case.resident == resident]
        first = group[0]
        warm_ids = torch.tensor([inputs[first.dataset][0]['input_ids'][:first.prefix_tokens]], device='cuda:0')
        warm_setting = replace(first, policy='adaptive', kv_mode='shared', expert_mode='overlap', error_rate=None)
        warmup = run_decode(model_case, warm_setting, tokenizer, draft, target, reset, warm_ids,
                            min(args.tokens, 64), args.seed)
        append_json(output / 'warmup.jsonl', {'case': warm_setting.to_dict(), **warmup})
        oracle_cache = {}
        for repeat in range(args.repeats):
            for sample_index in range(args.num_data):
                # Rotate policies/depths to reduce systematic run-order effects.
                offset = (repeat + sample_index) % len(group)
                order = group[offset:] + group[:offset]
                for setting in order:
                    sample = inputs[setting.dataset][sample_index]
                    ids = torch.tensor([sample['input_ids'][:setting.prefix_tokens]], device='cuda:0')
                    seed = seed_for(args.seed, setting.dataset, sample['source_index'], repeat)
                    common = {'case': setting.to_dict(), 'sample_index': sample_index,
                              'source_index': sample['source_index'], 'repeat': repeat,
                              'input_ids': ids[0].tolist()}
                    plans, oracle_reference = None, None
                    if args.experiment == 'oracle':
                        from benchmarks.oracle import build_oracle_plans
                        key = (setting.dataset, sample_index, repeat)
                        if key not in oracle_cache:
                            oracle_reference = run_decode(model_case, replace(setting, error_rate=None), tokenizer,
                                                          draft, target, reset, ids, args.tokens, seed, 'routing')
                            oracle_cache[key] = oracle_reference
                            append_json(output / 'oracle_sources.jsonl', {**common, **oracle_reference})
                        oracle_reference = oracle_cache[key]
                        plans = build_oracle_plans(oracle_reference['routing']['windows'], model_case.num_experts,
                                                  resident, seed, setting.error_rate,
                                                  expected_layers=range(int(model_case.skip_first_layer),
                                                                        model_case.layer_num + int(model_case.skip_first_layer)))
                    diagnostic_kind = ('latency' if args.experiment == 'latency' else
                                       'memory' if args.experiment == 'memory' else 'routing')
                    if args.experiment == 'fidelity':
                        record = run_fidelity(model_case, tokenizer, draft, target, reset, ids, seed)
                    else:
                        diagnostic = run_decode(model_case, setting, tokenizer, draft, target, reset, ids,
                                                args.tokens, seed, diagnostic_kind, plans)
                        append_json(output / 'diagnostics.jsonl', {**common, **diagnostic})
                        record = run_decode(model_case, setting, tokenizer, draft, target, reset, ids,
                                            args.tokens, seed, oracle_plans=plans)
                        if diagnostic['token_ids'] != record['token_ids']:
                            append_json(output / 'failed_measurements.jsonl', {**common, **record})
                            raise RuntimeError('Diagnostic and timed outputs differ; refusing to pair their metrics')
                        for field in ('target_recall', 'draft_distinct_mean', 'target_distinct_mean',
                                      'cache_hit_rate', 'cache_hit_rate_scope'):
                            if field in diagnostic:
                                record[field] = diagnostic[field]
                        if oracle_reference is not None:
                            if record['token_ids'] != oracle_reference['token_ids']:
                                append_json(output / 'failed_measurements.jsonl', {**common, **record})
                                raise RuntimeError('Oracle replay changed the decoding trajectory')
                            expected_routes = [[(layer['layer_id'], layer['target_ids']) for layer in window['layers']]
                                               for window in oracle_reference['routing']['windows']]
                            actual_routes = [[(layer['layer_id'], layer['target_ids']) for layer in window['layers']]
                                             for window in diagnostic['routing']['windows']]
                            if actual_routes != expected_routes:
                                raise RuntimeError('Target routing differs from the recorded oracle trajectory')
                            record['oracle_output_matches'] = True
                    record.update(common)
                    records.append(record)
                    append_json(output / 'records.jsonl', record)
                    summarize(records, plan['expected_records'], output)
                    value = f'TPOT={record["tpot_ms"]:.3f}' if 'tpot_ms' in record else 'fidelity recorded'
                    print(f'[{len(records)}/{plan["expected_records"]}] {setting.label} p{sample_index} r{repeat} {value}', flush=True)
        del draft, target, manager, reset, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    summary = summarize(records, plan['expected_records'], output)
    if not summary['complete']:
        raise RuntimeError('Experiment record count is incomplete')
    print(f'Complete: {output}', flush=True)


if __name__ == '__main__':
    main()
