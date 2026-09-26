"""Specter-only main workload with committed-token TPOT and target output checks."""
import argparse
import contextlib
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
from benchmarks.numa import check_numa, probe_numa

class CaptureTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = None
    def decode(self, ids, **kwargs):
        self.ids = ids.detach().cpu().tolist()
        return self.tokenizer.decode(ids, **kwargs)

def controller_for(case, target, device):
    from predictive_io_orchestrator.prefetch_planner import LayerAwarePrefetchPlanner
    manager = target.model.layers[1].block_sparse_moe.experts if case.name == 'phimoe' else target.model.layers[1].mlp.experts
    return LayerAwarePrefetchPlanner(case.layer_num, case.gamma, case.num_experts-case.offload_per_layer,
        case.num_experts, case.router_topk, manager, case.skip_first_layer, device=str(device))

DATASETS = ('GK', 'WT', 'HE', 'GP', 'C4')


def run_case(kind, case, tokenizer, draft, target, ids, tokens, seed, strategy, reset):
    import torch
    from initialization.model_loader import install_hooks
    from speculative_inference_controller.state import spec_inf
    from speculative_inference_controller.model_wrapper import ModelWrapper
    from benchmarks.token_timing import TokenCommitClock, observe_spec_inf
    controller = controller_for(case, target, ids.device)
    manager = controller.expert_manager
    manager.set_specter_async_demand_policy('resident')
    reset.restore()
    manager.reset_async_loading_stats()
    hooks = install_hooks(case, draft, target, controller) if kind != 'target' else []
    capture = CaptureTokenizer(tokenizer)
    clock = TokenCommitClock(ids.shape[1])
    function = None
    if kind != 'target':
        function = observe_spec_inf(spec_inf, clock)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    log = io.StringIO()
    start = time.perf_counter()
    clock.begin()
    try:
        with torch.inference_mode(), contextlib.redirect_stdout(log):
            if kind == 'target':
                wrapper = ModelWrapper(target)
                result = ids.clone()
                for _ in range(tokens):
                    probs = wrapper._forward_with_kvcache(result)
                    result = torch.cat((result, probs.argmax(-1, keepdim=True)), 1)
                    clock.commit(result.shape[1])
                capture.decode(result[0], skip_special_tokens=True)
                del wrapper
            else:
                function(draft, target, ids.clone(), tokens, case.gamma, capture,
                    prefetch_controller=controller, sampling_strategy=strategy, model_name=case.name)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000
    finally:
        for hook in hooks:
            hook.remove()
    record = {
        'kind': kind, 'strategy': strategy, 'seed': seed,
        'token_ids': capture.ids, 'generated_tokens': len(capture.ids) - ids.shape[1],
        'text': tokenizer.decode(capture.ids[ids.shape[1]:], skip_special_tokens=True),
        'digest': hashlib.sha256(json.dumps(capture.ids).encode()).hexdigest(),
        'wall_ms': wall_ms, 'e2e_ms_per_token': wall_ms / tokens,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'io': manager.get_async_loading_stats(), 'pio': controller.get_prefetch_policy_stats(),
        'log': log.getvalue(), **clock.result(tokens),
    }
    for label, pattern in [('accept_total', r'accept_total:(\d+)'), ('sd_rounds', r'step:(\d+)'),
                           ('acceptance_rate', r'acceptance_rate:([0-9.]+)'), ('window_utilization', r'window_utilization:([0-9.]+)')]:
        match = re.search(pattern, record['log'])
        if match:
            record[label] = float(match.group(1))
    del controller, clock, function
    return record


def append(path, record):
    with path.open('a') as f:
        f.write(json.dumps(record) + '\n')


def summarize(records, output, expected, datasets=DATASETS):
    rows = []
    for dataset in datasets:
        methods = {kind: [r for r in records if r['dataset'] == dataset and r['kind'] == kind] for kind in ['specter']}
        if not all(methods.values()):
            continue
        row = {'dataset': dataset}
        for kind, values in methods.items():
            row[kind + '_n'] = len(values)
            for metric in ['tpot_ms', 'ttft_ms', 'post_first_batch_ms_per_token', 'e2e_ms_per_token', 'first_commit_tokens']:
                row[kind + '_' + metric] = statistics.mean(v[metric] for v in values)
        rows.append(row)
    payload = {'completed_records': len(records), 'expected_records': expected,
        'complete': len(records) == expected and len(rows) == len(datasets), 'datasets': rows}
    if len(rows) == len(datasets):
        payload['macro_specter_tpot_ms'] = statistics.mean(r['specter_tpot_ms'] for r in rows)
    tmp = output / 'summary.tmp'
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(output / 'summary.json')
    if rows:
        with (output / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dsv2lite', 'qwen2moe', 'phimoe'], default='dsv2lite')
    parser.add_argument('--offload-per-layer', type=int)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--config', help='External resource configuration (or SPECTER_CONFIG).')
    parser.add_argument('--output', help='Output directory under paths.output, or an absolute external path.')
    parser.add_argument('--protocol-case', help='Validate arguments against the frozen AE main-figure protocol.')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--prefix-tokens', type=int, default=16, help='Maximum input prefix length; retain shorter inputs.')
    parser.add_argument('--check-numa', action='store_true', help='Require the CPU and preferred-memory placement configured in TOML.')
    parser.add_argument('--num-data', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--strategy', choices=['sampling', 'greedy'], default='greedy')
    parser.add_argument('--skip-greedy-check', action='store_true')
    parser.add_argument('--require-greedy-match', action='store_true',
        help='Stop before performance measurements if Specter differs from target.')
    return parser


def validate_arguments(parser, args):
    protocol = None
    protocol_case = None
    if args.protocol_case:
        protocol_path = ROOT / 'benchmarks/protocol.json'
        protocol = json.loads(protocol_path.read_text())
        if args.protocol_case not in protocol['cases']:
            parser.error('Unknown protocol case')
        protocol_case = protocol['cases'][args.protocol_case]
        if protocol_case['status'] != 'ready':
            parser.error('Protocol case is not frozen for execution')
        expected = {'model': protocol_case['model'], 'gamma': protocol_case['gamma'],
            'offload_per_layer': protocol_case['offload_per_layer'], 'strategy': protocol['strategy'],
            'tokens': protocol['output_tokens'], 'num_data': protocol['inputs_per_dataset'],
            'repeats': protocol['repeats'], 'skip_greedy_check': False,
            'datasets': protocol['datasets'], 'prefix_tokens': protocol['prefix_max_tokens']}
        for name, value in expected.items():
            if getattr(args, name) != value:
                parser.error(f'--{name.replace("_", "-")} must equal {value!r} for {args.protocol_case}')
    if args.require_greedy_match and args.skip_greedy_check:
        parser.error('--require-greedy-match requires greedy checks')
    if args.tokens < 33 or args.num_data < 1 or args.repeats < 1:
        parser.error('tokens >= 33 and positive samples/repeats required')
    if args.prefix_tokens < 1 or len(set(args.datasets)) != len(args.datasets):
        parser.error('positive prefix-tokens and unique datasets required')
    return protocol, protocol_case


def main():
    parser = build_parser()
    args = parser.parse_args()
    protocol, protocol_case = validate_arguments(parser, args)
    from configuration import configure
    resource_config = configure(args.config)
    args.output = str(resource_config.output_path(args.output, args.protocol_case or args.model))
    if args.check_numa:
        check_numa(resource_config.values.get('runtime', {}))
    import torch
    from initialization.model_loader import model_cases
    from speculative_inference_controller.model_init import HybridPrecisionModelInitializer
    from benchmarks.cache_reset import ResidentReset
    from data.loader import prepare_data
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.set_num_threads(resource_config.values.get('runtime', {}).get('threads', 8))
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('Select an A100')
    case = model_cases(resource_config)[args.model]
    # Match the inference cache budgets, for Specter.
    offload = args.offload_per_layer if args.offload_per_layer is not None else {'dsv2lite':48, 'qwen2moe':45, 'phimoe':14}[args.model]
    case = replace(case, offload_per_layer=offload, gamma=args.gamma or case.gamma)
    case = HybridPrecisionModelInitializer.resolve_model_paths(case)
    if not 0 <= offload < case.num_experts or case.gamma < 1:
        parser.error('Invalid cache budget or gamma')
    if protocol_case and (case.buffer_size != protocol_case['buffer_size'] or
                          case.num_experts - case.offload_per_layer != protocol_case['resident_per_layer']):
        parser.error('Resolved cache/buffer configuration differs from AE protocol')
    environment = {'executable': sys.executable, 'prefix': sys.prefix,
        'base_prefix': sys.base_prefix, 'python_version': sys.version,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'numa_policy': probe_numa(),
        'torch_cuda': torch.version.cuda, 'torch_origin': torch.__file__,
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch', 'transformers', 'triton', 'gptqmodel', 'flash_attn']}}
    config = {'args': vars(args), 'model_case': asdict(case), 'prefix_tokens': args.prefix_tokens,
        'resource_configuration_file': str(resource_config.file),
        'resource_configuration_toml': resource_config.file.read_text(encoding='utf-8'),
        'runtime_environment': environment,
        'hardware': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'sampling_profile': 'greedy argmax; current wrapper normalizes logits' if args.strategy == 'greedy' else 'untruncated T=1 top_k=0 top_p=1',
        'ae_protocol': protocol,
        'ae_case': protocol_case,
        'ae_protocol_sha256': hashlib.sha256((ROOT / 'benchmarks/protocol.json').read_bytes()).hexdigest() if protocol else None,
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'cache_budget_source': 'explicit AE protocol' if protocol_case else 'CLI override or inference defaults',
        'kernel_source_notice': (ROOT/'streamlined_execution_engine/kernels/NOTICE').read_text(),
        'tpot_definition': '(last committed token CUDA timestamp - first committed token CUDA timestamp) / (N-1)',
        'post_first_batch_definition': 'same duration / (N-first_commit_tokens)',
        'cache_reset': 'same initial resident set and LRU order before every case, outside timing',
        'expected_records': len(args.datasets) * args.num_data * args.repeats,
        'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ['__pycache__'])}}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print('[load]', asdict(case), flush=True)
    tokenizer, draft, target, sharing = HybridPrecisionModelInitializer().load(case, torch.device('cuda:0'))
    (output / 'sharing.json').write_text(json.dumps(sharing, indent=2))
    manager = target.model.layers[1].block_sparse_moe.experts if case.name == 'phimoe' else target.model.layers[1].mlp.experts
    reset = ResidentReset(manager)
    items = []
    for dataset in args.datasets:
        path = str(resource_config.path('datasets', dataset))
        # Same main-entry selection: first five nonempty prompts; preserve any
        # short prefix and record its length rather than silently resampling.
        prompts = prepare_data(path, args.num_data)
        if len(prompts) != args.num_data:
            raise ValueError(f'Insufficient prompts in {dataset}')
        for index, prompt in enumerate(prompts):
            ids = tokenizer.encode(prompt, return_tensors='pt')[:, :args.prefix_tokens].cuda()
            items.append((dataset, index, prompt, ids))
    (output / 'inputs.json').write_text(json.dumps([{'dataset': d, 'index': i, 'text': p, 'input_ids': ids[0].tolist()} for d, i, p, ids in items], indent=2))
    for kind in ['specter', 'target']:
        result = run_case(kind, case, tokenizer, draft, target, items[0][3], 64, 42, 'greedy', reset)
        append(output / 'warmup.jsonl', result)
        print('[warmup]', kind, result['tpot_ms'], flush=True)
    if not args.skip_greedy_check:
        agreements = []
        for dataset, index, prompt, ids in items:
            if index:
                continue
            group = {}
            for kind in ['target', 'specter']:
                result = run_case(kind, case, tokenizer, draft, target, ids, args.tokens, 42, 'greedy', reset)
                result.update(dataset=dataset, prompt_index=index)
                append(output / 'greedy_checks.jsonl', result)
                group[kind] = result
            agreement = {'dataset': dataset, 'specter_matches_target': group['specter']['token_ids'] == group['target']['token_ids']}
            agreements.append(agreement)
            print('[greedy-check]', agreement, flush=True)
        (output / 'greedy_summary.json').write_text(json.dumps(agreements, indent=2))
        if args.require_greedy_match and not all(
            a['specter_matches_target'] for a in agreements
        ):
            raise RuntimeError('Greedy correctness gate failed; performance measurements not started')
    records = []
    for repeat in range(args.repeats):
        for global_index, (dataset, index, prompt, ids) in enumerate(items):
            order = ['specter']
            seed = 42 + global_index * 1009 + repeat * 100003
            for kind in order:
                result = run_case(kind, case, tokenizer, draft, target, ids, args.tokens, seed, args.strategy, reset)
                result.update(model=case.name, dataset=dataset, prompt_index=index, repeat=repeat, prefix_tokens=ids.shape[1])
                append(output / 'records.jsonl', result)
                records.append(result)
                summarize(records, output, config['expected_records'], args.datasets)
                print(f'[result] {len(records)}/{config["expected_records"]} {dataset} p{index} r{repeat} {kind} TPOT={result["tpot_ms"]:.3f} TTFT={result["ttft_ms"]:.1f} first={result["first_commit_tokens"]} steady={result["post_first_batch_ms_per_token"]:.3f}', flush=True)
    print('[done]', output, flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        print('[failed]', traceback.format_exc(), flush=True)
        raise
