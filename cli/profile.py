"""Select draft depth using an offline TPOT sweep."""
import argparse
import contextlib
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dsv2lite', 'qwen2moe', 'phimoe'], default='dsv2lite')
    parser.add_argument('--config', help='External resource configuration (or SPECTER_CONFIG).')
    parser.add_argument('--prompts-json', required=True, help='JSON array of representative prompt strings.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--strategy', choices=['sampling', 'greedy'], default='greedy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix-tokens', type=int, help='Optional maximum input prefix length; retain shorter inputs.')
    parser.add_argument('--check-numa', action='store_true', help='Require the CPU and preferred-memory placement configured in TOML.')
    parser.add_argument('--output', help='Output name under paths.output, or an absolute external path.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    from configuration import configure
    config = configure(args.config)
    args.output = str(config.output_path(args.output, 'depth_profile.json'))
    prompts = json.loads(Path(args.prompts_json).read_text(encoding='utf-8'))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError('prompts-json must contain a nonempty array of nonempty strings')
    if args.tokens < 2 or args.repeats < 1:
        raise ValueError('tokens >= 2 and positive repeats required for decode TPOT')
    if args.prefix_tokens is not None and args.prefix_tokens < 1:
        raise ValueError('prefix-tokens must be positive')
    path = Path(args.output)
    raw_path = path.with_suffix('.jsonl')
    config_path = path.with_suffix('.config.json')
    for artifact in (path, raw_path, config_path):
        if artifact.exists():
            raise FileExistsError(f'Use a fresh output path: {artifact}')
    from benchmarks.numa import check_numa, probe_numa
    if args.check_numa:
        check_numa(config.values.get('runtime', {}))
    import torch
    from initialization.model_loader import model_cases, install_hooks
    from speculative_inference_controller.model_init import HybridPrecisionModelInitializer
    from speculative_inference_controller.state import spec_inf
    from speculative_inference_controller.depth_profiler import LightweightSpeculativeDepthProfiler
    from predictive_io_orchestrator.prefetch_planner import LayerAwarePrefetchPlanner
    from benchmarks.token_timing import TokenCommitClock, observe_spec_inf
    from benchmarks.cache_reset import ResidentReset
    torch.cuda.set_device(0)
    torch.set_num_threads(config.values.get('runtime', {}).get('threads', 8))
    torch.manual_seed(args.seed)
    case = model_cases(config)[args.model]
    case = replace(case, offload_per_layer={'dsv2lite':48, 'qwen2moe':45, 'phimoe':14}[args.model])
    path.parent.mkdir(parents=True, exist_ok=True)
    run_config = {
        'args': vars(args), 'model_case': asdict(case),
        'resource_configuration_file': str(config.file),
        'resource_configuration_toml': config.file.read_text(encoding='utf-8'),
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'hardware': torch.cuda.get_device_name(), 'numa_policy': probe_numa(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'prompts': prompts,
        'candidates': list(LightweightSpeculativeDepthProfiler.candidates),
        'expected_records': len(LightweightSpeculativeDepthProfiler.candidates) * len(prompts) * args.repeats,
        'sources': {str(source.relative_to(Path(__file__).resolve().parents[1])):
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    for source in Path(__file__).resolve().parents[1].rglob('*.py')
                    if '__pycache__' not in source.parts},
    }
    config_path.write_text(json.dumps(run_config, indent=2), encoding='utf-8')
    tokenizer, draft, target, sharing = HybridPrecisionModelInitializer().load(case, torch.device('cuda:0'))
    manager = target.model.layers[1].block_sparse_moe.experts if args.model == 'phimoe' else target.model.layers[1].mlp.experts
    reset = ResidentReset(manager)

    def evaluate(depth, prompt, repeat):
        ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        if args.prefix_tokens is not None:
            ids = ids[:, :args.prefix_tokens]
        controller = LayerAwarePrefetchPlanner(case.layer_num, depth, case.num_experts-case.offload_per_layer,
            case.num_experts, case.router_topk, manager, case.skip_first_layer, device='cuda:0')
        handles = install_hooks(case, draft, target, controller)
        manager.set_specter_async_demand_policy('resident')
        reset.restore()
        manager.reset_async_loading_stats()
        seed = args.seed + max(repeat, 0)
        torch.manual_seed(seed)
        clock = TokenCommitClock(ids.shape[1])
        decode = observe_spec_inf(spec_inf, clock)
        torch.cuda.synchronize()
        started = time.perf_counter()
        clock.begin()
        try:
            with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                result = decode(draft, target, ids, args.tokens, depth, tokenizer,
                    prefetch_controller=controller, sampling_strategy=args.strategy, model_name=args.model)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - started) * 1000
            timing = clock.result(args.tokens)
            tpot = timing['tpot_ms']
        finally:
            for handle in handles:
                handle.remove()
        record = {'depth': depth, 'prompt': prompt, 'repeat': repeat, 'seed': seed,
            'input_ids': ids[0].tolist(), 'prefix_tokens': ids.shape[1],
            'warmup': repeat < 0, 'wall_ms': wall_ms, 'e2e_ms_per_token': wall_ms / args.tokens,
            **timing, 'token_ids': result[0].tolist()}
        with raw_path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(f'K={depth} repeat={repeat} TPOT={tpot:.3f}', flush=True)
        return tpot

    # Warmup is separately marked and omitted from selection statistics.
    evaluate(2, prompts[0], -1)
    report = LightweightSpeculativeDepthProfiler().profile(evaluate, prompts, args.repeats)
    report.update(config=vars(args), run_configuration=run_config,
        completed_records=len(report['records']), expected_records=run_config['expected_records'],
        complete=len(report['records']) == run_config['expected_records'],
        sharing=sharing, hardware=torch.cuda.get_device_name(),
        tpot_definition='(last_commit_ms - first_commit_ms) / (N - 1)',
        cache_reset='initial resident set and LRU order before each evaluation, outside timing')
    path.write_text(json.dumps(report, indent=2))
    print(f'Selected K={report["selected_depth"]}; report={path}')

if __name__ == '__main__':
    main()
