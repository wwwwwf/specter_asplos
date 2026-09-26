"""Generate tokens using speculative inference on one A100."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

# Set device ordering before importing Torch/GPTQModel.
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
from configuration import configure_from_cli
configure_from_cli()
import torch
from initialization.model_loader import model_cases, install_hooks
from speculative_inference_controller.model_init import HybridPrecisionModelInitializer
from speculative_inference_controller.state import spec_inf
from predictive_io_orchestrator.prefetch_planner import LayerAwarePrefetchPlanner
from benchmarks.token_timing import TokenCommitClock, observe_spec_inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dsv2lite', 'qwen2moe', 'phimoe'], default='dsv2lite')
    parser.add_argument('--config', help='External resource configuration (or SPECTER_CONFIG).')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--gamma', type=int, help='Draft depth; default DS/Qwen/Phi = 16/8/16.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategy', choices=['greedy', 'sampling'], default='greedy')
    parser.add_argument('--offload-per-layer', type=int)
    parser.add_argument('--output', help='Output name under paths.output, or an absolute external path.')
    args = parser.parse_args()
    from configuration import configure
    config = configure(args.config)
    args.output = str(config.output_path(args.output, 'generation.json'))
    if args.max_new_tokens < 1 or (args.gamma is not None and args.gamma < 1):
        parser.error('max-new-tokens and gamma must be positive')
    torch.cuda.set_device(0)
    if 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Select an A100 using CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(config.values.get('runtime', {}).get('threads', 8))
    torch.manual_seed(args.seed)
    case = model_cases(config)[args.model]
    args.gamma = case.gamma if args.gamma is None else args.gamma
    offload = args.offload_per_layer if args.offload_per_layer is not None else {'dsv2lite':48, 'qwen2moe':45, 'phimoe':14}[args.model]
    if not 0 <= offload < case.num_experts:
        parser.error('offload-per-layer must be in [0, num_experts)')
    case = replace(case, gamma=args.gamma, offload_per_layer=offload)
    tokenizer, draft, target, sharing = HybridPrecisionModelInitializer().load(case, torch.device('cuda:0'))
    manager = target.model.layers[1].block_sparse_moe.experts if args.model == 'phimoe' else target.model.layers[1].mlp.experts
    controller = LayerAwarePrefetchPlanner(case.layer_num, case.gamma, case.num_experts-offload,
        case.num_experts, case.router_topk, manager, case.skip_first_layer, device='cuda:0')
    hooks = install_hooks(case, draft, target, controller)
    ids = tokenizer(args.prompt, return_tensors='pt').input_ids.cuda()
    clock = TokenCommitClock(ids.shape[1])
    decode = observe_spec_inf(spec_inf, clock)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    clock.begin()
    try:
        with torch.inference_mode():
            result = decode(draft, target, ids, args.max_new_tokens, args.gamma, tokenizer,
                prefetch_controller=controller, sampling_strategy=args.strategy, model_name=args.model)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        for hook in hooks:
            hook.remove()
    record = {'config': vars(args), 'hardware': torch.cuda.get_device_name(), 'token_ids': result[0].tolist(),
        'generated_text': tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True),
        'elapsed_s': elapsed, 'e2e_ms_per_token': elapsed * 1000 / args.max_new_tokens,
        **clock.result(args.max_new_tokens),
        'tokens_per_s': args.max_new_tokens / elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'sharing': sharing, 'pio': controller.get_prefetch_policy_stats(), 'io': manager.get_async_loading_stats()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    tpot = 'N/A (one token)' if record['tpot_ms'] is None else f'{record["tpot_ms"]:.3f} ms/token'
    print(f'Output: {path}; decode TPOT={tpot}; E2E={record["e2e_ms_per_token"]:.3f} ms/token')

if __name__ == '__main__':
    main()
