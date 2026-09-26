"""Local W4A16 routed projection versus sequential expert projections.

All paths use the same seeded inputs, routes and represented weight values.
This measures one projection, not a full FFN or the historical backend sweep.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import random
import re
import statistics


@dataclass(frozen=True)
class KernelCase:
    tokens: int = 16
    experts: int = 8
    hidden: int = 256
    features: int = 256
    top_k: int = 2
    seed: int = 42
    warmup: int = 10
    iterations: int = 50
    trials: int = 5
    device: str = 'cuda:0'

    def validate(self):
        for name in ('tokens', 'experts', 'hidden', 'features', 'top_k',
                     'warmup', 'iterations', 'trials'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.top_k > self.experts:
            raise ValueError('top_k must not exceed experts')
        if self.hidden % 128 or self.features % 128:
            raise ValueError('hidden and features must be multiples of 128')
        if not isinstance(self.device, str) or re.fullmatch(r'cuda(?::\d+)?', self.device) is None:
            raise ValueError('the local kernel benchmark requires a CUDA device')
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        return self


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='External TOML configuration or SPECTER_CONFIG.')
    parser.add_argument('--output', help='Output directory under paths.output or an absolute external directory.')
    parser.add_argument('--tokens', nargs='+', type=int, default=[1, 16, 128])
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-numa', action='store_true',
                        help='Require the configured CPU and preferred-memory NUMA placement.')
    return parser


def _prepare(case):
    """Prepare shared weights and fixed routing outside the timing boundary."""
    import torch
    from initialization.marlin_pack import repack, marlin_moe_permute_scales
    from streamlined_execution_engine.kernels import ops
    from streamlined_execution_engine.kernels.scalar_type import scalar_types

    generator = torch.Generator(device='cpu').manual_seed(case.seed)
    e, k, n = case.experts, case.hidden, case.features
    q = torch.randint(1, 16, (e, k, n), generator=generator, dtype=torch.int64)
    scales = (torch.rand(e, k // 128, n, generator=generator) * 0.03125 + 0.015625).half()
    shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 1, 8, 1)
    packed = (q.reshape(e, k // 8, 8, n) << shifts).sum(dim=2).to(torch.int32).to(case.device)
    perm = torch.empty(e, 0, dtype=torch.int32, device=case.device)
    packed = repack(packed, perm, e, k, n, 4)
    packed_scales = marlin_moe_permute_scales(scales.to(case.device), k, n, 128)
    reference_weights = ((q - 8).half() * scales.repeat_interleave(128, dim=1)).to(case.device)
    x = (torch.randn(case.tokens, k, generator=generator) * 0.25).half().to(case.device)
    routes_cpu = torch.rand(case.tokens, e, generator=generator).topk(case.top_k, dim=-1).indices
    routes = routes_cpu.to(case.device)
    alignment = ops.moe_align_block_size(routes, 8, e)
    weights = torch.ones(case.tokens, case.top_k, dtype=torch.float32, device=case.device)
    workspace_size = torch.cuda.get_device_properties(case.device).multi_processor_count * 4
    workspace = torch.zeros(workspace_size, dtype=torch.int32, device=case.device)
    output = torch.empty(case.tokens * case.top_k, n, dtype=torch.float16, device=case.device)
    sequential_output = torch.empty_like(output)
    reference_output = torch.empty_like(output)

    def gemm(inputs, out, quantized, scale, aligned, size_m, top_k):
        return ops.moe_wna16_marlin_gemm(
            inputs, out, quantized, scale, None, None, None, None,
            workspace, *aligned, weights,
            moe_block_size=8, top_k=top_k, mul_topk_weights=False,
            is_ep=False, b_q_type=scalar_types.uint4b8,
            size_m=size_m, size_n=n, size_k=k, is_k_full=True,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

    # CPU-derived row lists avoid a synchronizing nonzero() inside timed calls.
    groups = []
    counts = []
    flat_routes = routes_cpu.flatten()
    for expert in range(e):
        rows = (flat_routes == expert).nonzero().flatten()
        counts.append(rows.numel())
        if rows.numel() == 0:
            continue
        rows = rows.to(case.device)
        input_rows = torch.div(rows, case.top_k, rounding_mode='floor')
        local_routes = torch.zeros(rows.numel(), 1, dtype=torch.int64, device=case.device)
        local_alignment = ops.moe_align_block_size(local_routes, 8, 1)
        local_out = torch.empty(rows.numel(), n, dtype=torch.float16, device=case.device)
        groups.append((expert, rows, input_rows, local_alignment, local_out))

    def fused():
        return gemm(x, output, packed, packed_scales, alignment, case.tokens, case.top_k)

    def sequential():
        for expert, rows, input_rows, local_alignment, local_out in groups:
            inputs = x.index_select(0, input_rows)
            result = gemm(inputs, local_out, packed[expert:expert + 1],
                          packed_scales[expert:expert + 1], local_alignment,
                          rows.numel(), 1)
            sequential_output.index_copy_(0, rows, result)
        return sequential_output

    def reference():
        for expert, rows, input_rows, _, local_out in groups:
            inputs = x.index_select(0, input_rows)
            torch.mm(inputs, reference_weights[expert], out=local_out)
            reference_output.index_copy_(0, rows, local_out)
        return reference_output

    return {
        'paths': {'local_w4a16_batched': fused,
                  'local_w4a16_sequential': sequential,
                  'dequantized_fp16_sequential': reference},
        'routing': {'expert_counts': counts, 'active_experts': len(groups),
                    'top_k_ids': routes_cpu.tolist()},
        'weights': {'group_size': 128, 'bits': 4, 'zero_point': 8,
                    'quantized_storage_bytes': packed.numel() * packed.element_size(),
                    'scale_storage_bytes': packed_scales.numel() * packed_scales.element_size(),
                    'reference_storage_bytes': reference_weights.numel() * reference_weights.element_size()},
    }


def _compare(actual, expected):
    import torch
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError('projection produced a nonfinite value')
    if float(expected.abs().max()) <= 0.01:
        raise AssertionError('reference output is unexpectedly zero')
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    difference = actual.float() - expected.float()
    relative_l2 = float(difference.norm() / expected.float().norm())
    if relative_l2 >= 0.015:
        raise AssertionError(f'relative L2 error exceeds tolerance: {relative_l2}')
    return {'max_abs_error': float(difference.abs().max()),
            'relative_l2_error': relative_l2, 'rtol': 0.02, 'atol': 0.002,
            'relative_l2_limit': 0.015, 'passed': True}


def run_case(case):
    """Validate outputs, then time the three current local projection paths."""
    case.validate()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use --dry-run to inspect the workload')
    with torch.cuda.device(case.device), torch.inference_mode():
        prepared = _prepare(case)
        paths = prepared['paths']
        # Each path returns reusable scratch: preserve outputs before the next.
        outputs = {name: call().clone() for name, call in paths.items()}
        torch.cuda.synchronize(case.device)
        reference = outputs['dequantized_fp16_sequential']
        checks = {name: _compare(value, reference) for name, value in outputs.items()
                  if name != 'dequantized_fp16_sequential'}
        checks['batched_vs_sequential'] = _compare(
            outputs['local_w4a16_batched'], outputs['local_w4a16_sequential'])
        del outputs, reference
        for call in paths.values():
            for _ in range(case.warmup):
                call()
        torch.cuda.synchronize(case.device)
        samples = {name: [] for name in paths}
        orders = []
        generator = random.Random(case.seed)
        for _ in range(case.trials):
            order = list(paths)
            generator.shuffle(order)
            orders.append(order)
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(case.iterations):
                    paths[name]()
                end.record()
                end.synchronize()
                samples[name].append(float(start.elapsed_time(end)) / case.iterations)
        timings = {name: {'samples_ms': values, 'median_ms': statistics.median(values),
                          'min_ms': min(values), 'max_ms': max(values)}
                   for name, values in samples.items()}
        baseline = timings['local_w4a16_batched']['median_ms']
        return {
            'case': asdict(case), 'routing': prepared['routing'],
            'weights': prepared['weights'], 'correctness': checks,
            'timings': timings, 'trial_order': orders,
            'sequential_over_batched': timings['local_w4a16_sequential']['median_ms'] / baseline,
            'device': {'name': torch.cuda.get_device_name(case.device),
                       'capability': list(torch.cuda.get_device_capability(case.device)),
                       'torch': torch.__version__, 'cuda': torch.version.cuda},
        }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cases = [KernelCase(tokens=tokens, experts=args.experts, hidden=args.hidden,
                            features=args.features, top_k=args.top_k, seed=args.seed,
                            warmup=args.warmup, iterations=args.iterations,
                            trials=args.trials, device=args.device).validate()
                 for tokens in args.tokens]
        if len(set(args.tokens)) != len(args.tokens):
            raise ValueError('tokens must not contain duplicate sizes')
    except ValueError as error:
        parser.error(str(error))
    from configuration import configure
    config = configure(args.config)
    output = config.output_path(args.output, 'kernel')
    plan = {
        'experiment': 'local_routed_projection',
        'scope': 'one routed projection; no FFN activation, routing softmax, weighted reduction, or historical backend claim',
        'timing': 'CUDA events around repeated Python calls, including device idle between launches; includes sequential input gather and output scatter; excludes fixed routing alignment, weight packing, input creation and validation',
        'paths': ['local_w4a16_batched', 'local_w4a16_sequential', 'dequantized_fp16_sequential'],
        'dtype': 'float16', 'moe_block_size': 8,
        'check_numa': args.check_numa,
        'cases': [asdict(case) for case in cases], 'output': str(output),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return plan
    if output.exists():
        raise FileExistsError(f'Use a fresh output path: {output}')
    from benchmarks.numa import check_numa, probe_numa
    if args.check_numa:
        check_numa(config.values.get('runtime', {}))
    plan['numa_policy'] = probe_numa()
    # Reserve a fresh directory before importing CUDA or compiling kernels.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    results = []
    try:
        for case in cases:
            print(f'Projection tokens={case.tokens}, experts={case.experts}, shape={case.hidden}x{case.features}', flush=True)
            results.append(run_case(case))
            (output / 'results.json').write_text(json.dumps(
                {'plan': plan, 'complete': False, 'results': results}, indent=2), encoding='utf-8')
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps(
            {'type': type(error).__name__, 'message': str(error),
             'completed_cases': len(results)}, indent=2), encoding='utf-8')
        raise
    report = {'plan': plan, 'complete': True, 'results': results}
    (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
