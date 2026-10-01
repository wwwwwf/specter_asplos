"""Run independent baselines and Specter sequentially with matched workloads."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from baselines.protocol import METHODS, ROOT, settings, validate_workload, workload_arguments


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    workload_arguments(parser)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--memory', choices=['high', 'low', 'both'], default='high')
    parser.add_argument('--memory-policy', choices=['matched', 'same-cache', 'fixed'], default='matched',
        help='SpecMoEOff only: match Specter GPU footprint, keep equal resident counts, or use an explicit fixed count.')
    parser.add_argument('--specmoeoff-resident-per-layer', type=int,
        help='Fixed SpecMoEOff residency for one memory setting; requires --memory-policy fixed.')
    parser.add_argument('--mixtral-resident-per-layer', type=int,
        help='Fixed DeepSeek Mixtral residency for one tier, or the high tier with --memory both; high defaults to 32.')
    parser.add_argument('--mixtral-low-resident-per-layer', type=int,
        help='Explicit low-tier DeepSeek Mixtral residency; required for --memory both when Mixtral is selected.')
    parser.add_argument('--reference-root', type=Path,
        help='Reuse completed comparison-root/<high|low>/specter runs when Specter is omitted.')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def mixtral_residencies(args):
    """Resolve only explicit Mixtral capacities and the requested high preset."""
    explicit = args.mixtral_resident_per_layer
    low = args.mixtral_low_resident_per_layer
    if 'mixtral-offloading' not in args.methods:
        if explicit is not None or low is not None:
            raise ValueError('Mixtral residency options require mixtral-offloading in --methods')
        return {}
    if args.model != 'dsv2lite':
        raise ValueError('The original Mixtral-Offloading adapter currently supports only dsv2lite; omit it from --methods for Qwen/Phi')
    if any(value is not None and not 1 <= value <= 64 for value in (explicit, low)):
        raise ValueError('DeepSeek Mixtral residency must be between 1 and 64 experts per layer')
    if args.memory == 'high':
        if low is not None:
            raise ValueError('--mixtral-low-resident-per-layer requires --memory low or both')
        return {'high': 32 if explicit is None else explicit}
    if args.memory == 'low':
        if explicit is not None and low is not None and explicit != low:
            raise ValueError('Conflicting explicit low-tier Mixtral resident counts')
        selected = low if low is not None else explicit
        if selected is None:
            raise ValueError('Low-tier Mixtral requires an explicit --mixtral-resident-per-layer; no low capacity is inferred')
        return {'low': selected}
    if low is None:
        raise ValueError('--memory both with Mixtral requires --mixtral-low-resident-per-layer; no low capacity is inferred')
    return {'high': 32 if explicit is None else explicit, 'low': low}


def specmoeoff_residency(args):
    explicit = args.specmoeoff_resident_per_layer
    if explicit is not None and ('specmoeoff' not in args.methods or args.memory_policy != 'fixed'):
        raise ValueError('--specmoeoff-resident-per-layer requires specmoeoff and --memory-policy fixed')
    if args.memory_policy != 'fixed' or 'specmoeoff' not in args.methods:
        return None
    if args.memory == 'both':
        raise ValueError('Fixed SpecMoEOff residency requires a single --memory high or low setting')
    case = settings(args.model, args.memory, 'specmoeoff')
    experts = case['resident_per_layer'] + case['offload_per_layer']
    if explicit is None or not 1 <= explicit <= experts:
        raise ValueError(f'Fixed SpecMoEOff requires --specmoeoff-resident-per-layer between 1 and {experts}')
    return explicit


def build_commands(args, config_file, output):
    validate_workload(args)
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('Methods must be unique')
    mixtral = mixtral_residencies(args)
    specmoeoff = specmoeoff_residency(args)
    if (args.memory_policy == 'matched' and 'specmoeoff' in args.methods
            and 'specter' not in args.methods and args.reference_root is None):
        raise ValueError('Matched SpecMoEOff memory needs Specter in --methods or an existing --reference-root')
    methods = sorted(args.methods, key=lambda method: method != 'specter')
    commands = []
    for memory in (['high', 'low'] if args.memory == 'both' else [args.memory]):
        for method in methods:
            case = settings(args.model, memory, method)
            label = f'{memory}/{method}'
            command = [sys.executable, '-B', '-m',
                'benchmarks.run_tpot' if method == 'specter' else 'baselines.run_tpot',
                '--config', str(config_file), '--model', args.model,
                '--datasets', *args.datasets, '--num-data', str(args.num_data),
                '--repeats', str(args.repeats), '--tokens', str(args.tokens),
                '--prefix-tokens', str(args.prefix_tokens), '--output', str(Path(output) / label)]
            if method == 'specter':
                command += ['--strategy', 'greedy', '--gamma', str(case['gamma']),
                            '--offload-per-layer', str(case['offload_per_layer'])]
                if args.memory_policy in ('matched', 'fixed') or mixtral:
                    command += ['--record-process-memory']
            else:
                command += ['--method', method, '--memory', memory]
                if method == 'mixtral-offloading':
                    command += ['--resident-per-layer', str(mixtral[memory]),
                                '--buffer-size', '4', '--record-process-memory']
                elif args.memory_policy == 'fixed':
                    command += ['--resident-per-layer', str(specmoeoff),
                                '--buffer-size', str(case['buffer_size']), '--record-process-memory']
                elif args.memory_policy == 'matched':
                    reference = (args.reference_root or Path(output)) / memory / 'specter'
                    command += ['--memory-reference', str(reference)]
            if args.check_numa:
                command += ['--check-numa']
            if args.require_greedy_match:
                command += ['--require-greedy-match']
            commands.append((label, command))
    return commands


def validate_references(args, config_file):
    """Reject incompatible reused measurements before loading any models."""
    if args.reference_root is None or 'specter' in args.methods:
        return
    from baselines.compare import read_run
    resource_configuration = Path(config_file).read_text(encoding='utf-8')
    tiers = ['high', 'low'] if args.memory == 'both' else [args.memory]
    for tier in tiers:
        directory = args.reference_root / tier / 'specter'
        config, _, records = read_run(directory)
        for field in ('model', 'datasets', 'num_data', 'repeats', 'tokens', 'prefix_tokens'):
            if config['args'][field] != getattr(args, field):
                raise ValueError(f'Reference {tier} workload mismatch: {field}')
        if config.get('resource_configuration_toml') != resource_configuration:
            raise ValueError(f'Reference {tier} uses a different resource configuration')
        expected = settings(args.model, tier, 'specter')
        for field in ('gamma', 'offload_per_layer', 'buffer_size'):
            if config['model_case'].get(field) != expected[field]:
                raise ValueError(f'Reference {tier} configuration mismatch: {field}')
        if any(record.get('kind') != 'specter' for record in records.values()):
            raise ValueError(f'Reference {tier} must contain Specter measurements')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from configuration import configure
    config = configure(args.config)
    output = config.output_path(args.output, 'baseline_comparison')
    try:
        commands = build_commands(args, config.file, output)
    except ValueError as error:
        parser.error(str(error))
    print(f'Output root: {output}', flush=True)
    for label, command in commands:
        print(f'[{label}] ' + (subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)), flush=True)
    if args.dry_run:
        return
    if output.exists():
        raise FileExistsError(f'Use a fresh output path: {output}')
    validate_references(args, config.file)
    # Fail on missing datasets before starting any model-loading subprocess.
    from data.loader import prepare_datasets
    prepare_datasets(config, args.datasets, args.num_data)
    if args.check_numa:
        from benchmarks.numa import check_numa
        check_numa(config.values.get('runtime', {}))
    output.mkdir(parents=True)
    (output / 'plan.json').write_text(json.dumps({
        'args': vars(args), 'commands': commands,
        'comparison': {method: 'fixed-residency' if method == 'mixtral-offloading' or args.memory_policy == 'fixed' else args.memory_policy
                       for method in args.methods if method != 'specter'},
        'mixtral_residencies': mixtral_residencies(args),
        'memory_accounting': 'Fixed residency with per-method GPU measurements; the optional matched policy applies to SpecMoEOff.',
    }, indent=2, default=str), encoding='utf-8')
    for label, command in commands:
        print(f'[start] {label}', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    if ('specter' in args.methods and len(args.methods) > 1) or args.reference_root:
        from baselines.compare import compare_root
        compare_root(output, args.reference_root)
    print(f'[done] {output}', flush=True)


if __name__ == '__main__':
    main()
