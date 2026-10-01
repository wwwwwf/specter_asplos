"""One-command entry for the main workload and optional system experiments."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
from benchmarks.experiment_plan import EXPERIMENTS

MAIN_CASES = {'high': 'ds_high', 'low': 'ds_low4'}
DS_BASELINE_RESIDENCIES = {
    'high': {'mixtral-offloading': 32, 'specmoeoff': 12},
    'low': {'mixtral-offloading': 20, 'specmoeoff': 2},
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['main', 'depth', 'prefix', 'all', 'speedup', 'kernel', *EXPERIMENTS], default='main')
    parser.add_argument('--config', help='External TOML configuration (or SPECTER_CONFIG).')
    parser.add_argument('--output', help='Output name under paths.output, or an absolute external path; default: ae.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without importing Torch, loading data, or writing outputs.')
    parser.add_argument('--check-numa', action='store_true', help='Require the configured CPU and preferred-memory NUMA placement.')
    parser.add_argument('--memory', choices=['high', 'low', 'both'], default='high',
                        help='Memory setting for main, all, or speedup. Default: high.')
    parser.add_argument('--model', choices=['dsv2lite', 'qwen2moe', 'phimoe'], default='dsv2lite',
                        help='Model for speedup; other AE experiments keep their DeepSeek protocol.')
    parser.add_argument('--reference-root', type=Path,
                        help='For speedup: reuse a completed comparison root containing <high|low>/specter; run only baselines.')
    parser.add_argument('--require-greedy-match', action='store_true',
                        help='For speedup: require exact checked output agreement before accepting a comparison.')
    parser.add_argument('--num-data', type=int,
                        help='For speedup: first N nonempty inputs per dataset; default: 5, use 3 for the compact workload.')
    return parser


def build_commands(experiment, config_file, output, check_numa=False, memory='high',
                   model='dsv2lite', reference_root=None, require_greedy_match=False, num_data=None):
    """Keep the chosen workload explicit and inspectable before GPU execution."""
    output = Path(output)
    if memory not in (*MAIN_CASES, 'both'):
        raise ValueError(f'Unknown memory configuration: {memory}')
    if memory != 'high' and experiment not in ('main', 'all', 'speedup'):
        raise ValueError('--memory applies only to --experiment main or all or speedup')
    if model not in ('dsv2lite', 'qwen2moe', 'phimoe'):
        raise ValueError(f'Unknown model: {model}')
    if experiment != 'speedup' and (model != 'dsv2lite' or reference_root is not None or require_greedy_match or num_data is not None):
        raise ValueError('--model overrides, --reference-root, --require-greedy-match, and --num-data apply only to --experiment speedup')
    if num_data is not None and (not isinstance(num_data, int) or isinstance(num_data, bool) or num_data < 1):
        raise ValueError('--num-data must be a positive integer')
    commands = []
    python = [sys.executable, '-B', '-m']
    if experiment == 'speedup':
        protocol = json.loads((ROOT / 'benchmarks/protocol.json').read_text(encoding='utf-8'))
        tiers = ['high', 'low'] if model == 'dsv2lite' and memory == 'both' else [memory]
        if model == 'dsv2lite':
            methods = ['mixtral-offloading', 'specmoeoff']
        elif model == 'qwen2moe':
            if memory != 'high':
                raise ValueError('Qwen speedup currently provides the fixed high configuration only; use baselines.run for explicit low-memory experiments')
            methods = ['specmoeoff']
            memory_options = ['--memory-policy', 'fixed', '--specmoeoff-resident-per-layer', '6']
        else:
            methods = ['specmoeoff']
            memory_options = ['--memory-policy', 'matched']
        if reference_root is None:
            methods.insert(0, 'specter')
        for tier in tiers:
            label = f'speedup_{tier}' if len(tiers) > 1 else 'speedup'
            if model == 'dsv2lite':
                residency = DS_BASELINE_RESIDENCIES[tier]
                memory_options = ['--memory-policy', 'fixed',
                    '--mixtral-resident-per-layer', str(residency['mixtral-offloading']),
                    '--specmoeoff-resident-per-layer', str(residency['specmoeoff'])]
            command = python + ['baselines.run', '--config', str(config_file), '--model', model,
                '--methods', *methods, '--memory', tier, *memory_options,
                '--datasets', *protocol['datasets'], '--num-data', str(protocol['inputs_per_dataset'] if num_data is None else num_data),
                '--repeats', str(protocol['repeats']), '--tokens', str(protocol['output_tokens']),
                '--prefix-tokens', str(protocol['prefix_max_tokens']), '--output', str(output / label)]
            if reference_root is not None:
                command += ['--reference-root', str(Path(reference_root).expanduser().resolve())]
            if check_numa:
                command += ['--check-numa']
            if require_greedy_match:
                command += ['--require-greedy-match']
            commands.append((label, command))
        return commands
    if experiment in EXPERIMENTS:
        command = python + ['benchmarks.run_experiments', '--config', str(config_file),
                            '--experiment', experiment, '--output', str(output / experiment)]
        if check_numa:
            command += ['--check-numa']
        return [(experiment, command)]
    if experiment == 'kernel':
        command = python + ['benchmarks.kernel_bench', '--config', str(config_file),
                            '--output', str(output / 'kernel')]
        if check_numa:
            command += ['--check-numa']
        return [('kernel', command)]
    common = ['--config', str(config_file), '--model', 'dsv2lite',
              '--tokens', '128', '--repeats', '3', '--strategy', 'greedy']
    if check_numa:
        common += ['--check-numa']
    tpot = python + ['benchmarks.run_tpot'] + common + ['--gamma', '16', '--offload-per-layer', '48']
    if experiment in ('main', 'all'):
        protocol = json.loads((ROOT / 'benchmarks/protocol.json').read_text(encoding='utf-8'))
        tiers = list(MAIN_CASES) if memory == 'both' else [memory]
        for tier in tiers:
            case_name = MAIN_CASES[tier]
            case = protocol['cases'][case_name]
            label = 'main' if tier == 'high' else 'main_low'
            command = python + ['benchmarks.run_tpot', '--config', str(config_file),
                '--model', case['model'], '--protocol-case', case_name,
                '--gamma', str(case['gamma']), '--offload-per-layer', str(case['offload_per_layer']),
                '--tokens', str(protocol['output_tokens']), '--repeats', str(protocol['repeats']),
                '--strategy', protocol['strategy'], '--num-data', str(protocol['inputs_per_dataset']),
                '--datasets', *protocol['datasets'], '--prefix-tokens', str(protocol['prefix_max_tokens']),
                '--output', str(output / label)]
            if check_numa:
                command += ['--check-numa']
            commands.append((label, command))
    if experiment in ('depth', 'all'):
        commands.append(('depth', python + ['cli.profile'] + common + [
            '--prompts-json', str(output / 'depth' / 'prompts.json'),
            '--prefix-tokens', '16', '--seed', '42',
            '--output', str(output / 'depth' / 'depth_profile.json')]))
    if experiment in ('prefix', 'all'):
        for tokens in (8, 16, 32):
            commands.append((f'prefix_{tokens}', tpot + ['--datasets', 'GK', '--num-data', '3',
                '--prefix-tokens', str(tokens), '--output', str(output / f'prefix_{tokens}')]))
    return commands


def select_main_prompts(inputs):
    if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
        raise ValueError('main/inputs.json must contain the recorded input objects')
    gk = [item for item in inputs if item.get('dataset') == 'GK']
    selected = []
    for index in range(3):
        matches = [item for item in gk if item.get('index') == index]
        if len(matches) != 1:
            raise ValueError(f'main/inputs.json must contain exactly one GK input with index {index}')
        selected.append(matches[0].get('text'))
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in selected):
        raise ValueError('The first three GK prompts must be nonempty strings')
    return selected


def prepare_depth_prompts(config, output):
    main_inputs = output / 'main' / 'inputs.json'
    if main_inputs.exists():
        prompts = select_main_prompts(json.loads(main_inputs.read_text(encoding='utf-8')))
        source = str(main_inputs)
    else:
        from data.loader import prepare_data
        source = str(config.path('datasets', 'GK'))
        prompts = prepare_data(source, 3)
        if len(prompts) != 3 or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
            raise ValueError('Configured GK dataset must provide three nonempty prompts')
    directory = output / 'depth'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'prompts.json').write_text(json.dumps(prompts, indent=2), encoding='utf-8')
    (directory / 'prompt_source.json').write_text(json.dumps({
        'source': source, 'dataset': 'GK', 'indices': [0, 1, 2], 'selection': 'first three configured GK inputs',
    }, indent=2), encoding='utf-8')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from configuration import configure
    config = configure(args.config)
    output = config.output_path(args.output, 'ae')
    try:
        commands = build_commands(args.experiment, config.file, output, args.check_numa, args.memory,
                                  args.model, args.reference_root, args.require_greedy_match, args.num_data)
    except ValueError as error:
        parser.error(str(error))
    print(f'Output root: {output}', flush=True)
    for label, command in commands:
        if label == 'depth':
            main_inputs = output / 'main' / 'inputs.json'
            source = str(main_inputs) if main_inputs.exists() or args.experiment == 'all' else str(config.path('datasets', 'GK'))
            print(f'[prepare depth] first 3 GK inputs from {source} -> {output / "depth" / "prompts.json"}', flush=True)
        display = subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)
        print(f'[{label}] {display}', flush=True)
    if args.dry_run:
        return
    # Detect conflicting outputs before any expensive child process starts.
    for label, _ in commands:
        destination = output / label
        if destination.exists():
            raise FileExistsError(f'Use a fresh output path: {destination}')
    for label, command in commands:
        if label == 'depth':
            prepare_depth_prompts(config, output)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
