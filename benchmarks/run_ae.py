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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['main', 'depth', 'prefix', 'all', 'kernel', *EXPERIMENTS], default='main')
    parser.add_argument('--config', help='External TOML configuration (or SPECTER_CONFIG).')
    parser.add_argument('--output', help='Output name under paths.output, or an absolute external path; default: ae.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without importing Torch, loading data, or writing outputs.')
    parser.add_argument('--check-numa', action='store_true', help='Require the configured CPU and preferred-memory NUMA placement.')
    return parser


def build_commands(experiment, config_file, output, check_numa=False):
    """Keep the chosen workload explicit and inspectable before GPU execution."""
    output = Path(output)
    commands = []
    python = [sys.executable, '-B', '-m']
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
        commands.append(('main', tpot + ['--protocol-case', 'ds_high', '--num-data', '5',
            '--datasets', 'GK', 'WT', 'HE', 'GP', 'C4', '--prefix-tokens', '16',
            '--output', str(output / 'main')]))
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
    args = build_parser().parse_args(argv)
    from configuration import configure
    config = configure(args.config)
    output = config.output_path(args.output, 'ae')
    commands = build_commands(args.experiment, config.file, output, args.check_numa)
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
