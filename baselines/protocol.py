"""Explicit workload and residency settings for baseline measurements."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('GK', 'WT', 'HE', 'GP', 'C4')
METHODS = ('specter', 'mixtral-offloading', 'specmoeoff')
MODEL_CASES = {
    'dsv2lite': {'high': 'ds_high', 'low': 'ds_low4'},
    'qwen2moe': {'high': 'qwen_high', 'low': 'qwen_low'},
    'phimoe': {'high': 'phi_high', 'low': 'phi_low'},
}
TPOT_DEFINITION = '(last committed token CUDA timestamp - first committed token CUDA timestamp) / (N-1)'


def settings(model, memory, method):
    protocol = json.loads((ROOT / 'benchmarks/protocol.json').read_text(encoding='utf-8'))
    name = MODEL_CASES[model][memory]
    case = dict(protocol['cases'][name])
    case['protocol_case'] = name
    if method == 'mixtral-offloading':
        case.update(gamma=1, buffer_size=4)
        if model == 'dsv2lite' and memory == 'high':
            case.update(resident_per_layer=32, offload_per_layer=32)
    elif method == 'specmoeoff':
        case.update(gamma=3, buffer_size=0 if model == 'phimoe' else 32)
    elif method != 'specter':
        raise ValueError(f'Unknown method: {method}')
    return case


def workload_arguments(parser):
    parser.add_argument('--config', help='External resource TOML, or SPECTER_CONFIG.')
    parser.add_argument('--model', choices=MODEL_CASES, default='dsv2lite')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--num-data', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--prefix-tokens', type=int, default=16)
    parser.add_argument('--check-numa', action='store_true')
    parser.add_argument('--require-greedy-match', action='store_true')
    parser.add_argument('--output', help='Output name under paths.output, or an external absolute path.')


def validate_workload(args):
    if args.tokens < 33 or min(args.num_data, args.repeats, args.prefix_tokens) < 1:
        raise ValueError('tokens >= 33 and positive samples, repeats, and prefix length required')
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError('Dataset names must be unique')
