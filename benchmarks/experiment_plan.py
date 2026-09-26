"""Explicit workloads for optional system measurements."""
from dataclasses import asdict, dataclass
import math


MODELS = {'dsv2lite': (64, 16, 16), 'qwen2moe': (60, 15, 8), 'phimoe': (16, 2, 16)}
DATASETS = ('GK', 'WT', 'HE', 'GP', 'C4')
EXPERIMENTS = ('routing', 'working-set', 'sensitivity', 'cache', 'ablation',
               'latency', 'memory', 'fidelity', 'oracle')
DEPTHS = (2, 4, 6, 8, 12, 16, 20, 24, 32)


@dataclass(frozen=True)
class ExperimentCase:
    label: str
    dataset: str
    prefix_tokens: int
    depth: int
    resident: int
    strategy: str = 'greedy'
    temperature: float = 1.0
    policy: str = 'adaptive'
    kv_mode: str = 'shared'
    expert_mode: str = 'overlap'
    error_rate: float | None = None

    def to_dict(self):
        return asdict(self)


def build_cases(args):
    """Build a plan without importing Torch or touching datasets."""
    expert_count, default_resident, default_depth = MODELS[args.model]
    resident = default_resident if args.resident is None else args.resident
    depth = default_depth if args.depth is None else args.depth
    datasets = args.datasets or (list(DATASETS) if args.experiment == 'routing' else ['GK'])
    if args.prompts_json:
        if args.datasets:
            raise ValueError('--prompts-json and --datasets are mutually exclusive')
        datasets = ['CUSTOM']
    if args.tokens < 2 or args.num_data < 1 or args.repeats < 1:
        raise ValueError('tokens >= 2 and positive samples/repeats are required')
    if not 0 < resident <= expert_count or depth < 1 or args.prefix_tokens < 1:
        raise ValueError('Invalid expert residency, depth, or prefix length')
    if len(set(datasets)) != len(datasets):
        raise ValueError('Datasets must be unique')
    cases = []

    def add(label, dataset, **overrides):
        values = dict(label=label, dataset=dataset, prefix_tokens=args.prefix_tokens,
                      depth=depth, resident=resident)
        values.update(overrides)
        cases.append(ExperimentCase(**values))

    if args.experiment == 'working-set':
        candidates = args.depths or DEPTHS
        for dataset in datasets:
            for value in candidates:
                add(f'{dataset}_k{value}', dataset, depth=value)
    elif args.experiment == 'sensitivity':
        domains = datasets if args.datasets or args.prompts_json else ['WT', 'C4', 'GK']
        for dataset in domains:
            add(f'domain_{dataset}', dataset)
        dataset = datasets[0]
        for length in args.prefix_lengths:
            add(f'prefix_{length}', dataset, prefix_tokens=length)
        for temperature in (0.7, 1.0):
            add(f'sampling_{temperature:g}', dataset, strategy='sampling', temperature=temperature)
        add('greedy', dataset)
    elif args.experiment == 'cache':
        for dataset in datasets:
            for count in args.residents or [resident]:
                for policy in args.policies:
                    add(f'{dataset}_resident{count}_{policy}', dataset, resident=count, policy=policy)
    elif args.experiment == 'ablation':
        variants = [('full', 'adaptive', 'shared', 'overlap'),
                    ('no_prefetch', 'none', 'shared', 'overlap'),
                    ('copied_kv', 'none', 'copied', 'overlap'),
                    ('serial_experts', 'none', 'copied', 'serial')]
        for dataset in datasets:
            for name, policy, kv, expert in variants:
                add(f'{dataset}_{name}', dataset, policy=policy, kv_mode=kv, expert_mode=expert)
    elif args.experiment == 'oracle':
        for dataset in datasets:
            for rate in args.error_rates:
                if not math.isfinite(rate) or not 0 <= rate <= 1:
                    raise ValueError('Error rates must be finite and in [0, 1]')
                add(f'{dataset}_error{rate:g}', dataset, error_rate=rate)
    elif args.experiment == 'memory':
        for dataset in datasets:
            for mode in ('shared', 'copied'):
                add(f'{dataset}_{mode}', dataset, kv_mode=mode)
    else:
        for dataset in datasets:
            add(dataset, dataset)
    for case in cases:
        if case.depth < 1 or case.prefix_tokens < 1 or not 0 < case.resident <= expert_count:
            raise ValueError(f'Invalid case: {case.label}')
    if len({case.label for case in cases}) != len(cases):
        raise ValueError('Sweep values must be unique')
    if args.experiment == 'oracle' and any(case.depth < 2 for case in cases):
        raise ValueError('Oracle experiments require depth >= 2 so the adaptive planner submits predictions')
    if args.experiment != 'fidelity':
        minimum_tokens = max(case.depth for case in cases) + 2
        if args.tokens < minimum_tokens:
            raise ValueError(f'--tokens must be at least {minimum_tokens} (maximum depth + 2) '
                             'to measure a full first window and a later token commitment')
    return cases


def select_inputs(prompts, tokenizer, count, required_tokens, allow_short=False):
    """Use the same qualifying source inputs across a length sweep."""
    selected = []
    for source_index, text in enumerate(prompts):
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = tokenizer.encode(text)
        if not token_ids or (not allow_short and len(token_ids) < required_tokens):
            continue
        selected.append({'source_index': source_index, 'text': text, 'input_ids': token_ids,
                         'source_tokens': len(token_ids)})
        if len(selected) == count:
            return selected
    raise ValueError(f'Need {count} nonempty inputs with at least {required_tokens} tokens; '
                     f'found {len(selected)}. Supply longer prompts or --allow-short-prefix.')


def seed_for(seed, dataset, source_index, repeat):
    """Stable seeds do not depend on case order, process hash, or policy."""
    import hashlib
    digest = hashlib.sha256(f'{dataset}:{source_index}'.encode()).digest()
    return (seed + int.from_bytes(digest[:4], 'little') + repeat * 100003) % (2**31)
