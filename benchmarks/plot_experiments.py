"""Plot completed measured experiments and export the exact backing tables."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import textwrap


def number(value, label, *, fraction=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if value < 0 or (fraction and value > 1):
        raise ValueError(f'{label} is outside its valid range')
    return float(value)


def describe(values):
    values = sorted(number(value, 'observation') for value in values)
    if not values:
        raise ValueError('Cannot summarize empty observations')

    def quantile(q):
        index = q * (len(values) - 1)
        lower = math.floor(index)
        return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)

    return {'count': len(values), 'mean': statistics.mean(values),
            'p10': quantile(0.1), 'p90': quantile(0.9),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0}


def js_distance(left, right):
    """Base-2 Jensen-Shannon distance between nonempty count distributions."""
    for counts in (left, right):
        if not counts or sum(number(value, 'expert count') for value in counts.values()) <= 0:
            raise ValueError('JS distance requires nonempty expert distributions')
    totals = sum(left.values()), sum(right.values())
    divergence = 0.0
    for key in left.keys() | right.keys():
        p, q = left.get(key, 0) / totals[0], right.get(key, 0) / totals[1]
        midpoint = (p + q) / 2
        if p:
            divergence += p * math.log2(p / midpoint) / 2
        if q:
            divergence += q * math.log2(q / midpoint) / 2
    return math.sqrt(max(0.0, divergence))


def _reject_constant(value):
    raise ValueError(f'Invalid JSON number: {value}')


def _json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read {path.name}: {error}') from error


def _jsonl(path):
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise ValueError(f'Cannot read {path.name}: {error}') from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f'{path.name} is empty or contains blank records')
    try:
        rows = [json.loads(line, parse_constant=_reject_constant) for line in lines]
    except json.JSONDecodeError as error:
        raise ValueError(f'Malformed {path.name}: {error}') from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'{path.name} must contain JSON objects')
    return rows


def _key(record):
    return record['case']['label'], record['sample_index'], record['repeat']


def load_results(directory):
    """Require a complete run and one paired diagnostic for each measurement."""
    directory = Path(directory)
    if (directory / 'results.json').is_file():
        report = _json(directory / 'results.json')
        if report.get('complete') is not True or not report.get('results'):
            raise ValueError('Kernel results are empty or incomplete')
        planned = report['plan']['cases']
        if [row['case'] for row in report['results']] != planned:
            raise ValueError('Kernel results do not cover the planned cases')
        return {'experiment': 'kernel', 'kernel': report}
    config = _json(directory / 'config.json')
    summary = _json(directory / 'summary.json')
    records = _jsonl(directory / 'records.jsonl')
    expected = summary.get('expected_records')
    if (summary.get('complete') is not True or not isinstance(expected, int) or isinstance(expected, bool)
            or expected < 1 or summary.get('completed_records') != expected
            or len(records) != expected or config.get('expected_records') != expected):
        raise ValueError('Experiment is incomplete or its record counts disagree')
    cases = {case['label']: case for case in config['cases']}
    if not cases or len(cases) != len(config['cases']):
        raise ValueError('Planned cases are empty or duplicated')
    labels = Counter(record['case']['label'] for record in records)
    if set(labels) != set(cases) or len({_key(record) for record in records}) != len(records):
        raise ValueError('Measurements are missing cases or have duplicate sample keys')
    if any(record['case'] != cases[record['case']['label']] for record in records):
        raise ValueError('Measured case differs from its planned configuration')
    sample_count, repeats = config['args']['num_data'], config['args']['repeats']
    expected_keys = {(label, sample, repeat) for label in cases
                     for sample in range(sample_count) for repeat in range(repeats)}
    if {_key(record) for record in records} != expected_keys:
        raise ValueError('Raw measurements do not cover all planned sample/repeat pairs')
    summary_counts = {row['label']: row['samples'] for row in summary['rows']}
    if summary_counts != dict(labels) or len(summary_counts) != len(summary['rows']):
        raise ValueError('Summary case counts disagree with raw measurements')
    experiment = config['experiment']
    diagnostics = [] if experiment == 'fidelity' else _jsonl(directory / 'diagnostics.jsonl')
    if diagnostics:
        measured = {_key(record): record for record in records}
        if len(diagnostics) != len(records) or {_key(row) for row in diagnostics} != set(measured):
            raise ValueError('Diagnostics do not pair one-to-one with timed measurements')
        for row in diagnostics:
            partner = measured[_key(row)]
            if row.get('role') != 'diagnostic' or row['case'] != partner['case']:
                raise ValueError('Invalid diagnostic role or configuration')
            expected_kind = experiment if experiment in {'latency', 'memory'} else 'routing'
            if row.get('diagnostic') != expected_kind:
                raise ValueError('Diagnostic kind does not match the planned experiment')
            if row.get('token_ids') != partner.get('token_ids') or not row.get('token_ids'):
                raise ValueError('Diagnostic and timed token sequences differ or are absent')
    expected_role = 'fidelity' if experiment == 'fidelity' else 'timed'
    if any(record.get('role') != expected_role for record in records):
        raise ValueError('Unexpected measurement role')
    return {'experiment': experiment, 'config': config, 'summary': summary,
            'records': records, 'diagnostics': diagnostics}


def route_observations(diagnostics):
    rows, expected_layers = [], None
    for record in diagnostics:
        windows = record['routing']['windows']
        if not windows:
            raise ValueError('Routing diagnostics have no speculative windows')
        for window in windows:
            if window.get('status') != 'complete' or not window.get('layers'):
                raise ValueError('Routing diagnostics contain incomplete windows')
            layers = [layer['layer_id'] for layer in window['layers']]
            if any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 for layer in layers):
                raise ValueError('Decoder layer IDs must be nonnegative integers')
            if len(layers) != len(set(layers)):
                raise ValueError('Duplicate decoder layers in a routing window')
            if expected_layers is None:
                expected_layers = set(layers)
            if set(layers) != expected_layers:
                raise ValueError('Routing windows do not cover the same decoder layers')
            for layer in window['layers']:
                counts = {}
                for role in ('draft', 'target'):
                    routes = layer[f'{role}_ids']
                    if not routes or any(not token for token in routes):
                        raise ValueError('A complete layer-window has empty routes')
                    if any(not isinstance(expert, int) or isinstance(expert, bool) or expert < 0
                           for token in routes for expert in token):
                        raise ValueError('Invalid expert identity')
                    if len({len(token) for token in routes}) != 1 or any(len(set(token)) != len(token) for token in routes):
                        raise ValueError('Routes require a constant top-k without duplicate experts')
                    counts[role] = Counter(expert for token in routes for expert in token)
                rows.append({'case': record['case'], 'layer': layer['layer_id'],
                             'counts': counts,
                             'recall': len(counts['draft'].keys() & counts['target'].keys()) / len(counts['target'])})
    if not rows:
        raise ValueError('No completed routing observations')
    return rows


def routing_distances(observations):
    """Pool assignments within each layer, then give layers equal JS weight."""
    counts = defaultdict(Counter)
    for row in observations:
        for role in ('draft', 'target'):
            counts[(role, row['case']['dataset'], row['layer'])].update(row['counts'][role])
    datasets = sorted({key[1] for key in counts})
    layers = sorted({key[2] for key in counts})
    frequencies, distances, layer_distances = [], [], []
    for role in ('draft', 'target'):
        for dataset in datasets:
            for layer in layers:
                values = counts[(role, dataset, layer)]
                if not values:
                    raise ValueError('Every dataset must cover every measured decoder layer')
                total = sum(values.values())
                for expert, count in sorted(values.items()):
                    frequencies.append({'role': role, 'dataset': dataset, 'layer': layer,
                                        'expert': expert, 'assignments': count,
                                        'layer_assignments': total, 'probability': count / total})
        for left in datasets:
            for right in datasets:
                values = []
                for layer in layers:
                    value = js_distance(counts[(role, left, layer)], counts[(role, right, layer)])
                    layer_distances.append({'role': role, 'dataset_a': left, 'dataset_b': right,
                                            'layer': layer, 'distance': value})
                    values.append(value)
                distances.append({'role': role, 'dataset_a': left, 'dataset_b': right,
                                  'distance': statistics.mean(values), 'layers': len(values)})
    return frequencies, layer_distances, distances


def _aggregate(records, keys, field, *, fraction=False):
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record['case'][key] for key in keys)].append(number(record.get(field), field, fraction=fraction))
    return [{**dict(zip(keys, group)), **describe(values)} for group, values in groups.items()]


def prepare_tables(bundle):
    """Create finite, measured tables before importing a plotting backend."""
    experiment = bundle['experiment']
    tables = {}

    def table(name, rows, notes, *, kind=None, x=None, y='mean', group=None, ylabel=None, xlabel=None):
        if not rows:
            raise ValueError(f'No measured observations for {name}')
        if kind and kind != 'scatter':
            coordinates = [(row[x], row[y] if kind == 'heatmap' else row[group] if group else None) for row in rows]
            if len(set(coordinates)) != len(coordinates):
                raise ValueError(f'Duplicate coordinates in {name}; additional grouping is required')
        tables[name] = {'rows': rows, 'notes': notes,
                        'plot': {'kind': kind, 'x': x, 'y': y, 'group': group,
                                 'ylabel': ylabel, 'xlabel': xlabel}}

    if experiment == 'kernel':
        names = {'local_w4a16_batched': 'W4A16 batched', 'local_w4a16_sequential': 'W4A16 sequential',
                 'dequantized_fp16_sequential': 'FP16 sequential'}
        rows = []
        for result in bundle['kernel']['results']:
            if not result['correctness'] or any(check.get('passed') is not True for check in result['correctness'].values()):
                raise ValueError('Kernel correctness checks did not all pass')
            if set(result['timings']) != set(names):
                raise ValueError('Kernel benchmark must contain all three current paths')
            for path, timing in result['timings'].items():
                values = [number(value, 'kernel timing') for value in timing['samples_ms']]
                if len(values) != result['case']['trials'] or not values or min(values) <= 0:
                    raise ValueError('Kernel trials are missing or invalid')
                rows.append({'tokens': result['case']['tokens'], 'path': names[path],
                             'median_ms': statistics.median(values), 'min_ms': min(values),
                             'max_ms': max(values), 'samples_ms': values})
        table('kernel', rows, [bundle['kernel']['plan']['scope'], bundle['kernel']['plan']['timing']],
              kind='line', x='tokens', y='median_ms', group='path', ylabel='Projection latency (ms)', xlabel='Tokens')
        return tables

    records, diagnostics = bundle['records'], bundle['diagnostics']
    equal_samples = 'Arithmetic mean over recorded prompt/repeat measurements; TPOT uses the clean timed pass.'
    layer_windows = 'Each completed layer-window receives equal weight, including all recorded repeats; prefill and target tails excluded. Percentiles use linear interpolation.'
    if experiment not in {'fidelity', 'latency', 'memory'}:
        observations = route_observations(diagnostics)
    if experiment in {'routing', 'working-set'}:
        groups = defaultdict(list)
        for row in observations:
            for role in ('draft', 'target'):
                key = (row['case']['dataset'], row['case']['depth'], role)
                groups[key].append(len(row['counts'][role]))
        working = [{'dataset': dataset, 'depth': depth, 'role': role,
                    'series': f'{dataset} / {role}', **describe(values)}
                   for (dataset, depth, role), values in groups.items()]
        table('working_set', working, [layer_windows], kind='bars' if experiment == 'routing' else 'line',
              x='dataset' if experiment == 'routing' else 'depth',
              group='role' if experiment == 'routing' else 'series',
              ylabel='Distinct experts per layer-window', xlabel='Dataset' if experiment == 'routing' else 'Draft depth K')
        if experiment == 'routing':
            frequencies, per_layer, distances = routing_distances(observations)
            weighting = ['Expert assignments are pooled across all windows and repeats separately for each dataset and decoder layer.',
                         'Normalize each layer to unit probability. Omitted experts have zero observed assignments.',
                         'Compute sqrt(base-2 Jensen-Shannon divergence) within each layer; heatmaps show the arithmetic mean of layer distances.']
            table('expert_frequencies', frequencies, weighting)
            table('layer_distances', per_layer, weighting)
            for role in ('draft', 'target'):
                table(f'js_{role}', [row for row in distances if row['role'] == role], weighting,
                      kind='heatmap', x='dataset_a', y='dataset_b', group='distance', ylabel='Dataset', xlabel='Dataset')
        else:
            table('tpot', _aggregate(records, ['dataset', 'depth'], 'tpot_ms'), [equal_samples],
                  kind='line', x='depth', group='dataset', ylabel='TPOT (ms/token)', xlabel='Draft depth K')
    elif experiment == 'sensitivity':
        table('tpot', _aggregate(records, ['label'], 'tpot_ms'), [equal_samples],
              kind='bars', x='label', ylabel='TPOT (ms/token)', xlabel='Case')
        groups = defaultdict(list)
        for row in observations:
            groups[row['case']['label']].append(row['recall'])
        table('window_recall', [{'label': label, **describe(values)} for label, values in groups.items()],
              [layer_windows, 'Recall denominator: distinct target experts in the same verification layer-window.'],
              kind='bars', x='label', ylabel='Target working-set recall', xlabel='Case')
    elif experiment == 'cache':
        for name, field, ylabel in [('tpot', 'tpot_ms', 'TPOT (ms/token)'),
                                    ('cache_hit', 'cache_hit_rate', 'Demand residency hit rate')]:
            rows = _aggregate(records, ['dataset', 'resident', 'policy'], field, fraction=name == 'cache_hit')
            for row in rows:
                row['series'] = f"{row['dataset']} / {row['policy']}"
            residents = {row['resident'] for row in rows}
            single_resident = len(residents) == 1
            table(name, rows, [equal_samples, 'Cache hit rate is first averaged within each verification window, then within each decode; in-flight residents count as hits.'],
                  kind='bars' if single_resident else 'line', x='policy' if single_resident else 'resident',
                  group='dataset' if single_resident else 'series', ylabel=ylabel,
                  xlabel=f'Prefetch policy ({next(iter(residents))} resident experts per layer)' if single_resident else 'Resident experts per layer')
    elif experiment == 'ablation':
        rows = _aggregate(records, ['dataset', 'policy', 'kv_mode', 'expert_mode'], 'tpot_ms')
        for row in rows:
            parts = []
            if row['policy'] == 'none':
                parts.append('No prefetch')
            elif row['policy'] != 'adaptive':
                parts.append(f"{row['policy']} prefetch")
            if row['kv_mode'] == 'copied':
                parts.append('copied KV')
            if row['expert_mode'] == 'serial':
                parts.append('serial demand')
            row['variant'] = ' + '.join(parts) if parts else 'Full'
        table('ablation', rows, [equal_samples, 'Partial mechanism ablations retain canonical acceptance and target verification; copied KV and serial demand do not remove all SIC or SEE mechanisms.'],
              kind='bars', x='variant', group='dataset', ylabel='TPOT (ms/token)', xlabel='Enabled configuration changes')
    elif experiment == 'fidelity':
        histograms, layers = defaultdict(Counter), defaultdict(list)
        expected_layers = None
        for record in records:
            count = 0
            if not record['fidelity']['layers']:
                raise ValueError('Fidelity result contains no decoder layers')
            layer_ids = [layer['layer_id'] for layer in record['fidelity']['layers']]
            if expected_layers is None:
                expected_layers = set(layer_ids)
            if len(set(layer_ids)) != len(layer_ids) or set(layer_ids) != expected_layers:
                raise ValueError('Fidelity records have duplicate or inconsistent decoder layers')
            for layer in record['fidelity']['layers']:
                values = [number(value, 'fidelity recall', fraction=True) for value in layer['token_target_recall']]
                if not values or len(values) != layer['token_count']:
                    raise ValueError('Fidelity layer is empty or incomplete')
                histograms[record['case']['dataset']].update(values)
                layers[(record['case']['dataset'], layer['layer_id'])].extend(values)
                count += len(values)
            if count != record['fidelity']['token_layer_count']:
                raise ValueError('Fidelity token-layer count is inconsistent')
        notes = ['Matched input tokens, positions, and independent empty KV caches; each token-layer pair receives equal weight, including repeats.']
        rows = [{'dataset': dataset, 'recall': recall, 'count': count,
                 'fraction': count / sum(values.values())}
                for dataset, values in histograms.items() for recall, count in sorted(values.items())]
        table('fidelity_histogram', rows, notes, kind='bars', x='recall', y='fraction', group='dataset',
              ylabel='Fraction of token-layer pairs', xlabel='Matched-token target recall')
        table('fidelity_layers', [{'dataset': dataset, 'layer': layer, **describe(values)}
                                  for (dataset, layer), values in layers.items()], notes,
              kind='line', x='layer', group='dataset', ylabel='Mean matched-token recall', xlabel='Decoder layer')
    elif experiment == 'latency':
        from benchmarks.latency import summarize_intervals
        groups = defaultdict(list)
        raw = []
        for diagnostic in diagnostics:
            trace = diagnostic['latency']
            if trace.get('failed') is not False or not trace.get('records'):
                raise ValueError('Latency trace is failed or empty')
            iterations = defaultdict(list)
            for event in trace['records']:
                if event.get('incomplete'):
                    raise ValueError('Latency trace contains an incomplete window')
                if event['iteration'] >= 0 and event['phase'] in {'draft', 'prefetch_barrier', 'verify'}:
                    iterations[event['iteration']].append(event)
            if not iterations:
                raise ValueError('Latency trace has no speculative iterations')
            if set(iterations) != set(range(max(iterations) + 1)):
                raise ValueError('Latency trace has missing speculative iterations')
            for iteration, events in iterations.items():
                kinds = {event['kind'] for event in events}
                if not {'draft_window', 'target_verify'} <= kinds:
                    raise ValueError('Latency iteration lacks draft or verification events')
                summary = summarize_intervals(events)
                values = {'Draft window': summary['span_union_ms']['draft_window'],
                          'Target forward': summary['span_union_ms']['target_verify'],
                          'H2D copies': summary['h2d_union_ms'],
                          'Compute-stream waits': summary['exposed_wait_union_ms']}
                for metric, value in values.items():
                    groups[(diagnostic['case']['label'], metric)].append(value)
                    raw.append({'case': diagnostic['case']['label'], 'sample_index': diagnostic['sample_index'],
                                'repeat': diagnostic['repeat'], 'iteration': iteration, 'metric': metric, 'ms': value})
        notes = ['Means over complete speculative iterations; prefill and the standalone target tail are excluded.',
                 'All times are CUDA event intervals. Intersections are counted once within each metric.',
                 'Phases and transfers overlap and must not be summed; target-forward spans include launch gaps and waits, not only kernel compute.',
                 'Traces come from a separate instrumented diagnostic pass.']
        table('latency_iterations', raw, notes)
        table('latency', [{'case': case, 'metric': metric, **describe(values)}
                          for (case, metric), values in groups.items()], notes,
              kind='bars', x='case', group='metric', ylabel='Mean interval duration (ms)', xlabel='Case')
    elif experiment == 'memory':
        raw = []
        groups = defaultdict(list)
        for record in diagnostics:
            snapshots = record['memory']
            if [item['label'] for item in snapshots] != ['loaded', 'prefill', 'first_commit']:
                raise ValueError('Memory snapshots must cover loaded, prefill, and first_commit stages')
            for snapshot in snapshots:
                report = snapshot['report']
                if report.get('complete') is not True:
                    raise ValueError('Memory storage traversal is incomplete')

                def gpu(devices):
                    return sum(number(value, 'GPU storage bytes') for device, value in devices.items() if device.split(':')[0] == 'cuda')

                total = gpu(report['by_device'])
                if total <= 0:
                    raise ValueError('Memory snapshot has no measured CUDA storage')
                row = {'dataset': record['case']['dataset'], 'kv_mode': record['case']['kv_mode'],
                       'stage': snapshot['label'], 'sample_index': record['sample_index'], 'repeat': record['repeat'],
                       'gpu_bytes': total, 'shared_model_gpu_bytes': gpu(report['models']['shared_by_device']),
                       'kv_gpu_bytes': gpu(report['cache']['by_device']),
                       'shared_kv_gpu_bytes': gpu(report['cache']['shared_by_device'])}
                raw.append(row)
                groups[(row['dataset'], row['kv_mode'], row['stage'])].append(row)
        notes = ['Live reachable Torch storage regions, deduplicated by memory address; retained capacity differs from CUDA allocator occupancy and peak allocation.']
        table('memory_snapshots', raw, notes)
        for name, field, divisor, ylabel in [('memory_storage', 'gpu_bytes', 2**30, 'Retained GPU storage (GiB)'),
                                           ('memory_kv', 'kv_gpu_bytes', 2**20, 'Unique retained KV storage (MiB)')]:
            rows = [{'dataset': dataset, 'kv_mode': mode, 'stage': stage, 'series': f'{dataset} / {mode}',
                     **describe(row[field] / divisor for row in values)}
                    for (dataset, mode, stage), values in groups.items()]
            table(name, rows, notes, kind='bars', x='stage', group='series', ylabel=ylabel, xlabel='Capture stage')
        rows = _aggregate(records, ['dataset', 'kv_mode'], 'peak_allocated_bytes')
        for row in rows:
            for key in ('mean', 'p10', 'p90', 'stdev'):
                row[key] /= 2**30
        table('memory_peak', rows, [equal_samples, 'Peak CUDA allocation in the timed decode; not a storage-sharing estimate.'],
              kind='bars', x='dataset', group='kv_mode', ylabel='Peak CUDA allocation (GiB)', xlabel='Dataset')
    elif experiment == 'oracle':
        groups = defaultdict(list)
        for record in records:
            if record.get('oracle_output_matches') is not True:
                raise ValueError('Oracle output equivalence was not verified')
            precision = number(record.get('prefetch_accuracy'), 'actual prefetch precision', fraction=True)
            rate = number(record['case']['error_rate'], 'injected error rate', fraction=True)
            groups[(record['case']['dataset'], rate)].append((precision, number(record.get('tpot_ms'), 'TPOT')))
        rows = [{'dataset': dataset, 'injected_error_rate': rate,
                 'precision': statistics.mean(pair[0] for pair in values),
                 'tpot_ms': statistics.mean(pair[1] for pair in values), 'count': len(values),
                 'label': f'{dataset}: injected {rate:g}'} for (dataset, rate), values in groups.items()]
        table('oracle', rows, [equal_samples, 'Precision is the measured distinct predicted-expert precision, averaged over layers and used windows within each decode; injected error rate is not used as a substitute.'],
              kind='scatter', x='precision', y='tpot_ms', group='dataset',
              ylabel='TPOT (ms/token)', xlabel='Actual prefetch prediction precision\nPoint labels: injected error fraction')
    else:
        raise ValueError(f'Unsupported measured experiment: {experiment}')
    return tables


def render_tables(tables, output):
    """Write tables and static PDF/PNG figures using a lazy matplotlib import."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = 'Arial' if any(item.name == 'Arial' for item in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    plt.rcParams.update({'font.family': font, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 180})
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in tables.items():
        rows, plot = payload['rows'], payload['plot']
        (output / f'{name}.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                              for key, value in row.items()} for row in rows)
        kind = plot['kind']
        if kind is None:
            continue
        x, y, group = plot['x'], plot['y'], plot['group']
        category_count = len({row[x] for row in rows})
        width = max(7.8, min(16, 1.5 * category_count)) if kind == 'bars' else 7.8
        series = list(dict.fromkeys(row[group] for row in rows)) if group and kind != 'heatmap' else [None]
        height = 4.8 + (0.25 * math.ceil(len(series) / 4) if group else 0)
        fig, ax = plt.subplots(figsize=(6.0, 5.2) if kind == 'heatmap' else (width, height), layout='constrained')
        if kind == 'heatmap':
            labels = sorted({row[x] for row in rows})
            matrix = {(row[x], row[y]): row[group] for row in rows}
            values = [[matrix[(left, right)] for left in labels] for right in labels]
            artist = ax.imshow(values, vmin=0, vmax=1, cmap='YlOrBr')
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks(range(len(labels)), labels)
            for j, line in enumerate(values):
                for i, value in enumerate(line):
                    ax.text(i, j, f'{value:.3f}', ha='center', va='center',
                            color='white' if value > 0.65 else 'black')
            fig.colorbar(artist, ax=ax, label='Mean layerwise JS distance')
        else:
            categories = list(dict.fromkeys(row[x] for row in rows))
            if len(series) > 10:
                ax.set_prop_cycle(color=plt.get_cmap('tab20').colors)
            if kind in {'line', 'scatter'} or all(isinstance(value, (int, float)) for value in categories):
                categories.sort()
            for index, label in enumerate(series):
                selected = [row for row in rows if group is None or row[group] == label]
                selected.sort(key=lambda row: categories.index(row[x]))
                if kind != 'scatter' and len({row[x] for row in selected}) != len(selected):
                    raise ValueError(f'Duplicate plot coordinates in {name}; additional grouping is required')
                ys = [row[y] for row in selected]
                if kind == 'bars':
                    width = 0.78 / len(series)
                    xs = [categories.index(row[x]) + (index - (len(series) - 1) / 2) * width for row in selected]
                    ax.bar(xs, ys, width * 0.92, label=label)
                    if all('p10' in row and 'p90' in row for row in selected):
                        # Percentile endpoints need not bracket the mean.
                        ax.vlines(xs, [row['p10'] for row in selected], [row['p90'] for row in selected],
                                  color='black', linewidth=1)
                        ax.scatter(xs, ys, color='black', marker='_', s=35, zorder=3)
                else:
                    xs = [row[x] for row in selected]
                    if kind == 'line':
                        ax.plot(xs, ys, marker='o', label=label)
                    else:
                        ax.scatter(xs, ys, label=label)
                        midpoint = (min(categories) + max(categories)) / 2
                        for a, b, row in zip(xs, ys, selected):
                            right = a <= midpoint
                            ax.annotate(f"{row['injected_error_rate']:g}", (a, b),
                                        xytext=(6 if right else -6, 6), ha='left' if right else 'right',
                                        textcoords='offset points', fontsize=9)
            if kind == 'bars':
                labels = [f'{value:.3g}' if isinstance(value, (int, float)) else textwrap.fill(str(value).replace('_', ' '), 22)
                          for value in categories]
                ax.set_xticks(range(len(categories)), labels)
            elif kind == 'line':
                stride = max(1, math.ceil(len(categories) / 12))
                ticks = list(dict.fromkeys(categories[::stride] + categories[-1:]))
                ax.set_xticks(ticks)
            ax.margins(y=0.12 if kind == 'scatter' else 0.07)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.25)
            ax.set_axisbelow(True)
            if group:
                ax.legend(frameon=False, fontsize=9, loc='lower center',
                          bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(series)))
        ax.set_xlabel(plot['xlabel'])
        ax.set_ylabel(plot['ylabel'])
        for extension in ('pdf', 'png'):
            fig.savefig(output / f'{name}.{extension}')
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Completed external experiment directory.')
    parser.add_argument('--output', help='Fresh external figure directory; defaults to INPUT/figures.')
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parents[1]
    directory = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else directory / 'figures'
    for path in (directory, output):
        if path == source_root or source_root in path.parents:
            parser.error('Input and output must be outside the source directory')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing figure directory: {output}')
    try:
        tables = prepare_tables(load_results(directory))
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f'Malformed measured result: {error}') from error
    render_tables(tables, output)
    print(f'Wrote measured tables and figures: {output}')


if __name__ == '__main__':
    main()
