"""Scoped route diagnostics. Run separately from latency measurements."""
from collections import Counter
import math
import statistics


def route_specs(case, model):
    """Return decoder layer IDs and routed blocks for a supported model."""
    current = model
    for _ in range(5):
        if hasattr(current, 'layers'):
            break
        current = getattr(current, 'model', None)
        if current is None:
            raise ValueError('Cannot locate decoder layers')
    else:
        raise ValueError('Cannot locate decoder layers')
    name = case.name
    if name not in ('qwen2moe', 'dsv2lite', 'phimoe'):
        raise ValueError(f'Unsupported route model: {name}')
    attr = 'block_sparse_moe' if name == 'phimoe' else 'mlp'
    specs = [(index, getattr(layer, attr)) for index, layer in enumerate(current.layers)
             if not (getattr(case, 'skip_first_layer', False) and index == 0)]
    if len(specs) != case.layer_num:
        raise ValueError(f'Expected {case.layer_num} routed layers, got {len(specs)}')
    return specs


def _rows(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    rows = []
    width = None
    for row in value:
        if not isinstance(row, (list, tuple)):
            raise ValueError('Routes must have token x top-k shape')
        if width is None:
            width = len(row)
        if len(row) != width or not row:
            raise ValueError('Route rows must have a constant positive top-k')
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in row):
            raise ValueError('Expert IDs must be nonnegative integers')
        if len(set(row)) != len(row):
            raise ValueError('A token cannot select an expert twice')
        rows.append(list(row))
    return rows


def route_statistics(routes, num_experts=None):
    rows = _rows(routes)
    counts = Counter(expert for row in rows for expert in row)
    if num_experts is not None:
        if num_experts <= 0 or any(x >= num_experts for x in counts):
            raise ValueError('Expert IDs must fit the expert universe')
    assignments = sum(counts.values())
    entropy = -sum((n / assignments) * math.log(n / assignments) for n in counts.values())
    adjacent = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(rows, rows[1:])]
    return {
        'token_count': len(rows), 'assignments': assignments,
        'expert_ids': sorted(counts), 'expert_counts': {str(k): counts[k] for k in sorted(counts)},
        'unique_count': len(counts),
        'working_set_fraction': len(counts) / num_experts if num_experts else None,
        'distinct_token_sets': len({tuple(sorted(row)) for row in rows}),
        'entropy_nats': entropy,
        'effective_experts': math.exp(entropy) if assignments else 0.0,
        'normalized_entropy': entropy / math.log(num_experts) if num_experts and num_experts > 1 else None,
        'adjacent_token_jaccard_mean': statistics.mean(adjacent) if adjacent else None,
    }


def window_recall(draft_routes, target_routes):
    draft = {x for row in _rows(draft_routes) for x in row}
    target = {x for row in _rows(target_routes) for x in row}
    overlap = len(draft & target)
    return {'intersection_count': overlap, 'target_unique_count': len(target),
            'draft_unique_count': len(draft),
            'target_recall': overlap / len(target) if target else None}


def matched_route_fidelity(draft_routes, target_routes):
    """Compare routes for identical input tokens and positions, layer by layer.

    Inputs are layer-ID mappings to token x top-k IDs. Callers must supply
    matched token sequences and cache contexts; speculative windows do not
    establish that precondition.
    """
    draft_routes = {int(k): _rows(v) for k, v in draft_routes.items()}
    target_routes = {int(k): _rows(v) for k, v in target_routes.items()}
    if set(draft_routes) != set(target_routes):
        raise ValueError('Matched routes require the same decoder layers')
    layers, values, exact = [], [], []
    for layer in sorted(draft_routes):
        draft, target = draft_routes[layer], target_routes[layer]
        if len(draft) != len(target):
            raise ValueError('Matched routes require the same token count in each layer')
        recalls = [len(set(a) & set(b)) / len(set(b)) for a, b in zip(draft, target)]
        equal = [set(a) == set(b) for a, b in zip(draft, target)]
        values.extend(recalls)
        exact.extend(equal)
        layers.append({'layer_id': layer, 'token_count': len(target),
                       'token_target_recall': recalls,
                       'mean_target_recall': statistics.mean(recalls) if recalls else None,
                       'exact_set_match_rate': statistics.mean(equal) if equal else None})
    return {'metadata': {
                'scope': 'matched_input_routes',
                'precondition': 'Identical input tokens, positions, and declared cache contexts supplied by caller',
                'denominator': 'Unique target experts for each token and layer',
                'aggregation': 'Arithmetic mean over token-layer pairs',
            }, 'layers': layers, 'token_layer_count': len(values),
            'mean_target_recall': statistics.mean(values) if values else None,
            'exact_set_match_rate': statistics.mean(exact) if exact else None}


def _target_ids(case, block, output):
    if case.name == 'dsv2lite':
        return output[0]
    import torch
    if case.name == 'qwen2moe':
        return torch.topk(torch.softmax(output, dim=1, dtype=torch.float), block.top_k, dim=-1).indices
    # The evaluation SparseMixer selects successive maxima, including ties.
    first = output.max(dim=-1, keepdim=True).indices
    second = output.scatter(-1, first, float('-inf')).max(dim=-1, keepdim=True).indices
    return torch.cat((first, second), dim=-1)


def _install_routes(case, model, role, callback):
    handles = []
    try:
        for layer_id, block in route_specs(case, model):
            if role == 'draft':
                def hook(module, args, output, layer=layer_id):
                    callback(layer, lambda: module._selected_experts)
                handles.append(block.register_forward_hook(hook))
            elif role == 'target':
                def hook(module, args, output, layer=layer_id, owner=block):
                    callback(layer, lambda: _target_ids(case, owner, output))
                handles.append(block.gate.register_forward_hook(hook))
            else:
                raise ValueError('Route role must be draft or target')
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles


class _Patches:
    def __init__(self):
        self.saved = []

    def set(self, obj, name, value):
        self.saved.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, name, existed, value in reversed(self.saved):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self.saved.clear()


class ForwardRouteCapture:
    """Capture standalone model forwards without moving IDs to the CPU in hooks."""
    def __init__(self, case, model, role='draft'):
        self.case, self.model, self.role = case, model, role
        self._handles, self._events = [], []
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError('Route capture is already active')
        self._events.clear()
        self._handles = _install_routes(self.case, self.model, self.role, self._capture)
        self._active = True
        return self

    def _capture(self, layer, getter):
        self._events.append((layer, getter().detach().clone()))

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False

    def result(self):
        if self._active:
            raise RuntimeError('Materialize routes after the capture context exits')
        expected = [layer for layer, _ in route_specs(self.case, self.model)]
        calls = []
        for index in range(0, len(self._events), len(expected)):
            events = self._events[index:index + len(expected)]
            if [layer for layer, _ in events] != expected:
                raise ValueError('Incomplete or reordered model route forward')
            calls.append({'layers': [{'layer_id': layer, 'ids': _rows(ids)} for layer, ids in events]})
        return {'metadata': {'role': self.role, 'layer_index': 'Zero-based decoder layer; dense layers omitted',
                             'shape': 'token x top-k; batch flattened in model order'}, 'calls': calls}


def _distribution(values):
    values = sorted(x for x in values if x is not None)
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'min': values[0] if values else None, 'max': values[-1] if values else None,
            'median': statistics.median(values) if values else None,
            'p90': values[max(0, math.ceil(0.9 * len(values)) - 1)] if values else None,
            'values': values}


class RouteCapture:
    """Observe completed speculation windows; exclude prefill and target tails."""
    def __init__(self, case, draft, target, controller):
        self.case, self.draft, self.target, self.controller = case, draft, target, controller
        self._handles, self._windows, self._current = [], [], None
        self._patches = _Patches()
        self._active = False

    def _stats(self):
        return dict(self.controller.expert_manager.get_async_loading_stats())

    def _capture(self, role, layer, getter):
        if self._current is not None:
            self._current[role].setdefault(layer, []).append(getter().detach().clone())

    def __enter__(self):
        if self._active or getattr(self.controller, '_benchmark_route_capture', False):
            raise RuntimeError('A route capture already owns this controller')
        self._windows.clear()
        self._active = True
        try:
            self._patches.set(self.controller, '_benchmark_route_capture', True)
            for role, model in (('draft', self.draft), ('target', self.target)):
                self._handles.extend(_install_routes(self.case, model, role,
                    lambda layer, getter, role=role: self._capture(role, layer, getter)))
            original_begin = self.controller.begin_window
            original_end = self.controller.end_window
            original_abort = self.controller.abort_window
            def begin(draft_len):
                if self._current is not None:
                    raise RuntimeError('A route window is already active')
                original_begin(draft_len)
                self._current = {'draft_len': draft_len, 'draft': {}, 'target': {},
                                 'start_stats': self._stats(), 'layer_io': {}, 'starts': {}}
            def finish(status):
                if self._current is not None:
                    self._current['status'] = status
                    self._current['end_stats'] = self._stats()
                    self._windows.append(self._current)
                    self._current = None
            def end():
                result = original_end()
                finish('complete')
                return result
            def abort():
                try:
                    return original_abort()
                finally:
                    finish('aborted')
            self._patches.set(self.controller, 'begin_window', begin)
            self._patches.set(self.controller, 'end_window', end)
            self._patches.set(self.controller, 'abort_window', abort)
            for layer, block in route_specs(self.case, self.target):
                def before(module, args, layer=layer):
                    if self._current is not None:
                        self._current['starts'][layer] = self._stats()
                def after(module, args, output, layer=layer):
                    if self._current is not None:
                        start = self._current['starts'].pop(layer)
                        end = self._stats()
                        values = self._current['layer_io'].setdefault(layer, Counter())
                        values.update({key: end.get(key, 0) - start.get(key, 0)
                                       for key in ('demand_requests', 'demand_hits', 'demand_misses')})
                self._handles.append(block.register_forward_pre_hook(before))
                self._handles.append(block.register_forward_hook(after))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        if self._current is not None:
            self._current['status'] = 'aborted'
            self._current['end_stats'] = self._stats()
            self._windows.append(self._current)
            self._current = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._patches.restore()
        self._active = False

    def result(self):
        if self._active:
            raise RuntimeError('Materialize routes after the capture context exits')
        windows, observations = [], []
        expected = [layer for layer, _ in route_specs(self.case, self.target)]
        for index, raw in enumerate(self._windows):
            if raw['status'] == 'complete' and any(set(raw[role]) != set(expected) for role in ('draft', 'target')):
                raise ValueError(f'Completed window {index} has missing or unexpected decoder layers')
            layers = []
            for layer in expected:
                draft = [row for tensor in raw['draft'].get(layer, []) for row in _rows(tensor)]
                target = [row for tensor in raw['target'].get(layer, []) for row in _rows(tensor)]
                if raw['status'] == 'complete' and (not draft or not target):
                    raise ValueError(f'Completed window {index} lacks routes at decoder layer {layer}')
                item = {'layer_id': layer, 'draft_ids': draft, 'target_ids': target,
                        'draft': route_statistics(draft, self.case.num_experts),
                        'target': route_statistics(target, self.case.num_experts),
                        **window_recall(draft, target),
                        'io': dict(raw['layer_io'].get(layer, {}))}
                layers.append(item)
                if raw['status'] == 'complete':
                    observations.append(item)
            keys = raw['start_stats'].keys() | raw['end_stats'].keys()
            windows.append({'window_id': index, 'draft_len': raw['draft_len'], 'status': raw['status'],
                            'layers': layers, 'io': {key: raw['end_stats'].get(key, 0) - raw['start_stats'].get(key, 0)
                                                    for key in sorted(keys)}})
        demand = {key: sum(row['io'].get(key, 0) for row in observations)
                  for key in ('demand_requests', 'demand_hits', 'demand_misses')}
        demand['hit_rate'] = demand['demand_hits'] / demand['demand_requests'] if demand['demand_requests'] else None
        distributions = {role: {key: _distribution(row[role][key] for row in observations)
                                for key in ('unique_count', 'working_set_fraction', 'distinct_token_sets',
                                            'entropy_nats', 'effective_experts', 'normalized_entropy',
                                            'adjacent_token_jaccard_mean')}
                         for role in ('draft', 'target')}
        return {'metadata': {
                    'scope': 'Completed speculative windows; excludes prefill, standalone target tails, and aborted windows',
                    'instrumentation': 'Untimed diagnostic pass; detached cloned device IDs materialized after exit',
                    'target_route_source': 'DeepSeek gate IDs; Qwen float32 softmax top-k; Phi successive maxima',
                    'layer_index': 'Zero-based decoder layer; dense layers omitted',
                    'route_shape': 'token x top-k; batch flattened in model order',
                    'recall_denominator': 'Unique target experts in one decoder layer and verification window',
                    'recall_aggregation': 'Arithmetic mean over completed layer-window pairs',
                    'target_scope': 'All verification inputs, including the final proposal used for bonus logits',
                    'fidelity': 'Window recall is not matched-token route fidelity',
                    'demand_denominator': 'Unique expert requests per target layer invocation, summed over completed windows',
                    'demand_hits': 'Resident at demand-plan creation; in-flight resident transfers count as hits',
                    'entropy_denominator': 'Expert assignments; normalized entropy divides by log(total routed experts)',
                    'distribution_aggregation': 'One equally weighted sample per completed layer-window pair; p90 nearest rank',
                }, 'windows': windows, 'summary': {
                    'complete_windows': sum(w['status'] == 'complete' for w in windows),
                    'aborted_windows': sum(w['status'] == 'aborted' for w in windows),
                    'layer_window_count': len(observations),
                    'target_recall': _distribution(row['target_recall'] for row in observations),
                    'demand': demand, 'route_distributions': distributions}}
