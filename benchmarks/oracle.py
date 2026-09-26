"""Replay recorded target routes as controlled prefetch predictions."""
from collections import Counter
import math
import random


def build_oracle_plans(windows, num_experts, resident, seed, error_rate, expected_layers=None):
    if not math.isfinite(error_rate) or not 0 <= error_rate <= 1:
        raise ValueError('error_rate must be in [0, 1]')
    if not 0 < resident <= num_experts:
        raise ValueError('Invalid resident capacity')
    if not windows:
        raise ValueError('Oracle replay requires completed target-route windows')
    from benchmarks.routing import _rows
    expected = set(expected_layers) if expected_layers is not None else None
    rng = random.Random(seed)
    result = []
    for window in windows:
        if window.get('status', 'complete') != 'complete':
            raise ValueError('An aborted window cannot be an oracle source')
        layer_ids = [layer['layer_id'] for layer in window['layers']]
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError('Oracle windows require unique decoder layers')
        if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layer_ids):
            raise ValueError('Invalid oracle decoder layer')
        if expected is None:
            expected = set(layer_ids)
        if set(layer_ids) != expected:
            raise ValueError('Oracle window layers differ from the expected model')
        layers = []
        for layer in window['layers']:
            rows = _rows(layer['target_ids'])
            if not rows:
                raise ValueError('Oracle target routes must be nonempty')
            counts = Counter(expert for row in rows for expert in row)
            if any(expert < 0 or expert >= num_experts for expert in counts):
                raise ValueError('Oracle expert ID outside model range')
            # Perturb distinct identities, preserving their activation weights.
            identities = sorted(counts)
            positions = set(rng.sample(range(len(identities)), round(error_rate * len(identities))))
            predicted = Counter()
            replacements = []
            for index, expert in enumerate(identities):
                replacement = rng.randrange(num_experts) if index in positions else expert
                predicted[replacement] += counts[expert]
                if index in positions:
                    replacements.append([expert, replacement])
            selected = [expert for expert, _ in sorted(predicted.items(), key=lambda item: (-item[1], item[0]))[:resident]]
            correct = len(set(selected).intersection(counts))
            layers.append({'layer_id': layer['layer_id'], 'selected': selected,
                           'target_set': identities, 'correct': correct, 'predicted': len(selected),
                           'accuracy': correct / len(selected) if selected else None,
                           'replacements': replacements})
        result.append({'layers': layers, 'draft_len': window.get('draft_len')})
    return result


class OraclePrefetch:
    """Use offline oracle plans while retaining the current scheduling policy."""
    def __init__(self, controller, plans):
        self.controller, self.plans = controller, plans
        self.index = -1
        self.used = []
        self._saved = {}

    def __enter__(self):
        if self._saved:
            raise RuntimeError('Oracle context is already active')
        self.index = -1
        self.used = []
        controller = self.controller
        for name in ('begin_window', 'get_all_layer_topm_uids'):
            self._saved[name] = (name in controller.__dict__, controller.__dict__.get(name), getattr(controller, name))
        original_begin = self._saved['begin_window'][2]

        def begin(draft_len):
            self.index += 1
            if self.index >= len(self.plans):
                raise RuntimeError('Decoding exceeded the recorded oracle windows')
            recorded_length = self.plans[self.index].get('draft_len')
            if recorded_length is not None and recorded_length != draft_len:
                raise RuntimeError('Draft length differs from the recorded oracle window')
            original_begin(draft_len)

        def get_plans(count):
            if self.index < 0 or self.index >= len(self.plans):
                raise RuntimeError('No active oracle window')
            result = [[] for _ in range(controller.layer_num)]
            for layer in self.plans[self.index]['layers']:
                layer_id = layer['layer_id']
                index = layer_id - int(controller.skip_fisrt_layer)
                if not 0 <= index < len(result):
                    raise ValueError('Oracle layer does not match this model')
                result[index] = [(layer_id, expert) for expert in layer['selected'][:count]]
            self.used.append(self.index)
            return result

        controller.begin_window = begin
        controller.get_all_layer_topm_uids = get_plans
        return self

    def __exit__(self, *exc):
        for name, (owned, value, _) in self._saved.items():
            if owned:
                setattr(self.controller, name, value)
            else:
                delattr(self.controller, name)
        self._saved.clear()

    def result(self):
        values = [layer['accuracy'] for index in sorted(set(self.used))
                  for layer in self.plans[index]['layers'] if layer['accuracy'] is not None]
        return {'used_windows': sorted(set(self.used)), 'planned_windows': len(self.plans),
                'visited_windows': self.index + 1, 'plans': self.plans,
                'prefetch_accuracy': sum(values) / len(values) if values else None,
                'accuracy_definition': 'Mean distinct predicted expert precision over layers and used windows',
                'replacement_rule': 'Uniform same-layer replacement; a replacement may still be correct'}
