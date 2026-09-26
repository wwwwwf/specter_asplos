"""Account for live Torch storage reachable from models and explicit KV caches.

This is retained storage, including expert offload pools and scratch capacity.
It is not CUDA allocator occupancy, peak memory, or a per-step saving estimate.
Torch is imported only when a report is requested.
"""
from collections import deque
from collections.abc import Mapping
import json
import types


def storage_report(draft, target, caches=None):
    """Return a JSON-safe snapshot, counting aliased backing storage once.

    ``draft`` may be a model or its existing ``.model`` wrapper. ``caches`` is
    an optional mapping, usually ``{'draft': branch, 'target': committed}``.
    Ordinary object attributes are visited without evaluating properties, so
    unregistered expert pools, raw UntypedStorage, and KV scratch are included.
    """
    import torch

    stores = {}
    owners = {}
    parameter_keys = {'draft': set(), 'target': set()}
    skipped = []
    skipped_types = (type, types.ModuleType, types.FunctionType,
                     types.BuiltinFunctionType, types.MethodType)

    def record(value, owner, path, kind):
        tensor = isinstance(value, torch.Tensor)
        if tensor:
            if value.layout != torch.strided or value.device.type == 'meta':
                skipped.append({'owner': owner, 'path': path,
                                'reason': 'non-strided or meta tensor'})
                return
            storage = value.untyped_storage()
        else:
            storage = value.untyped() if hasattr(value, 'untyped') else value
            if kind == 'plain_tensor':
                kind = 'raw_storage'
        size = int(storage.nbytes())
        device = str(storage.device)
        # Nonempty aliased storages have the same base pointer and capacity.
        # Distinct zero-size allocations all have pointer zero and cost zero.
        key = (device, int(storage.data_ptr()), size)
        if key not in stores:
            stores[key] = {'id': len(stores), 'device': device,
                           'storage_bytes': size, 'references': []}
        reference = {'owner': owner, 'path': path, 'kind': kind,
                     'referenced_storage_bytes': size}
        if tensor:
            reference.update(dtype=str(value.dtype), shape=list(value.shape),
                             logical_bytes=int(value.numel() * value.element_size()),
                             storage_offset=int(value.storage_offset()))
        stores[key]['references'].append(reference)
        owners[owner].add(key)
        if kind == 'parameter' and owner in parameter_keys:
            parameter_keys[owner].add(key)

    def walk(value, owner, path, seen, kind='plain_tensor'):
        if isinstance(value, torch.Tensor) or torch.is_storage(value):
            record(value, owner, path, kind)
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool) + skipped_types):
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, Mapping):
            for name, child in value.items():
                walk(child, owner, f'{path}[{name!r}]', seen, kind)
        elif isinstance(value, (list, tuple, deque)):
            for index, child in enumerate(value):
                walk(child, owner, f'{path}[{index}]', seen, kind)
        elif isinstance(value, torch.nn.Module):
            # Read the registries directly: named_parameters() removes aliases.
            for name, child in value._parameters.items():
                if child is not None:
                    record(child, owner, f'{path}.{name}', 'parameter')
            for name, child in value._buffers.items():
                if child is not None:
                    record(child, owner, f'{path}.{name}', 'buffer')
            for name, child in value._modules.items():
                walk(child, owner, f'{path}.{name}', seen)
            for name, child in vars(value).items():
                if name in ('_parameters', '_buffers', '_modules') or 'hook' in name:
                    continue
                walk(child, owner, f'{path}.{name}', seen)
        elif hasattr(value, '__dict__'):
            for name, child in vars(value).items():
                if 'hook' not in name:
                    walk(child, owner, f'{path}.{name}', seen, kind)

    for name, model in (('draft', draft), ('target', target)):
        owners[name] = set()
        if model is not None and not isinstance(model, torch.nn.Module):
            model = getattr(model, 'model', model)
        walk(model, name, name, set())
    if caches is not None and not isinstance(caches, Mapping):
        raise TypeError('caches must be a mapping of labels to cache objects')
    for name, cache in (caches or {}).items():
        owner = f'cache:{name}'
        owners[owner] = set()
        walk(cache, owner, owner, set(), 'cache')

    # ExpertWrapper may expose a whole UntypedStorage and separately backed
    # slices of it. Pointer equality alone cannot deduplicate those overlaps.
    # Split address ranges into disjoint regions and retain their provenance.
    observed_storage_count = len(stores)
    owner_storage_counts = {owner: len(keys) for owner, keys in owners.items()}
    events = {}
    for key in stores:
        device, pointer, size = key
        if size:
            points = events.setdefault(device, {})
            points.setdefault(pointer, [set(), set()])[0].add(key)
            points.setdefault(pointer + size, [set(), set()])[1].add(key)
    regions = {}
    owners = {owner: set() for owner in owners}
    parameter_keys = {'draft': set(), 'target': set()}
    for device, points in sorted(events.items()):
        active = set()
        previous = None
        for address in sorted(points):
            if active and previous is not None and address > previous:
                index = len(regions)
                references = [reference for key in sorted(active)
                              for reference in stores[key]['references']]
                regions[index] = {'id': index, 'device': device,
                                  'storage_bytes': address - previous,
                                  'references': references}
                for reference in references:
                    owner = reference['owner']
                    owners[owner].add(index)
                    if reference['kind'] == 'parameter' and owner in parameter_keys:
                        parameter_keys[owner].add(index)
            added, removed = points[address]
            active.difference_update(removed)
            active.update(added)
            previous = address
    stores = regions

    def total(keys):
        return sum(stores[key]['storage_bytes'] for key in keys)

    def by_device(keys):
        result = {}
        for key in keys:
            device = stores[key]['device']
            result[device] = result.get(device, 0) + stores[key]['storage_bytes']
        return dict(sorted(result.items()))

    cache_keys = set().union(*(keys for owner, keys in owners.items()
                              if owner.startswith('cache:')))
    model_keys = owners['draft'] | owners['target']
    cache_counts = {}
    for owner, keys in owners.items():
        if owner.startswith('cache:'):
            for key in keys:
                cache_counts[key] = cache_counts.get(key, 0) + 1
    shared_cache = {key for key, count in cache_counts.items() if count > 1}
    shared_model = owners['draft'] & owners['target']
    shared_parameters = parameter_keys['draft'] & parameter_keys['target']
    return {
        'scope': 'live Torch storage reachable from models and explicit caches',
        'measurement': 'retained storage capacity; excludes allocator reserve and unreachable temporaries',
        'complete': not skipped,
        'skipped': skipped,
        'total_bytes': total(stores),
        'by_device': by_device(stores),
        'storage_count': observed_storage_count,
        'region_count': len(stores),
        'owners': {owner: {'bytes': total(keys), 'by_device': by_device(keys),
                           'storage_count': owner_storage_counts[owner],
                           'region_count': len(keys)} for owner, keys in owners.items()},
        'models': {
            'unique_bytes': total(model_keys),
            'shared_bytes': total(shared_model),
            'shared_by_device': by_device(shared_model),
            'draft_only_bytes': total(owners['draft'] - owners['target']),
            'target_only_bytes': total(owners['target'] - owners['draft']),
            'parameter_bytes': {name: total(keys) for name, keys in parameter_keys.items()},
            'shared_parameter_bytes': total(shared_parameters),
        },
        'cache': {
            'unique_bytes': total(cache_keys),
            'by_device': by_device(cache_keys),
            'shared_bytes': total(shared_cache),
            'shared_by_device': by_device(shared_cache),
            'shared_with_models_bytes': total(cache_keys & model_keys),
            'additional_bytes': total(cache_keys - model_keys),
        },
        'storages': list(stores.values()),
    }


class MemorySnapshots:
    """Capture reports at state boundaries without retaining old KV tensors."""

    def __init__(self, draft, target):
        self.draft = draft
        self.target = target
        self.snapshots = []

    def capture(self, label, caches=None, metadata=None):
        if not isinstance(label, str) or not label:
            raise ValueError('snapshot label must be nonempty')
        report = storage_report(self.draft, self.target, caches)
        snapshot = {'label': label, 'metadata': json.loads(json.dumps(metadata or {})),
                    'report': report}
        self.snapshots.append(snapshot)
        return snapshot
