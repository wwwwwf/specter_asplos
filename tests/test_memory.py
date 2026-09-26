"""Storage accounting checks with CPU tensors and real storage aliases."""
from collections import deque
import gc
import json
from types import SimpleNamespace
import unittest
import weakref

try:
    import torch
except ImportError:
    torch = None

from benchmarks.memory import MemorySnapshots, storage_report


@unittest.skipIf(torch is None, 'requires Torch CPU tensors')
class MemoryTests(unittest.TestCase):
    def test_parameters_views_plain_scratch_and_raw_expert_pools_are_deduplicated(self):
        shared = torch.nn.Parameter(torch.arange(16, dtype=torch.float32))
        draft = torch.nn.Module()
        target = torch.nn.Module()
        draft.weight = shared
        target.weight = shared
        draft.register_buffer('tied_view', shared.detach()[4:8])
        draft.scratch = torch.empty(10, dtype=torch.float16)
        target.register_buffer('target_state', torch.empty(5, dtype=torch.float32))
        pool = torch.UntypedStorage(128)
        target.expert_cache = SimpleNamespace(
            main_modules=[SimpleNamespace(storage=pool)],
            device_expert_buffers=deque([SimpleNamespace(storage=pool)]))
        target.expert_cache.cycle = target.expert_cache
        report = storage_report(SimpleNamespace(model=draft), target)
        self.assertEqual(report['total_bytes'], 64 + 20 + 20 + 128)
        self.assertEqual(report['models']['shared_bytes'], 64)
        self.assertEqual(report['models']['shared_parameter_bytes'], 64)
        self.assertEqual(report['models']['parameter_bytes'], {'draft': 64, 'target': 64})
        self.assertEqual(report['owners']['draft']['bytes'], 84)
        self.assertEqual(report['owners']['target']['bytes'], 212)
        self.assertTrue(report['complete'])
        self.assertEqual(report['by_device'], {'cpu': 232})
        paths = [ref['path'] for storage in report['storages'] for ref in storage['references']]
        self.assertTrue(any('device_expert_buffers' in path for path in paths))
        json.dumps(report)

    def test_raw_storage_subranges_count_overlapping_bytes_once(self):
        storage = torch.UntypedStorage(128)
        draft = SimpleNamespace(raw=storage[16:48])
        target = SimpleNamespace(raw=storage)
        report = storage_report(draft, target)
        self.assertEqual(report['total_bytes'], 128)
        self.assertEqual(report['owners']['draft']['bytes'], 32)
        self.assertEqual(report['owners']['target']['bytes'], 128)
        self.assertEqual(report['owners']['target']['storage_count'], 1)
        self.assertEqual(report['owners']['target']['region_count'], 3)
        self.assertEqual(report['models']['shared_bytes'], 32)
        self.assertEqual(report['models']['target_only_bytes'], 96)
        self.assertEqual(sum(item['storage_bytes'] for item in report['storages']), 128)

    def test_shared_prefix_and_reserved_scratch_use_capacity_not_visible_size(self):
        prefix_k = torch.zeros(1, 2, 8, 4, dtype=torch.float16)
        prefix_v = torch.zeros_like(prefix_k)
        scratch = torch.empty(1, 2, 16, 4, dtype=torch.float16)
        target = SimpleNamespace(key_cache=[prefix_k], value_cache=[prefix_v])
        draft = SimpleNamespace(key_cache=[prefix_k[..., :6, :]], value_cache=[prefix_v],
                                scratch_keys=[scratch[..., :2, :]], scratch_values=[],
                                _scratch_key_buffers=[scratch])
        report = storage_report(None, None, {'draft': draft, 'target': target})
        self.assertEqual(report['cache']['unique_bytes'], 128 + 128 + 256)
        self.assertEqual(report['cache']['shared_bytes'], 256)
        self.assertEqual(report['owners']['cache:draft']['bytes'], 512)
        self.assertEqual(report['owners']['cache:target']['bytes'], 256)
        self.assertEqual(report['cache']['additional_bytes'], 512)
        # Supplying a model-owned tensor as a cache does not double count it.
        model = torch.nn.Module()
        model.register_buffer('prefix', prefix_k)
        report = storage_report(model, None, {'target': target})
        self.assertEqual(report['total_bytes'], 256)
        self.assertEqual(report['cache']['shared_with_models_bytes'], 128)
        self.assertEqual(report['cache']['additional_bytes'], 128)

    def test_independent_prefixes_do_not_report_shared_storage(self):
        original = torch.ones(8, dtype=torch.float16)
        clone = original.clone()
        report = storage_report(None, None, {'draft': [clone], 'target': [original]})
        self.assertEqual(report['total_bytes'], 32)
        self.assertEqual(report['cache']['shared_bytes'], 0)

    def test_snapshots_keep_only_json_metadata_and_release_old_cache(self):
        snapshots = MemorySnapshots(None, None)
        cache = torch.empty(8)
        reference = weakref.ref(cache)
        metadata = {'tokens': [8]}
        snapshots.capture('prefill', {'target': cache}, metadata)
        metadata['tokens'].append(9)
        del cache
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(snapshots.snapshots[0]['metadata'], {'tokens': [8]})
        json.dumps(snapshots.snapshots)
        with self.assertRaises(TypeError):
            snapshots.capture('bad', metadata={'tensor': torch.empty(1)})

    def test_unsupported_tensor_is_reported_as_incomplete(self):
        model = torch.nn.Module()
        model.register_buffer('unmaterialized', torch.empty(4, device='meta'))
        report = storage_report(model, None)
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['skipped']), 1)
        self.assertEqual(report['total_bytes'], 0)
        with self.assertRaises(TypeError):
            storage_report(None, None, [torch.empty(1)])


if __name__ == '__main__':
    unittest.main()
