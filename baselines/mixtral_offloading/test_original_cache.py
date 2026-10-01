"""CPU semantics and source-integrity checks for the unmodified upstream cache."""
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import torch

from .original_cache import cache_stats, reset_cache, snapshot_initial
from .upstream.src.expert_cache import ExpertCache


class ScalarExpert:
    def __init__(self, value=0):
        self.tensor = torch.tensor([value], dtype=torch.int64)
        self.storage = self.tensor.untyped_storage()

    def __call__(self):
        return self.tensor.item()


def make_cache(resident, offloaded, buffers=4, groups=1, spare_main=0, spare_offloaded=0):
    # Only the test allocator skips host pinning, since these scalar experts
    # live on CPU. All original cache methods execute without replacement.
    with patch.object(torch.UntypedStorage, "pin_memory", lambda storage, device=None: storage):
        cache = ExpertCache(ScalarExpert, resident * groups + spare_main,
                            offloaded * groups + spare_offloaded, buffers)
    for layer in range(groups):
        for index in range(resident + offloaded):
            cache.add_expert((layer, index), ScalarExpert(100 * layer + index),
                             eviction_group=layer, offload=index < offloaded)
    return cache


class OriginalCacheTests(unittest.TestCase):
    def test_vendor_bytes_match_pinned_git_manifest(self):
        root = Path(__file__).parent / "upstream"
        manifest = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["commit"], "ce545188b804238f0b23a59fc45e6a8f8b390c40")
        for entry in manifest["files"]:
            data = (root / entry["path"]).read_bytes()
            self.assertEqual(len(data), entry["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])

    def assert_weights(self, cache):
        for uid, info in cache.registered_experts.items():
            storage = (cache.offloaded_storages[info.index] if info.offloaded
                       else cache.main_modules[info.index].storage)
            self.assertEqual(torch.as_tensor(storage, dtype=torch.int64).item(), 100 * uid[0] + uid[1])

    def test_original_resident_first_pipeline_consumes_more_than_capacity(self):
        rng = random.Random(42)
        for resident in [1, 2, 8, 32]:
            cache = make_cache(resident, 64 - resident, buffers=4, groups=2)
            for step in range(20):
                layer = step % 2
                ids = rng.sample(range(64), rng.randint(1, 64))
                seen = []
                for uid, expert in cache.load_experts(*[(layer, i) for i in ids], unordered=True):
                    seen.append(uid[1])
                    self.assertEqual(expert(), layer * 100 + uid[1])
                self.assertCountEqual(seen, ids)
                self.assert_weights(cache)

    def test_external_reset_restores_original_membership_slots_and_lru(self):
        cache = make_cache(32, 32, buffers=4, groups=2)
        initial = snapshot_initial(cache)
        def run():
            result = []
            for layer in [0, 1, 0]:
                for uid, expert in cache.load_experts(*[(layer, i) for i in range(64)], unordered=True):
                    result.append((uid, expert()))
            return result, cache_stats(cache)
        expected = run()
        reset_cache(cache, initial)
        self.assertEqual(snapshot_initial(cache), initial)
        self.assertEqual(cache_stats(cache), {"cache_hits": 0, "cache_misses": 0})
        self.assert_weights(cache)
        self.assertEqual(run(), expected)
        self.assertEqual(len(cache.device_expert_buffers), 4)

    def test_unused_trailing_capacity_survives_demand_and_reset(self):
        # Historical DeepSeek reserves 27 layers but registers only 26 MoE
        # layers. Exercise the same trailing None slots with a small CPU cache.
        cache = make_cache(3, 5, buffers=4, groups=2, spare_main=2, spare_offloaded=3)
        from .backend import cache_metadata
        metadata = cache_metadata(cache)
        for name, expected in {'resident_per_layer': 3, 'cache_group_count': 2,
                               'resident_expert_slots': 6, 'offloaded_expert_slots': 10,
                               'allocated_gpu_expert_slots': 8, 'allocated_host_expert_slots': 13,
                               'unused_gpu_expert_slots': 2, 'unused_host_expert_slots': 3,
                               'cache_gpu_storage_bytes': 12 * cache.module_size}.items():
            self.assertEqual(metadata[name], expected, name)
        self.assertEqual(len(cache.main_modules), 8)
        self.assertEqual(len(cache.offloaded_storages), 13)
        self.assertEqual(cache.main_infos[-2:], [None, None])
        self.assertEqual(cache.offloaded_infos[-3:], [None, None, None])
        spare_modules = tuple(cache.main_modules[-2:])
        spare_storages = tuple(cache.offloaded_storages[-3:])
        initial = snapshot_initial(cache)
        self.assertEqual(initial['main'][-2:], (None, None))
        self.assertEqual(initial['offloaded'][-3:], (None, None, None))

        def consume():
            outputs = []
            for layer in [0, 1, 0]:
                for uid, expert in cache.load_experts(*[(layer, i) for i in range(8)], unordered=True):
                    outputs.append((uid, expert()))
                    self.assertEqual(expert(), layer * 100 + uid[1])
            return outputs, cache_stats(cache)

        expected = consume()
        self.assert_weights(cache)
        reset_cache(cache, initial)
        self.assertEqual(snapshot_initial(cache), initial)
        self.assertEqual(cache_stats(cache), {'cache_hits': 0, 'cache_misses': 0})
        self.assert_weights(cache)
        self.assertEqual(consume(), expected)
        self.assertEqual(cache.main_infos[-2:], [None, None])
        self.assertEqual(cache.offloaded_infos[-3:], [None, None, None])
        self.assertTrue(all(a is b for a, b in zip(spare_modules, cache.main_modules[-2:])))
        self.assertTrue(all(a is b for a, b in zip(spare_storages, cache.offloaded_storages[-3:])))
        self.assertEqual(cache_metadata(cache), metadata)


if __name__ == "__main__":
    unittest.main()
