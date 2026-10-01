"""CPU state-machine tests; fake streams never initialize CUDA."""
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .cache import ExpertCache, ExpertInfo, EvictionGroupInfo


class Storage:
    def __init__(self, value=None):
        self.value = value

    def copy_(self, other, **kwargs):
        self.value = other.value


class Event:
    def __init__(self, **kwargs):
        self.recorded = False

    def record(self, stream):
        self.recorded = True


class Stream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        if not event.recorded:
            raise AssertionError('Waited on an unrecorded event')
        self.waits.append(event)


def make_cache(residents=2, groups=2, experts=6):
    cache = ExpertCache.__new__(ExpertCache)
    cache._make_module = lambda: SimpleNamespace(storage=Storage())
    cache.main_modules = [cache._make_module() for _ in range(groups * residents)]
    cache.main_infos = [None] * len(cache.main_modules)
    cache._slot_ready = [None] * len(cache.main_modules)
    cache._slot_last_use = [None] * len(cache.main_modules)
    cache.module_size = 10
    cache.device = 'fake-device'
    cache.device_expert_buffers = deque()
    cache.offloaded_storages = []
    cache.offloaded_infos = []
    cache.registered_experts = {}
    cache.group_infos = defaultdict(EvictionGroupInfo)
    cache.stream_tx = Stream()
    cache.active = False
    cache.stats = defaultdict(int)
    cache._initial = None
    for layer in range(groups):
        for expert in range(experts):
            uid = (layer, expert)
            mem = len(cache.offloaded_storages)
            cache.offloaded_storages.append(Storage(uid))
            info = ExpertInfo(uid, layer, mem_index=mem)
            cache.registered_experts[uid] = info
            cache.offloaded_infos.append(info)
            group = cache.group_infos[layer]
            if expert >= experts - residents:
                slot = layer * residents + expert - experts + residents
                info.offloaded, info.cache_index = False, slot
                cache.main_infos[slot] = info
                cache.main_modules[slot].storage.value = uid
                group.main_infos[uid] = info
            else:
                group.offloaded_infos[uid] = info
    return cache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.stream = Stream()
        self.fake_cuda = SimpleNamespace(Event=Event, current_stream=lambda *args: self.stream,
                                         stream=lambda stream: nullcontext(), synchronize=lambda *args: None)
        self.patch = patch('baselines.specmoeoff.cache.torch.cuda', self.fake_cuda)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_over_capacity_demand_preserves_every_expert_until_consumed(self):
        for capacity in (1, 2, 3):
            cache = make_cache(residents=capacity)
            uids = [(0, i) for i in range(6)]
            observed = [(uid, module.storage.value) for uid, module in cache.load_experts(*uids, unordered=True)]
            self.assertEqual(set(uid for uid, _ in observed), set(uids))
            self.assertTrue(all(uid == value for uid, value in observed), observed)
            self.assertFalse(cache.active)
            self.assertTrue(any(event is not None for event in cache._slot_last_use))

    def test_prefetch_protects_requested_hits_and_restore_recovers_lru(self):
        cache = make_cache()
        cache.snapshot_initial()
        initial = dict(cache._initial)
        for _ in cache.prefetch_experts((0, 0), (0, 5)):
            pass
        self.assertEqual(set(cache.group_infos[0].main_infos), {(0, 0), (0, 5)})
        for uid, module in cache.load_experts(*[(0, i) for i in range(6)], unordered=True):
            self.assertEqual(uid, module.storage.value)
        cache.restore()
        self.assertEqual(initial, cache._initial)
        for key, (resident, offloaded) in initial.items():
            self.assertEqual(tuple(cache.group_infos[key].main_infos), resident)
            self.assertEqual(tuple(cache.group_infos[key].offloaded_infos), offloaded)

    def test_resize_rebuilds_correct_weights_without_checkpoint_reload(self):
        cache = make_cache(residents=1)
        for capacity in (3, 2, 1, 6):
            cache.resize_residency(capacity)
            self.assertEqual(len(cache.main_modules), capacity * 2)
            for key, group in cache.group_infos.items():
                self.assertEqual(len(group.main_infos), capacity)
                for uid, info in group.main_infos.items():
                    self.assertEqual(cache.main_modules[info.cache_index].storage.value, uid)
            cache.restore()

    def test_closed_consumer_releases_demand_and_records_slot_use(self):
        cache = make_cache(residents=1)
        iterator = cache.load_experts((0, 0), (0, 1), unordered=True)
        uid, module = next(iterator)
        self.assertEqual(uid, module.storage.value)
        iterator.close()
        self.assertFalse(cache.active)
        self.assertIsNotNone(cache._slot_last_use[0])


if __name__ == '__main__':
    unittest.main()
