from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from benchmarks.cache_reset import ResidentReset


class CacheResetTests(unittest.TestCase):
    def test_restores_resident_members_order_and_counters(self):
        infos = {i: object() for i in range(4)}
        group = SimpleNamespace(main_infos=OrderedDict((i, infos[i]) for i in (3, 1)), hits=0, misses=0)
        manager = SimpleNamespace(group_infos={0: group}, registered_experts=infos,
                                  enable_specter_async_loading=lambda: None)
        def prefetch(*uids, unordered):
            self.assertFalse(unordered)
            group.main_infos = OrderedDict((i, infos[i]) for i in reversed(uids))
        manager.prefetch_experts_async = prefetch
        reset = ResidentReset(manager)
        group.main_infos = OrderedDict((i, infos[i]) for i in (0, 2))
        group.hits, group.misses = 7, 9
        with patch('torch.cuda.synchronize'):
            reset.restore()
        self.assertEqual(list(group.main_infos), [3, 1])
        self.assertEqual((group.hits, group.misses), (0, 0))
