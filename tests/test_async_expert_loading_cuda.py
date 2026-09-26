#!/usr/bin/env python3
"""CUDA stress tests for Specter's expert-slot event protocol."""

from __future__ import annotations

import sys
import threading
import unittest
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictive_io_orchestrator.expert_cache import (  # noqa: E402
    EvictionGroupInfo,
    ExpertCache,
    ExpertInfo,
)


class CudaAsyncExpertCache(ExpertCache):
    def __init__(
        self,
        *,
        capacity: int,
        total_experts: int,
        staging_buffers: int,
        elements: int = 1 << 18,
    ) -> None:
        self.device = torch.device("cuda")
        self.module_size = elements * torch.empty(
            (), dtype=torch.int32
        ).element_size()
        self.active = False
        self.stream_tx = torch.cuda.Stream(device=self.device)
        self.main_modules = [
            SimpleNamespace(
                storage=torch.empty(
                    elements,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            for _ in range(capacity)
        ]
        self.device_expert_buffers = deque(
            SimpleNamespace(
                storage=torch.empty(
                    elements,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            for _ in range(staging_buffers)
        )
        self.offloaded_storages = []
        for expert_id in range(total_experts):
            storage = torch.empty(
                elements,
                dtype=torch.int32,
                pin_memory=True,
            )
            storage.fill_(expert_id)
            self.offloaded_storages.append(storage)

        self.main_infos = [None] * capacity
        self.offloaded_infos = [None] * total_experts
        self.registered_experts = {}
        self.group_infos = defaultdict(EvictionGroupInfo)
        group = self.group_infos[0]
        for expert_id in range(total_experts):
            resident = expert_id < capacity
            info = ExpertInfo(
                uid=(0, expert_id),
                eviction_group=0,
                offloaded=not resident,
                cache_index=expert_id if resident else -1,
                mem_index=expert_id,
            )
            self.registered_experts[info.uid] = info
            self.offloaded_infos[expert_id] = info
            group.offloaded_infos[info.uid] = info
            if resident:
                self.main_modules[expert_id].storage.copy_(
                    self.offloaded_storages[expert_id],
                    non_blocking=True,
                )
                self.main_infos[expert_id] = info
                group.main_infos[info.uid] = info

        self.specter_async_loading_enabled = False
        self._async_slot_ready_events = []
        self._async_slot_last_use_events = []
        self._async_slot_reuse_events = []
        self._async_slot_generations = []
        self._async_staging_capacity = 0
        self._async_staging_available = deque()
        self._async_staging_ready_events = {}
        self._async_staging_last_use_events = {}
        self._async_staging_reuse_events = {}
        self._async_staging_generations = {}
        self._async_state_lock = threading.RLock()
        self._async_demand_active = False
        self.async_loading_stats = defaultdict(int)
        self.enable_specter_async_loading()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CudaAsyncExpertLoadingTest(unittest.TestCase):
    def exercise(self, staging_buffers: int) -> None:
        capacity = 4
        total_experts = 12
        cache = CudaAsyncExpertCache(
            capacity=capacity,
            total_experts=total_experts,
            staging_buffers=staging_buffers,
        )
        checks = []
        expected = []

        for step in range(40):
            prefetch_ids = [
                (step + offset) % total_experts
                for offset in range(capacity)
            ]
            cache.prefetch_experts_async(
                *((0, expert_id) for expert_id in prefetch_ids),
                unordered=True,
            )
            demand_ids = [
                (step * 3 + offset) % total_experts
                for offset in range(capacity + 3)
            ]
            for uid, expert in cache.load_experts_async(
                *((0, expert_id) for expert_id in demand_ids),
                unordered=True,
            ):
                # Both operations must complete before this slot can be reused.
                checks.append(
                    (
                        expert.storage[0].clone(),
                        expert.storage.to(torch.int64).sum(),
                    )
                )
                expected.append(uid[1])

        torch.cuda.synchronize()
        elements = cache.main_modules[0].storage.numel()
        for (first, checksum), expert_id in zip(checks, expected):
            self.assertEqual(int(first.item()), expert_id)
            self.assertEqual(int(checksum.item()), expert_id * elements)
        self.assertFalse(cache._async_demand_active)
        self.assertEqual(
            len(cache._async_staging_available), staging_buffers
        )

    def test_persistent_slot_rolling_has_no_read_write_conflicts(self) -> None:
        self.exercise(staging_buffers=0)

    def test_staging_rolling_has_no_read_write_conflicts(self) -> None:
        self.exercise(staging_buffers=2)

    def test_mode_switch_handoff_orders_legacy_slot_writes(self) -> None:
        cache = CudaAsyncExpertCache(
            capacity=2,
            total_experts=4,
            staging_buffers=0,
        )
        slot = cache.main_modules[0].storage
        before = slot.to(torch.int64).sum()

        cache.disable_specter_async_loading()
        with torch.cuda.stream(cache.stream_tx):
            slot.fill_(99)
        cache.enable_specter_async_loading()
        after = slot.to(torch.int64).sum()

        torch.cuda.synchronize()
        elements = slot.numel()
        self.assertEqual(int(before.item()), 0)
        self.assertEqual(int(after.item()), 99 * elements)

    def test_bounded_prefetch_protects_complete_resident_set(self) -> None:
        cache = CudaAsyncExpertCache(
            capacity=4,
            total_experts=8,
            staging_buffers=0,
        )
        desired = ((0, 0), (0, 1), (0, 4), (0, 5))
        handles = cache.prefetch_experts_bounded_async(
            *desired,
            max_new_loads=1,
            unordered=True,
        )
        torch.cuda.synchronize()

        self.assertFalse(cache.registered_experts[(0, 0)].offloaded)
        self.assertFalse(cache.registered_experts[(0, 1)].offloaded)
        self.assertFalse(cache.registered_experts[(0, 4)].offloaded)
        self.assertTrue(cache.registered_experts[(0, 5)].offloaded)
        self.assertEqual(len(handles), 3)
        self.assertEqual(cache.async_loading_stats["prefetch_misses"], 1)
        self.assertEqual(
            cache.async_loading_stats["prefetch_deferred_misses"],
            1,
        )



if __name__ == "__main__":
    unittest.main()
