#!/usr/bin/env python3
"""CPU state-machine tests for Specter's asynchronous expert loader."""

from __future__ import annotations

import sys
import threading
import unittest
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictive_io_orchestrator.expert_cache import (  # noqa: E402
    EvictionGroupInfo,
    ExpertCache,
    ExpertInfo,
)


class FakeEvent:
    next_id = 0

    def __init__(self) -> None:
        self.event_id = FakeEvent.next_id
        FakeEvent.next_id += 1
        self.records = []

    def record(self, stream) -> None:
        self.records.append(stream.name)


class FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waits = []

    def wait_event(self, event) -> None:
        self.waits.append(event.event_id)


class CpuAsyncExpertCache(ExpertCache):
    def __init__(
        self,
        capacity: int,
        total_experts: int,
        staging_buffers: int = 0,
    ) -> None:
        self.module_size = 1
        self.device = torch.device("cpu")
        self.active = False
        self.stream_tx = FakeStream("transfer")
        self.compute_stream = FakeStream("compute")
        self.copy_log = []
        self.main_modules = [
            SimpleNamespace(storage=torch.tensor([index], dtype=torch.uint8))
            for index in range(capacity)
        ]
        self.main_infos = [None] * capacity
        self.device_expert_buffers = deque(
            SimpleNamespace(
                storage=torch.tensor([255], dtype=torch.uint8)
            )
            for _ in range(staging_buffers)
        )
        self.offloaded_storages = [
            torch.tensor([index], dtype=torch.uint8)
            for index in range(total_experts)
        ]
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

    def _new_async_event(self):
        return FakeEvent()

    def _async_current_stream(self):
        return self.compute_stream

    def _async_copy_storage(
        self,
        destination,
        source,
        last_use_event,
        ready_event,
    ):
        if last_use_event is not None:
            self.stream_tx.wait_event(last_use_event)
        destination.copy_(source)
        ready_event.record(self.stream_tx)
        self.copy_log.append(
            (int(source.item()), last_use_event is not None)
        )


class AsyncExpertLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeEvent.next_id = 0

    def test_demand_starts_before_first_yield_and_rolls_safely(self) -> None:
        cache = CpuAsyncExpertCache(capacity=2, total_experts=5)
        expert_iter = cache.load_experts_async(
            (0, 0),
            (0, 2),
            (0, 3),
            (0, 4),
            unordered=True,
        )

        self.assertEqual(cache.copy_log, [(2, True)])
        observed = []
        for uid, module in expert_iter:
            observed.append((uid[1], int(module.storage.item())))

        self.assertEqual(observed, [(0, 0), (2, 2), (3, 3), (4, 4)])
        self.assertEqual(
            cache.copy_log,
            [(2, True), (3, True), (4, True)],
        )
        self.assertEqual(
            {info.uid[1] for info in cache.main_infos},
            {3, 4},
        )
        stats = cache.get_async_loading_stats()
        self.assertEqual(stats["initial_inflight_loads"], 1)
        self.assertEqual(stats["rolling_loads"], 2)
        self.assertEqual(stats["demand_misses"], 3)

    def test_prefetch_becomes_ready_resident_demand(self) -> None:
        cache = CpuAsyncExpertCache(capacity=2, total_experts=4)
        prefetched = cache.prefetch_experts_async(
            (0, 2),
            (0, 3),
            unordered=True,
        )
        self.assertEqual([uid[1] for uid, _, _ in prefetched], [2, 3])
        self.assertEqual(cache.copy_log, [(2, True), (3, True)])

        observed = [
            (uid[1], int(module.storage.item()))
            for uid, module in cache.load_experts_async(
                (0, 2),
                (0, 3),
                unordered=True,
            )
        ]
        self.assertEqual(observed, [(2, 2), (3, 3)])
        self.assertEqual(cache.copy_log, [(2, True), (3, True)])
        self.assertGreaterEqual(len(cache.compute_stream.waits), 2)

    def test_initial_resident_slots_have_read_and_reuse_barriers(self) -> None:
        cache = CpuAsyncExpertCache(capacity=2, total_experts=3)

        resident = cache._async_handle_for_info(
            cache.registered_experts[(0, 0)]
        )
        self.assertIsNotNone(resident.ready_event)

        initial_barrier = cache._async_slot_last_use_events[1]
        cache.prefetch_experts_async((0, 2), unordered=True)
        self.assertIn(initial_barrier.event_id, cache.stream_tx.waits)
        self.assertEqual(
            cache.get_async_loading_stats()["initial_slot_barriers"],
            2,
        )

    def test_overlapping_demand_and_prefetch_are_rejected(self) -> None:
        cache = CpuAsyncExpertCache(capacity=2, total_experts=4)
        first = cache.load_experts_async((0, 0), (0, 2))

        with self.assertRaisesRegex(RuntimeError, "overlapping"):
            cache.load_experts_async((0, 1), (0, 3))
        with self.assertRaisesRegex(RuntimeError, "cannot mutate"):
            cache.prefetch_experts_async((0, 3))

        list(first)
        observed = [
            uid
            for uid, _ in cache.load_experts_async((0, 1), (0, 3))
        ]
        self.assertEqual(observed, [(0, 1), (0, 3)])

    def test_disabling_async_hands_stream_ownership_both_ways(self) -> None:
        cache = CpuAsyncExpertCache(capacity=2, total_experts=3)
        cache.prefetch_experts_async((0, 2))
        transfer_waits_before = len(cache.stream_tx.waits)
        compute_waits_before = len(cache.compute_stream.waits)
        cache.disable_specter_async_loading()

        self.assertFalse(cache.specter_async_loading_enabled)
        self.assertEqual(
            len(cache.stream_tx.waits), transfer_waits_before + 1
        )
        self.assertEqual(
            len(cache.compute_stream.waits), compute_waits_before + 1
        )

    def test_reenabling_async_observes_prior_legacy_transfers(self) -> None:
        cache = CpuAsyncExpertCache(capacity=2, total_experts=2)
        cache.disable_specter_async_loading()
        compute_waits_before = len(cache.compute_stream.waits)

        cache.enable_specter_async_loading()

        self.assertTrue(cache.specter_async_loading_enabled)
        self.assertEqual(
            len(cache.compute_stream.waits), compute_waits_before + 1
        )

    def test_staging_buffers_preserve_hits_and_start_all_initial_misses(self) -> None:
        cache = CpuAsyncExpertCache(
            capacity=2,
            total_experts=5,
            staging_buffers=2,
        )
        expert_iter = cache.load_experts_async(
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            unordered=True,
        )

        self.assertEqual(cache.copy_log, [(2, False), (3, False)])
        observed = [
            (uid[1], int(module.storage.item()))
            for uid, module in expert_iter
        ]
        self.assertEqual(
            observed,
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        )
        self.assertEqual(
            cache.copy_log,
            [(2, False), (3, False), (4, True)],
        )
        self.assertEqual(
            {info.uid[1] for info in cache.main_infos},
            {0, 1},
        )
        self.assertEqual(len(cache._async_staging_available), 2)
        stats = cache.get_async_loading_stats()
        self.assertEqual(stats["staging_loads"], 3)
        self.assertEqual(stats["initial_inflight_loads"], 2)
        self.assertEqual(stats["rolling_loads"], 1)

    def test_legacy_loader_remains_available_when_async_is_disabled(self) -> None:
        cache = CpuAsyncExpertCache(capacity=2, total_experts=2)
        cache.specter_async_loading_enabled = False
        observed = [
            uid[1]
            for uid, _ in cache.load_experts(
                (0, 0),
                (0, 1),
                unordered=True,
            )
        ]
        self.assertEqual(observed, [0, 1])
        self.assertEqual(cache.copy_log, [])




if __name__ == "__main__":
    unittest.main()
