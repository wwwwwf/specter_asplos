"""Independent immutable-host, per-layer LRU cache for the retained adaptation.

Derived from the author's specoffmoe legacy cache, itself based on
Mixtral-Offloading (MIT, see LICENSE.mixtral-offloading). Slot reuse is guarded
by CUDA events; unlike the old generator, requests may safely exceed capacity.
"""
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ExpertInfo:
    uid: Any
    eviction_group: int
    offloaded: bool = True
    cache_index: int = -1
    mem_index: int = -1


@dataclass
class EvictionGroupInfo:
    main_infos: OrderedDict = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def mark_used(self, info):
        if info.offloaded:
            self.misses += 1
            self.offloaded_infos.move_to_end(info.uid)
        else:
            self.hits += 1
            self.main_infos.move_to_end(info.uid)


class ExpertCache:
    """Keep a canonical host copy of every expert and reuse resident slots.

    Predictive Top-M admissions run on a transfer stream. Demand consumes hits
    first, then a bounded rolling window of misses. No Specter planner or
    Specter asynchronous cache implementation is used.
    """

    specter_async_loading_enabled = False
    specter_async_overlap_shared_experts = False

    def __init__(self, make_module, main_size, offload_size, buffer_size, model_type=None):
        if main_size <= 0 or offload_size < 0 or buffer_size < 0:
            raise ValueError("Invalid resident/offload/buffer capacity")
        create = (lambda: make_module(model_type)) if model_type is not None else make_module
        self._make_module = create
        self.main_modules = [create() for _ in range(main_size)]
        self.module_type = type(self.main_modules[0])
        self.module_size = len(self.main_modules[0].storage)
        self.device = self.main_modules[0].storage.device
        self.main_infos = [None] * main_size
        self.offloaded_storages = [self._host_storage() for _ in range(main_size + offload_size)]
        self.offloaded_infos = [None] * len(self.offloaded_storages)
        # Retain the adaptation's configured allocation/footprint. These are
        # not silently repurposed as Specter's transient staging slots.
        self.device_expert_buffers = deque(create() for _ in range(buffer_size))
        self.offloaded_storage_buffers = deque(self._host_storage() for _ in range(buffer_size))
        self.registered_experts = {}
        self.group_infos = defaultdict(EvictionGroupInfo)
        self.stream_tx = torch.cuda.Stream(device=self.device)
        self.active = False
        self._slot_ready = [None] * main_size
        self._slot_last_use = [None] * main_size
        self.stats = defaultdict(int)
        self._initial = None

    def _host_storage(self):
        # Fail rather than silently benchmark pageable host memory.
        return torch.UntypedStorage(self.module_size).pin_memory()

    def add_expert(self, uid, module, eviction_group=0, offload=None):
        return self.add_expert_storage(uid, module.storage, eviction_group, offload)

    def add_expert_storage(self, uid, storage, eviction_group=0, offload=None):
        if len(storage) != self.module_size:
            raise ValueError("Expert storage size mismatch")
        group = self.group_infos[eviction_group]
        if uid not in self.registered_experts:
            host_index = len(self.registered_experts)
            if host_index >= len(self.offloaded_storages):
                raise ValueError("Host expert capacity exceeded")
            self.offloaded_storages[host_index].copy_(storage)
            info = ExpertInfo(uid, eviction_group, mem_index=host_index)
            self.registered_experts[uid] = info
            self.offloaded_infos[host_index] = info
            group.offloaded_infos[uid] = info
        else:
            info = self.registered_experts[uid]
            if info.eviction_group != eviction_group:
                raise ValueError("Expert cannot change eviction group")
        if offload is False or offload is None:
            if not info.offloaded:
                raise ValueError("Expert already resident")
            try:
                slot = self.main_infos.index(None)
            except ValueError as exc:
                raise ValueError("Resident expert capacity exceeded") from exc
            self.main_modules[slot].storage.copy_(storage)
            self.main_infos[slot] = info
            info.offloaded, info.cache_index = False, slot
            group.offloaded_infos.pop(uid)
            group.main_infos[uid] = info

    def _event(self, stream):
        event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
        return event

    def _load(self, info, protected=()):
        if not info.offloaded:
            return info.cache_index
        group = self.group_infos[info.eviction_group]
        victim = next((item for uid, item in group.main_infos.items() if uid not in protected), None)
        if victim is None:
            raise RuntimeError("No evictable slot; predictive request exceeds layer cache")
        slot = victim.cache_index
        with torch.cuda.stream(self.stream_tx):
            for event in (self._slot_ready[slot], self._slot_last_use[slot]):
                if event is not None:
                    self.stream_tx.wait_event(event)
            self.main_modules[slot].storage.copy_(self.offloaded_storages[info.mem_index], non_blocking=True)
            self._slot_ready[slot] = self._event(self.stream_tx)
        self._slot_last_use[slot] = None
        group.main_infos.pop(victim.uid)
        group.offloaded_infos.pop(info.uid)
        victim.offloaded, victim.cache_index = True, -1
        info.offloaded, info.cache_index = False, slot
        group.offloaded_infos[victim.uid] = victim
        group.main_infos[info.uid] = info
        self.main_infos[slot] = info
        self.stats['h2d_submissions'] += 1
        self.stats['h2d_bytes'] += self.module_size
        return slot

    def load_experts(self, *uids, unordered=False, prefetch=False):
        if prefetch:
            raise ValueError("Use prefetch_experts for predictive admissions")
        if self.active or len(set(uids)) != len(uids):
            raise RuntimeError("Concurrent demand or duplicate experts")
        if not uids:
            return
        infos = [self.registered_experts[uid] for uid in uids]
        if len({info.eviction_group for info in infos}) != 1:
            raise ValueError("Demand must remain in one layer")
        if unordered:
            infos.sort(key=lambda info: info.offloaded)
        group = self.group_infos[infos[0].eviction_group]
        hits = [info for info in infos if not info.offloaded]
        misses = [info for info in infos if info.offloaded]
        self.stats['demand_requests'] += len(infos)
        self.stats['demand_hits'] += len(hits)
        self.stats['demand_misses'] += len(misses)
        for info in infos:
            group.mark_used(info)
        self.active = True
        pending = deque()
        unscheduled = deque(misses)
        protected = {info.uid for info in hits}

        def fill():
            while unscheduled and len(protected) < len(group.main_infos):
                item = unscheduled.popleft()
                slot = self._load(item, protected)
                pending.append((item, slot))
                protected.add(item.uid)

        try:
            # For the normal unordered MoE path, pre-submit available misses
            # without evicting hits; roll each slot only after its consumer.
            if not unordered:
                for info in infos:
                    slot = self._load(info)
                    stream = torch.cuda.current_stream(self.device)
                    if self._slot_ready[slot] is not None:
                        stream.wait_event(self._slot_ready[slot])
                    try:
                        yield info.uid, self.main_modules[slot]
                    finally:
                        self._slot_last_use[slot] = self._event(stream)
                return
            fill()
            for info in hits:
                slot = info.cache_index
                stream = torch.cuda.current_stream(self.device)
                if self._slot_ready[slot] is not None:
                    stream.wait_event(self._slot_ready[slot])
                try:
                    yield info.uid, self.main_modules[slot]
                finally:
                    self._slot_last_use[slot] = self._event(stream)
                protected.remove(info.uid)
                fill()
            while pending:
                info, slot = pending.popleft()
                stream = torch.cuda.current_stream(self.device)
                stream.wait_event(self._slot_ready[slot])
                try:
                    yield info.uid, self.main_modules[slot]
                finally:
                    self._slot_last_use[slot] = self._event(stream)
                protected.remove(info.uid)
                fill()
        finally:
            self.active = False

    def prefetch_experts(self, *uids, unordered=False):
        del unordered
        if self.active or len(set(uids)) != len(uids):
            raise RuntimeError("Predictive admission during active demand or duplicate UID")
        protected = set(uids)
        for uid in uids:
            info = self.registered_experts[uid]
            self.group_infos[info.eviction_group].mark_used(info)
            slot = self._load(info, protected)
            yield uid, self.main_modules[slot], self._slot_ready[slot]

    def snapshot_initial(self):
        torch.cuda.synchronize(self.device)
        self._initial = {key: (tuple(group.main_infos), tuple(group.offloaded_infos))
                         for key, group in self.group_infos.items()}

    def resize_residency(self, resident_per_layer):
        """Rebuild resident mappings from canonical host weights, outside timing.

        No checkpoint is reloaded. Transfer buffers retain their configured
        size. Only the difference in resident slot count is allocated/freed.
        """
        resident_per_layer = int(resident_per_layer)
        if self.active or resident_per_layer < 1:
            raise ValueError('Resize requires inactive demand and positive residency')
        torch.cuda.synchronize(self.device)
        all_groups = {key: sorted((*group.main_infos, *group.offloaded_infos))
                      for key, group in self.group_infos.items()}
        if any(resident_per_layer > len(uids) for uids in all_groups.values()):
            raise ValueError('Residency exceeds expert count')
        count = len(all_groups) * resident_per_layer
        extra = [self._make_module() for _ in range(max(0, count - len(self.main_modules)))]
        self.main_modules = (self.main_modules + extra)[:count]
        self.main_infos = [None] * count
        self._slot_ready = [None] * count
        self._slot_last_use = [None] * count
        slot = 0
        for key, uids in all_groups.items():
            group = self.group_infos[key]
            group.main_infos.clear()
            group.offloaded_infos.clear()
            selected = set(uids[-resident_per_layer:])
            for uid in uids:
                info = self.registered_experts[uid]
                info.offloaded, info.cache_index = True, -1
                if uid in selected:
                    self.main_modules[slot].storage.copy_(self.offloaded_storages[info.mem_index])
                    info.offloaded, info.cache_index = False, slot
                    self.main_infos[slot] = info
                    group.main_infos[uid] = info
                    slot += 1
                else:
                    group.offloaded_infos[uid] = info
            group.hits = group.misses = 0
        torch.cuda.synchronize(self.device)
        self.stats.clear()
        self.snapshot_initial()

    def restore(self):
        if self._initial is None or self.active:
            raise RuntimeError("Missing initial cache state or active demand")
        torch.cuda.synchronize(self.device)
        for key, (resident, offloaded) in self._initial.items():
            for _ in self.prefetch_experts(*resident):
                pass
            group = self.group_infos[key]
            if set(group.main_infos) != set(resident):
                raise RuntimeError("Initial resident restoration failed")
            group.main_infos = OrderedDict((uid, self.registered_experts[uid]) for uid in resident)
            group.offloaded_infos = OrderedDict((uid, self.registered_experts[uid]) for uid in offloaded)
            group.hits = group.misses = 0
        torch.cuda.synchronize(self.device)
        self.stats.clear()
