from streamlined_execution_engine.overlap_optimizer import OverlapOrientedParallelismOptimizer
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List
from collections import deque, defaultdict, OrderedDict
from streamlined_execution_engine.expert_storage import ExpertWrapper
import psutil
import torch
from torch import nn
import subprocess
import time
import threading
import queue
from queue import Queue

def print_gpu_memory():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8')
        # Parse memory usage from the command output.
        memory_info = result.strip().split("\n")
        for gpu, mem_info in enumerate(memory_info):
            used, total = mem_info.split(', ')
            print(f"GPU {gpu}: used memory {used} MB / total memory {total} MB")
    except Exception as e:
        print(f"Failed to query GPU memory: {e}")

ExpertUID = Any


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    cache_index: int = -1
    mem_index: int = -1


@dataclass(frozen=True)
class AsyncExpertHandle:
    uid: ExpertUID
    info: ExpertInfo
    expert: ExpertWrapper
    cache_index: int
    generation: int
    ready_event: Any = None
    staging: bool = False

@dataclass
class EvictionGroupInfo:
    # infos in main and offload devices; ordered from least recently used to most
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        for uid, info in self.main_infos.items():
            return info  # least recently used
        raise ValueError("No evictable experts")

    def load(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos.pop(info_to_evict.uid)
        self.main_infos[info_to_load.uid] = self.offloaded_infos[info_to_load.uid]
        self.main_infos.move_to_end(info_to_load.uid, last=True)


    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class VerificationExpertCacheResidencyManager(OverlapOrientedParallelismOptimizer):
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, model_type:int):
        """Dynamically loads an array of modules with identical hyperparameters"""
        print("expert cache init")
        self.module_type = self.module_size = self.device = None
        self.active = False
        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()
        self.main_modules = []       
        # compute use default stream
        self.stream_rx = torch.cuda.Stream()
        self.stream_tx = torch.cuda.Stream()
        self.stream_prefetch = torch.cuda.Stream()
        self.global_event = None

        #print("main_modules")
        #print(main_size)
        for i in range(main_size):
            #print(f"=={i} of {main_size}==")  # Print the current index.
            self.main_modules.append(self._check_module(make_module(model_type)))
        
        print_gpu_memory()
        # self.main_modules = [self._check_module(make_module()) for i in range(main_size)]
        print("main infos")
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        print("offload storages")
        # self.offloaded_storages = [
        #     torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(offload_size)]
        self.offloaded_storages = []
        # all in cpu
        for i in range(offload_size + main_size):
            #print(f"{i} of {offload_size}")
            memory = psutil.virtual_memory()
            #print(f"Available memory: {memory.available / (1024**3):.2f} GB")
            storage = self._make_host_offload_storage()
            self.offloaded_storages.append(storage)
        print("offload infos")
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size + main_size)]

        memory = psutil.virtual_memory()
        #print(f"Available memory: {memory.available / (1024**3):.2f} GB")

        # temporary storage to shave off latency
        print("device_expert_buffers")
        self.device_expert_buffers = deque([self._check_module(make_module(model_type)) for _ in range(buffer_size)])
        print("offloaded_storage_buffers")
        self.offloaded_storage_buffers = deque([
            self._make_host_offload_storage() for _ in range(buffer_size)])
        print("group_infos")
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)
        self.specter_async_loading_enabled = False
        # ``staging`` preserves the prediction-managed resident set, while
        # ``resident`` admits demand misses into the cache.  The latter is
        # important for models whose demand experts are reused across steps.
        self.specter_async_demand_policy = "staging"
        self._async_slot_ready_events: List[Any] = []
        self._async_slot_last_use_events: List[Any] = []
        self._async_slot_reuse_events: List[Any] = []
        self._async_slot_generations: List[int] = []
        self._async_staging_capacity = 0
        self._async_staging_available = deque()
        self._async_staging_ready_events: Dict[int, Any] = {}
        self._async_staging_last_use_events: Dict[int, Any] = {}
        self._async_staging_reuse_events: Dict[int, Any] = {}
        self._async_staging_generations: Dict[int, int] = {}
        self._async_state_lock = threading.RLock()
        self._async_demand_active = False
        self.async_loading_stats = defaultdict(int)

    def _check_module(self, module: ExpertWrapper):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def _make_host_offload_storage(self) -> torch.UntypedStorage:
        storage = torch.UntypedStorage(self.module_size)
        try:
            return storage.pin_memory()
        except RuntimeError as exc:
            print(f"[warn] host pin_memory() failed, fallback to unpinned storage ({exc})")
            return storage

    def add_expert(self, uid: ExpertUID, module: ExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        #assert uid not in self.registered_experts, f"expert {uid} already registered"
        # if uid in self.registered_experts:
        #     print(f"expert {uid} is registed in cpu, adding to cache")
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = self.registered_experts[uid]
                    info.cache_index = i
                    info.offloaded = False
                    self.main_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found spot on device
        elif offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, cache_index=-1, mem_index=i)
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def enable_specter_async_loading(self):
        """Enable the Specter-only event-driven expert loading path."""
        with self._async_state_lock:
            if self.specter_async_loading_enabled:
                return
            slot_count = len(self.main_modules)

            # Resident slots were populated on the current stream while the
            # model was loaded.  The first transfer-stream eviction must not
            # overwrite them until those writes complete, and the first
            # compute-stream hit must observe the initialized contents.
            current_stream = self._async_current_stream()
            prior_transfer_done = self._new_async_event()
            self._async_record_event(prior_transfer_done, self.stream_tx)
            self._async_wait_event(current_stream, prior_transfer_done)
            initial_last_use = self._new_async_event()
            self._async_record_event(initial_last_use, current_stream)
            initial_ready_events = []
            for _ in range(slot_count):
                ready_event = self._new_async_event()
                self._async_record_event(ready_event, current_stream)
                initial_ready_events.append(ready_event)

            self._async_slot_ready_events = initial_ready_events
            self._async_slot_last_use_events = [
                initial_last_use
            ] * slot_count
            self._async_slot_reuse_events = [None] * slot_count
            self._async_slot_generations = [0] * slot_count
            self._async_staging_capacity = len(self.device_expert_buffers)
            self._async_staging_available = self.device_expert_buffers
            self._async_staging_ready_events = {}
            self._async_staging_last_use_events = {}
            self._async_staging_reuse_events = {}
            self._async_staging_generations = {
                id(module): 0 for module in self.device_expert_buffers
            }
            self._async_demand_active = False
            self.async_loading_stats.clear()
            self.async_loading_stats["initial_slot_barriers"] = slot_count
            self.specter_async_loading_enabled = True

    def set_specter_async_demand_policy(self, policy: str):
        """Choose whether async demand misses use staging or resident slots."""
        if policy not in {"staging", "resident"}:
            raise ValueError(
                "async demand policy must be 'staging' or 'resident'"
            )
        with self._async_state_lock:
            if self._async_demand_active:
                raise RuntimeError(
                    "cannot change async demand policy during an active demand"
                )
            self.specter_async_demand_policy = policy

    def disable_specter_async_loading(self):
        """Return to the legacy loader without exposing in-flight writes."""
        with self._async_state_lock:
            if not self.specter_async_loading_enabled:
                return
            if self._async_demand_active:
                raise RuntimeError(
                    "cannot disable async loading during an active demand"
                )
            # Hand ownership back in both directions.  Existing H2D writes
            # finish before the current stream proceeds, while any later
            # legacy H2D copy on stream_tx remains behind current compute.
            current_stream = self._async_current_stream()
            current_done = self._new_async_event()
            self._async_record_event(current_done, current_stream)
            self._async_wait_event(self.stream_tx, current_done)
            transfer_done = self._new_async_event()
            self._async_record_event(transfer_done, self.stream_tx)
            self._async_wait_event(current_stream, transfer_done)
            self.specter_async_loading_enabled = False

    def get_async_loading_stats(self) -> Dict[str, int]:
        return dict(self.async_loading_stats)

    def reset_async_loading_stats(self):
        self.async_loading_stats.clear()

    def _async_event_for_slot(self, events: List[Any], cache_index: int):
        event = events[cache_index]
        if event is None:
            event = self._new_async_event()
            events[cache_index] = event
        return event

    def _async_event_for_staging(self, events: Dict[int, Any], expert):
        expert_id = id(expert)
        event = events.get(expert_id)
        if event is None:
            event = self._new_async_event()
            events[expert_id] = event
        return event

    def _async_handle_for_info(self, info: ExpertInfo) -> AsyncExpertHandle:
        cache_index = info.cache_index
        if info.offloaded or cache_index < 0:
            raise RuntimeError(f"expert {info.uid} is not resident")
        return AsyncExpertHandle(
            uid=info.uid,
            info=info,
            expert=self.main_modules[cache_index],
            cache_index=cache_index,
            generation=self._async_slot_generations[cache_index],
            ready_event=self._async_slot_ready_events[cache_index],
        )

    def _validate_async_handle(self, handle: AsyncExpertHandle):
        if handle.staging:
            generation = self._async_staging_generations.get(
                id(handle.expert)
            )
            if generation != handle.generation:
                raise RuntimeError(
                    f"staging buffer was reused before {handle.uid}"
                )
            return
        cache_index = handle.cache_index
        if self._async_slot_generations[cache_index] != handle.generation:
            raise RuntimeError(
                f"expert slot {cache_index} was reused before {handle.uid}"
            )
        if self.main_infos[cache_index] is not handle.info:
            raise RuntimeError(
                f"expert slot {cache_index} no longer contains {handle.uid}"
            )

    def _submit_async_staging_load(
        self,
        info_to_load: ExpertInfo,
        expert: ExpertWrapper,
    ) -> AsyncExpertHandle:
        if not info_to_load.offloaded or info_to_load.mem_index < 0:
            raise RuntimeError("invalid staging expert-cache transition")
        expert_id = id(expert)
        ready_event = self._async_event_for_staging(
            self._async_staging_ready_events,
            expert,
        )
        last_use_event = self._async_staging_last_use_events.get(expert_id)
        self._async_copy_storage(
            expert.storage,
            self.offloaded_storages[info_to_load.mem_index],
            last_use_event,
            ready_event,
        )
        generation = self._async_staging_generations.get(expert_id, 0) + 1
        self._async_staging_generations[expert_id] = generation
        self._async_staging_last_use_events[expert_id] = None
        self.async_loading_stats["h2d_submissions"] += 1
        self.async_loading_stats["h2d_bytes"] += int(self.module_size)
        self.async_loading_stats["staging_loads"] += 1
        if last_use_event is not None:
            self.async_loading_stats["staging_reuse_dependencies"] += 1
        return AsyncExpertHandle(
            uid=info_to_load.uid,
            info=info_to_load,
            expert=expert,
            cache_index=-1,
            generation=generation,
            ready_event=ready_event,
            staging=True,
        )

    def _submit_async_load(
        self,
        info_to_load: ExpertInfo,
        info_to_evict: ExpertInfo,
    ) -> AsyncExpertHandle:
        if not info_to_load.offloaded or info_to_evict.offloaded:
            raise RuntimeError("invalid async expert-cache transition")
        if info_to_load.eviction_group != info_to_evict.eviction_group:
            raise RuntimeError("async load cannot cross eviction groups")
        if info_to_load.mem_index < 0:
            raise RuntimeError(
                f"expert {info_to_load.uid} has no immutable host backing"
            )

        cache_index = info_to_evict.cache_index
        if cache_index < 0 or info_to_load.cache_index != -1:
            raise RuntimeError("invalid async expert-cache indices")

        ready_event = self._async_event_for_slot(
            self._async_slot_ready_events,
            cache_index,
        )
        last_use_event = self._async_slot_last_use_events[cache_index]
        self._async_copy_storage(
            self.main_modules[cache_index].storage,
            self.offloaded_storages[info_to_load.mem_index],
            last_use_event,
            ready_event,
        )

        self.main_infos[cache_index] = info_to_load
        self.offloaded_infos[info_to_load.mem_index] = info_to_evict
        info_to_evict.offloaded = True
        info_to_load.offloaded = False
        info_to_evict.cache_index = -1
        info_to_load.cache_index = cache_index
        self.group_infos[info_to_load.eviction_group].load(
            info_to_load,
            info_to_evict,
        )

        self._async_slot_generations[cache_index] += 1
        self._async_slot_last_use_events[cache_index] = None
        self.async_loading_stats["h2d_submissions"] += 1
        self.async_loading_stats["h2d_bytes"] += int(self.module_size)
        if last_use_event is not None:
            self.async_loading_stats["slot_reuse_dependencies"] += 1
        return self._async_handle_for_info(info_to_load)

    def _prepare_async_demand(
        self,
        uids: Tuple[ExpertUID, ...],
        unordered: bool,
    ):
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate demand expert UIDs")
        if unordered:
            uids = tuple(
                sorted(
                    uids,
                    key=lambda uid: self.registered_experts[uid].offloaded,
                )
            )
        infos = [self.registered_experts[uid] for uid in uids]
        if not infos:
            return deque(), deque(), False, []
        groups = {info.eviction_group for info in infos}
        if len(groups) != 1:
            raise ValueError("demand experts must share an eviction group")

        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)
        hits = [info for info in infos if not info.offloaded]
        misses = [info for info in infos if info.offloaded]
        capacity = len(eviction_group.main_infos)
        if capacity <= 0:
            raise RuntimeError("eviction group has no resident cache slots")
        if len(hits) > capacity:
            raise RuntimeError("resident demand set exceeds cache capacity")

        use_staging = (
            getattr(
                self,
                "specter_async_demand_policy",
                "staging",
            ) == "staging"
            and self._async_staging_capacity > 0
        )
        staging_modules = []
        if use_staging:
            if (
                len(self._async_staging_available)
                != self._async_staging_capacity
            ):
                raise RuntimeError(
                    "a previous async demand iterator still owns staging buffers"
                )
            initial_count = min(
                len(misses),
                self._async_staging_capacity,
            )
            for _ in range(initial_count):
                staging_modules.append(
                    self._async_staging_available.popleft()
                )
            try:
                loaded = [
                    self._submit_async_staging_load(info, expert)
                    for info, expert in zip(
                        misses[:initial_count],
                        staging_modules,
                    )
                ]
            except Exception:
                self._async_staging_available.extend(staging_modules)
                raise
        else:
            protected_uids = {info.uid for info in hits}
            victims = [
                info
                for uid, info in list(eviction_group.main_infos.items())
                if uid not in protected_uids
            ]
            initial_count = min(len(misses), len(victims))
            loaded = [
                self._submit_async_load(misses[index], victims[index])
                for index in range(initial_count)
            ]
        handles = deque(
            [self._async_handle_for_info(info) for info in hits] + loaded
        )
        remaining = deque(misses[initial_count:])

        self.async_loading_stats["demand_plans"] += 1
        self.async_loading_stats["demand_requests"] += len(infos)
        self.async_loading_stats["demand_hits"] += len(hits)
        self.async_loading_stats["demand_misses"] += len(misses)
        self.async_loading_stats["initial_inflight_loads"] += initial_count
        return handles, remaining, use_staging, staging_modules

    def _iterate_async_demand(
        self,
        handles,
        remaining,
        use_staging,
        staging_modules,
    ):
        current_stream = self._async_current_stream()
        last_used_slots = {}
        last_used_staging = {}
        try:
            while handles:
                handle = handles.popleft()
                self._validate_async_handle(handle)
                if handle.ready_event is not None:
                    self._async_wait_event(current_stream, handle.ready_event)
                    self.async_loading_stats["ready_dependencies"] += 1
                try:
                    yield handle.uid, handle.expert
                finally:
                    if handle.staging:
                        last_used_staging[id(handle.expert)] = (
                            handle.generation,
                            handle.expert,
                        )
                    else:
                        last_used_slots[handle.cache_index] = (
                            handle.generation
                        )

                if remaining:
                    if use_staging:
                        if not handle.staging:
                            continue
                        reuse_event = self._async_event_for_staging(
                            self._async_staging_reuse_events,
                            handle.expert,
                        )
                        self._async_record_event(
                            reuse_event,
                            current_stream,
                        )
                        self._async_staging_last_use_events[
                            id(handle.expert)
                        ] = reuse_event
                        handles.append(
                            self._submit_async_staging_load(
                                remaining.popleft(),
                                handle.expert,
                            )
                        )
                    else:
                        reuse_event = self._async_event_for_slot(
                            self._async_slot_reuse_events,
                            handle.cache_index,
                        )
                        self._async_record_event(
                            reuse_event,
                            current_stream,
                        )
                        self._async_slot_last_use_events[
                            handle.cache_index
                        ] = reuse_event
                        handles.append(
                            self._submit_async_load(
                                remaining.popleft(),
                                handle.info,
                            )
                        )
                    self.async_loading_stats["rolling_loads"] += 1
        finally:
            final_slots = [
                cache_index
                for cache_index, generation in last_used_slots.items()
                if self._async_slot_generations[cache_index] == generation
            ]
            final_staging = [
                expert
                for expert_id, (
                    generation,
                    expert,
                ) in last_used_staging.items()
                if self._async_staging_generations.get(expert_id)
                == generation
            ]
            if final_slots or final_staging:
                done_event = self._new_async_event()
                self._async_record_event(done_event, current_stream)
                for cache_index in final_slots:
                    self._async_slot_last_use_events[cache_index] = done_event
                for expert in final_staging:
                    self._async_staging_last_use_events[
                        id(expert)
                    ] = done_event
                self.async_loading_stats["layer_use_events"] += 1
            available_ids = {
                id(expert) for expert in self._async_staging_available
            }
            for expert in staging_modules:
                if id(expert) not in available_ids:
                    self._async_staging_available.append(expert)
                    available_ids.add(id(expert))
            with self._async_state_lock:
                self._async_demand_active = False

    def load_experts_async(
        self,
        *uids: ExpertUID,
        unordered: bool = True,
        prefetch: bool = False,
    ) -> Iterator[Tuple[ExpertUID, ExpertWrapper]]:
        """Plan H2D first, then yield experts through a safe rolling window.

        Depending on ``specter_async_demand_policy``, demand misses either use
        transient buffers or enter the persistent cache through the same
        event-driven slot protocol.
        """
        del prefetch
        if not self.specter_async_loading_enabled:
            raise RuntimeError("Specter async loading is not enabled")
        with self._async_state_lock:
            if self._async_demand_active:
                raise RuntimeError(
                    "overlapping async expert-demand iterators are unsafe"
                )
            self._async_demand_active = True
            try:
                (
                    handles,
                    remaining,
                    use_staging,
                    staging_modules,
                ) = self._prepare_async_demand(
                    tuple(uids),
                    unordered,
                )
            except Exception:
                self._async_demand_active = False
                raise
        return self._iterate_async_demand(
            handles,
            remaining,
            use_staging,
            staging_modules,
        )

    def prefetch_experts_async(
        self,
        *uids: ExpertUID,
        unordered: bool = True,
        prefetch: bool = True,
    ):
        with self._async_state_lock:
            if self._async_demand_active:
                raise RuntimeError(
                    "predictive prefetch cannot mutate cache metadata while "
                    "an async demand iterator owns expert slots"
                )
            return self._prefetch_experts_async_locked(
                *uids,
                unordered=unordered,
                prefetch=prefetch,
            )

    def prefetch_experts_bounded_async(
        self,
        *uids: ExpertUID,
        max_new_loads: int,
        unordered: bool = True,
        prefetch: bool = True,
    ):
        """Protect the complete desired set but admit bounded new copies."""
        if max_new_loads < 0:
            raise ValueError("max_new_loads must be non-negative")
        with self._async_state_lock:
            if self._async_demand_active:
                raise RuntimeError(
                    "predictive prefetch cannot mutate cache metadata while "
                    "an async demand iterator owns expert slots"
                )
            return self._prefetch_experts_async_locked(
                *uids,
                unordered=unordered,
                prefetch=prefetch,
                max_new_loads=int(max_new_loads),
            )

    def _prefetch_experts_async_locked(
        self,
        *uids: ExpertUID,
        unordered: bool = True,
        prefetch: bool = True,
        max_new_loads: Optional[int] = None,
    ):
        """Submit safe predictive H2D copies without host synchronization."""
        del prefetch
        if not self.specter_async_loading_enabled:
            raise RuntimeError("Specter async loading is not enabled")
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate prefetch expert UIDs")
        if unordered:
            uids = tuple(
                sorted(
                    uids,
                    key=lambda uid: self.registered_experts[uid].offloaded,
                )
            )
        infos = [self.registered_experts[uid] for uid in uids]
        if not infos:
            return []
        groups = {info.eviction_group for info in infos}
        if len(groups) != 1:
            raise ValueError("prefetch experts must share an eviction group")

        eviction_group = self.group_infos[infos[0].eviction_group]
        capacity = len(eviction_group.main_infos)
        if len(infos) > capacity:
            raise ValueError(
                "prefetch set exceeds the eviction-group cache capacity"
            )
        for info in infos:
            eviction_group.mark_used(info)
        hits = [info for info in infos if not info.offloaded]
        misses = [info for info in infos if info.offloaded]
        admitted_misses = (
            misses
            if max_new_loads is None
            else misses[:max_new_loads]
        )
        protected_uids = {info.uid for info in hits}
        victims = [
            info
            for uid, info in list(eviction_group.main_infos.items())
            if uid not in protected_uids
        ]
        if len(victims) < len(admitted_misses):
            raise RuntimeError("insufficient safe cache slots for prefetch")

        loaded_by_uid = {}
        for info, victim in zip(admitted_misses, victims):
            loaded_by_uid[info.uid] = self._submit_async_load(info, victim)
        handles = [
            loaded_by_uid.get(info.uid) or self._async_handle_for_info(info)
            for info in infos
            if not info.offloaded or info.uid in loaded_by_uid
        ]
        self.async_loading_stats["prefetch_plans"] += 1
        self.async_loading_stats["prefetch_requests"] += len(infos)
        self.async_loading_stats["prefetch_candidate_misses"] += len(misses)
        self.async_loading_stats["prefetch_misses"] += len(admitted_misses)
        self.async_loading_stats["prefetch_deferred_misses"] += (
            len(misses) - len(admitted_misses)
        )
        return [
            (handle.uid, handle.expert, handle.ready_event)
            for handle in handles
        ]

    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False, prefetch: bool = False) -> Iterator[Tuple[ExpertUID, ExpertWrapper]]:
        """
        :example:
        >>> for uid, expert in expert_cache.load_experts(*list_of_uids, unordered=True):
        >>>     for uid, expert in expert_iter:
        >>>         result += expert(x) * get_moe_weight(uid)

        :param uids: iterate over the specified expert uids. Same uids as in add_expert
        :param unordered: if True, allows cache to iterate experts in arbitrary order
            The order is chosen to minimize the total wait time.
        :returns: an iterator that yields (uid, expert) pairs, only usable inside the for loop

        """
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:  # yield non-offloaded experts first
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]
        # print("in normal load")
        # print([uid for uid in uids])
        # print([(info.eviction_group, type(info.eviction_group)) for info in infos])
        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)
        t0=time.time()
        try:
            self.active = True
            # save pre-loaded experts before they can be swapped
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.cache_index] for info in pre_loaded_infos])
            #print(f"hit expert: {len(pre_loaded_infos)}")
            # begin loading experts into free buffers in background (via non-blocking copy)
            infos_to_load = deque([info for info in infos if info.offloaded])
            t0 = time.time()
            nums = len(infos_to_load)

            infos_in_loading = deque([])
            experts_in_loading = deque([])

            # get hitted expert and call to swap others
            for info in infos:
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                else:
                    if len(infos_to_load) > 0:
                        info_to_load = infos_to_load.popleft()
                        infos_in_loading.append(info_to_load)
                        experts_in_loading.append(
                            self._load(info_to_load, eviction_group.choose_expert_to_evict()))
            # yield others
            while len(experts_in_loading) > 0:
                info = infos_in_loading.popleft()
                expert, load_event = experts_in_loading.popleft()
                torch.cuda.current_stream().wait_event(load_event)
                yield (info.uid, expert)
        finally:
            self.active = False

    def _load(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        """Swap an offloaded expert (info_to_load) with an on-device expert (info_to_evict) return the loaded expert"""
        torch.cuda.nvtx.range_push(f"[LOAD] Expert {info_to_load.uid}")
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        assert info_to_evict.cache_index != -1 and info_to_load.cache_index == -1
        
        load_done_event = torch.cuda.Event(enable_timing=False, blocking=False)
        with torch.cuda.stream(self.stream_tx):
            self.main_modules[info_to_evict.cache_index].storage.copy_(self.offloaded_storages[info_to_load.mem_index], non_blocking=True)
        
        self.main_infos[info_to_evict.cache_index] = info_to_load
        self.offloaded_infos[info_to_load.mem_index] = info_to_evict
        
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_evict.cache_index, info_to_load.cache_index = info_to_load.cache_index, info_to_evict.cache_index
        
        self.group_infos[info_to_load.eviction_group].load(info_to_load, info_to_evict)
        torch.cuda.nvtx.range_pop()
        load_done_event.record(self.stream_tx)
        
        return self.main_modules[info_to_load.cache_index], load_done_event  
    

    def _print_all_expert_states(self):
        """Print the status and parameter summaries of registered experts for debugging."""
        if not self.registered_experts:
            print("  > No registered experts")
            return

        print(f"  Registered experts: {len(self.registered_experts)}")
        for uid, info in self.registered_experts.items():
            loc = "GPU" if not info.offloaded else "CPU"
            cache_idx = info.cache_index if not info.offloaded else "-"
            mem_idx = info.mem_index if info.offloaded else "-"
            # Use the first few storage values as a parameter fingerprint.
            try:
                if not info.offloaded:
                    storage = self.main_modules[info.cache_index].storage
                else:
                    storage = self.offloaded_storages[info.mem_index]
                # Cast to float32 to inspect the first four values (assuming floating-point weights).
                param_preview = torch.as_tensor(storage[:16], dtype=torch.float16)[:4].tolist()
                param_str = ", ".join(f"{x:.4f}" for x in param_preview)
            except Exception as e:
                param_str = f"<ERROR: {str(e)}>"

            print(f"  UID: {uid} | Location: {loc} | CacheIdx: {cache_idx} | MemIdx: {mem_idx} | Parameter preview: [{param_str}]")

ExpertCache = VerificationExpertCacheResidencyManager
