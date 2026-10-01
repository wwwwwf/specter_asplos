"""Untimed observation/reset helpers for the byte-identical upstream cache.

No loading, eviction, transfer, iteration, or resize implementation is patched.
Inference calls upstream.src.expert_cache.ExpertCache directly.
"""
from collections import OrderedDict

import torch


def snapshot_initial(cache):
    """Record IDs and LRU order without copying expert tensor storage."""
    if cache.active:
        raise RuntimeError("Cannot snapshot an active expert iterator")
    if cache.device.type == "cuda":
        torch.cuda.synchronize(cache.device)
    return {
        "main": tuple(info.uid if info is not None else None for info in cache.main_infos),
        "offloaded": tuple(info.uid if info is not None else None for info in cache.offloaded_infos),
        "groups": {key: (tuple(group.main_infos), tuple(group.offloaded_infos))
                   for key, group in cache.group_infos.items()},
    }


def reset_cache(cache, initial):
    """Restore initial cache state outside timing using only upstream swaps.

    Residency, slot indices, and both LRU orders are restored. Buffer allocation
    and upstream cache code remain untouched; no full host snapshot is retained.
    """
    if cache.active:
        raise RuntimeError("Cannot reset an active expert iterator")
    if (len(cache.main_infos) != len(initial["main"])
            or len(cache.offloaded_infos) != len(initial["offloaded"])):
        raise ValueError("Upstream baseline uses fixed cache capacity")
    if cache.device.type == "cuda":
        torch.cuda.synchronize(cache.device)
    for index, uid in enumerate(initial["main"]):
        if uid is None:
            if cache.main_infos[index] is not None:
                raise RuntimeError("An unused historical slot became occupied")
            continue
        wanted = cache.registered_experts[uid]
        if wanted.offloaded:
            cache._swap(wanted, cache.main_infos[index])
        elif wanted.index != index:
            other = wanted.index
            cache.main_modules[index], cache.main_modules[other] = cache.main_modules[other], cache.main_modules[index]
            cache.main_infos[index], cache.main_infos[other] = cache.main_infos[other], cache.main_infos[index]
            cache.main_infos[index].index = index
            cache.main_infos[other].index = other
    for index, uid in enumerate(initial["offloaded"]):
        if uid is None:
            if cache.offloaded_infos[index] is not None:
                raise RuntimeError("An unused historical host slot became occupied")
            continue
        other = cache.registered_experts[uid].index
        if other != index:
            cache.offloaded_storages[index], cache.offloaded_storages[other] = cache.offloaded_storages[other], cache.offloaded_storages[index]
            cache.offloaded_infos[index], cache.offloaded_infos[other] = cache.offloaded_infos[other], cache.offloaded_infos[index]
            cache.offloaded_infos[index].index = index
            cache.offloaded_infos[other].index = other
    for key, (main, offloaded) in initial["groups"].items():
        group = cache.group_infos[key]
        group.main_infos = OrderedDict((uid, cache.registered_experts[uid]) for uid in main)
        group.offloaded_infos = OrderedDict((uid, cache.registered_experts[uid]) for uid in offloaded)
        group.hits = group.misses = 0
    if cache.device.type == "cuda":
        torch.cuda.synchronize(cache.device)


def cache_stats(cache):
    return {"cache_hits": sum(group.hits for group in cache.group_infos.values()),
            "cache_misses": sum(group.misses for group in cache.group_infos.values())}
