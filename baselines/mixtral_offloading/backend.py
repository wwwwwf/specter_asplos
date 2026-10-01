"""Greedy AR execution with one observable commitment per generated token."""
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache
from .original_cache import cache_stats, reset_cache, snapshot_initial


@dataclass
class Backend:
    tokenizer: object
    target: object
    manager: object
    metadata: dict = field(default_factory=dict)
    draft: object = None
    initial_cache: dict = field(default_factory=dict)

    def reset(self):
        reset_cache(self.manager, self.initial_cache)

    @torch.inference_mode()
    def generate(self, input_ids, num_tokens, *, seed=42, clock=None, target_only=False):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("The AR baseline requires a nonempty batch of exactly one prompt")
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        # A fresh KV cache per prompt. No EOS stopping: exactly num_tokens are
        # committed so token-timing denominators match the shared benchmark.
        output_ids = torch.empty((1, input_ids.shape[1] + num_tokens),
                                 dtype=input_ids.dtype, device=input_ids.device)
        output_ids[:, :input_ids.shape[1]] = input_ids
        past_key_values = DynamicCache()
        current = input_ids
        for step in range(num_tokens):
            result = self.target(input_ids=current, past_key_values=past_key_values,
                                 use_cache=True, return_dict=True)
            # Match the artifact's native-precision normalization before
            # greedy selection, including rounding behavior near ties.
            next_token = result.logits[:, -1:].softmax(dim=-1).argmax(dim=-1)
            length = input_ids.shape[1] + step + 1
            output_ids[:, length - 1:length] = next_token
            if clock is not None:
                clock.commit(length)
            past_key_values = result.past_key_values
            if not isinstance(past_key_values, DynamicCache):
                raise TypeError("The adapted target must preserve DynamicCache")
            current = next_token
        return {"token_ids": output_ids, "stats": cache_stats(self.manager)}


def load_backend(case, device):
    from transformers import AutoTokenizer
    from .model import build_target
    if case.offload_per_layer and case.buffer_size < 2:
        raise ValueError("Mixtral-Offloading needs --buffer-size >= 2; historical runs used 4")
    with torch.inference_mode():
        target, manager, metadata = build_target(case, device)
    tokenizer = AutoTokenizer.from_pretrained(case.state_path, trust_remote_code=True)
    metadata.update({"baseline": "mixtral_offloading", "algorithm": "greedy_autoregressive",
                     "cache_policy": "layer_local_lru", "loading": "on_demand_buffered_swap",
                     "upstream_commit": "ce545188b804238f0b23a59fc45e6a8f8b390c40",
                     "adaptation": "Historical author DeepSeek FP16 dispatch; original Mixtral-Offloading demand/swap cache",
                     "predictive_prefetch": False, "cache_reset": "initial_residency_and_lru",
                     "kv_cache": "DynamicCache", "greedy_selection": "native_softmax_argmax"})
    metadata.update(cache_metadata(manager))
    return Backend(tokenizer, target, manager, metadata, initial_cache=snapshot_initial(manager))


def cache_metadata(manager):
    groups = len(manager.group_infos)
    capacities = {len(group.main_infos) for group in manager.group_infos.values()}
    if len(capacities) != 1:
        raise ValueError("Each MoE layer must have the same resident capacity")
    resident = capacities.pop()
    return {"expert_slot_bytes": manager.module_size, "cache_group_count": groups,
            "resident_per_layer": resident, "resident_experts_per_layer": resident,
            "resident_expert_slots": sum(len(group.main_infos) for group in manager.group_infos.values()),
            "offloaded_expert_slots": sum(len(group.offloaded_infos) for group in manager.group_infos.values()),
            "allocated_gpu_expert_slots": len(manager.main_modules),
            "allocated_host_expert_slots": len(manager.offloaded_storages),
            "unused_gpu_expert_slots": sum(info is None for info in manager.main_infos),
            "unused_host_expert_slots": sum(info is None for info in manager.offloaded_infos),
            "cache_gpu_storage_bytes": manager.module_size * (
                len(manager.main_modules) + len(manager.device_expert_buffers))}
