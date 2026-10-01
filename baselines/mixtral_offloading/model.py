"""Historical author DeepSeek adaptation of Mixtral-Offloading.

The expert dispatch class is extracted without changes in legacy_dispatch.py.
This file adapts checkpoint names, FP16 storage, and the current target API.
The vendored upstream cache retains the historical demand/swap mechanism.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from models.target.modeling_deepseek import MoEGate
from .legacy_dispatch import SparseMoeWrapperDeepseekv2


class FP16ExpertWrapper(nn.Module):
    """Flat raw storage needed by upstream swapping, for unquantized weights.

    Upstream MixtralExpertWrapper only packs HQQ w1/w2/w3 metadata. This adapter
    changes the storage format, without changing the expert's forward method.
    Packing occurs only while loading, outside inference timing.
    """
    def __init__(self, expert, device):
        super().__init__()
        self.expert = expert.half()
        parameters = list(self.expert.parameters())
        self.storage = torch.UntypedStorage(sum(p.numel() * p.element_size() for p in parameters), device=device)
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                size = parameter.numel() * parameter.element_size()
                view = torch.as_tensor(self.storage[offset:offset + size], dtype=parameter.dtype,
                                       device=device).view(parameter.shape)
                view.copy_(parameter)
                parameter.data = view
                offset += size

    def forward(self, hidden_states):
        return self.expert(hidden_states)


class LegacyGate(MoEGate):
    """Bridge the local gate's three outputs to the historical four-output API.

    Router logits were only returned for diagnostics in the historical wrapper;
    the target model consumes the MoE output tensor. Routing arithmetic remains
    the DeepSeek reference implementation, including FP32 routing weights.
    """
    def forward(self, hidden_states):
        selected, weights, auxiliary = super().forward(hidden_states)
        return selected, weights, auxiliary, None


class DeepseekLegacyMoe(SparseMoeWrapperDeepseekv2):
    """Preserve historical execution and unwrap its obsolete diagnostic tuple."""
    def forward(self, hidden_states):
        output, _router_logits = super().forward(hidden_states)
        return output



def _load_component(module, state_path, weight_map, prefix=''):
    """Read only named parameters, supporting local-key and full-key shards."""
    from safetensors import safe_open
    groups = defaultdict(list)
    for local_name in module.state_dict():
        full_name = prefix + local_name
        if full_name not in weight_map:
            raise KeyError(f'Checkpoint index is missing {full_name}')
        groups[weight_map[full_name]].append((local_name, full_name))
    for filename, names in groups.items():
        with safe_open(str(Path(state_path) / filename), framework='pt', device='cpu') as source:
            keys = set(source.keys())
            tensors = {}
            for local_name, full_name in names:
                key = full_name if full_name in keys else local_name
                if key not in keys:
                    raise KeyError(f'{filename} is missing {full_name} / {local_name}')
                tensors[local_name] = source.get_tensor(key)
            module.load_state_dict(tensors, strict=False)


def _construct_target(config, device):
    """Construct attention/trunk plus historical-dispatch MoEs without weights."""
    from models.target.modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2MLP
    layers = [index for index in range(config.num_hidden_layers)
              if index >= config.first_k_dense_replace and index % config.moe_layer_freq == 0]
    stripped = deepcopy(config)
    stripped.n_routed_experts = 0
    stripped.torch_dtype = torch.float16
    stripped._attn_implementation = 'eager'
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device(device):
            target = DeepseekV2ForCausalLM(stripped)
            for index in layers:
                shared = (DeepseekV2MLP(config, intermediate_size=config.n_shared_experts * config.moe_intermediate_size)
                          if config.n_shared_experts is not None else None)
                target.model.layers[index].mlp = DeepseekLegacyMoe(config, index, LegacyGate(config), shared, None)
    finally:
        torch.set_default_dtype(previous_dtype)
    target.config = target.model.config = config
    return target, layers


def build_target(case, device, cache_cls=None):
    """Return target/cache/metadata with the exact upstream demand/swap methods."""
    if case.name != 'dsv2lite':
        raise ValueError('The historical FP16 adapter currently supports DeepSeek-V2-Lite only')
    from models.target.configuration_deepseek import DeepseekV2Config
    from models.target.modeling_deepseek import DeepseekV2MLP
    from .upstream.src.expert_cache import ExpertCache as UpstreamExpertCache
    if cache_cls is None:
        cache_cls = UpstreamExpertCache
    if (cache_cls.load_experts is not UpstreamExpertCache.load_experts
            or cache_cls._swap is not UpstreamExpertCache._swap):
        raise TypeError('Original Mixtral-Offloading demand and swap methods must remain unchanged')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Full checkpoint execution requires CUDA; CPU tests use the smaller construction helpers')
    state_path = Path(case.state_path).expanduser().resolve()
    config = DeepseekV2Config.from_pretrained(str(state_path))
    if getattr(config, 'ep_size', 1) != 1:
        raise ValueError('Only single-device DeepSeek targets are supported')
    experts = config.n_routed_experts
    if experts != case.num_experts or not 0 <= case.offload_per_layer < experts:
        raise ValueError('Expert count/residency mismatch')
    if case.offload_per_layer and case.buffer_size < 2:
        raise ValueError('Original offloading pipeline needs at least two transfer buffers')
    with (state_path / 'model.safetensors.index.json').open(encoding='utf-8') as source:
        weight_map = json.load(source)['weight_map']
    target, layers = _construct_target(config, device)
    if len(layers) != case.layer_num:
        raise ValueError('Actual routed-layer count differs from model case')
    _load_component(target, state_path, weight_map)

    def make_expert():
        return DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size).half()

    def make_module():
        return FP16ExpertWrapper(make_expert(), device)

    resident = experts - case.offload_per_layer
    # Historical build_offload_model allocates by all hidden layers, including
    # DeepSeek's first dense layer. Keep its spare slots for memory fidelity;
    # only registered MoE-layer slots participate in the cache's LRU groups.
    cache = cache_cls(make_module=make_module, main_size=config.num_hidden_layers * resident,
                      offload_size=config.num_hidden_layers * case.offload_per_layer, buffer_size=case.buffer_size)
    for position, index in enumerate(layers, start=1):
        target.model.layers[index].mlp.experts = cache
        for expert_id in range(experts):
            expert = make_expert()
            _load_component(expert, state_path, weight_map, f'model.layers.{index}.mlp.experts.{expert_id}.')
            wrapped = FP16ExpertWrapper(expert, device)
            cache.add_expert((index, expert_id), wrapped, eviction_group=index,
                             offload=expert_id < case.offload_per_layer)
            del wrapped, expert
        print(f'[mixtral-offloading] Loaded MoE layer {position}/{len(layers)} '
              f'(model layer {index}, {experts} experts)', flush=True)
    target.eval()
    torch.cuda.synchronize(device)
    metadata = {
        'model': case.name, 'target_dtype': 'float16', 'target_attention': 'DeepSeek eager',
        'upstream_commit': 'ce545188b804238f0b23a59fc45e6a8f8b390c40',
        'cache_class': f'{type(cache).__module__}.{type(cache).__name__}',
        'original_demand_and_swap_methods': True,
        'moe_dispatch': 'historical one_hot/nonzero/precomputed where/GPU-tensor gather/index_add_',
        'dispatch_source': json.loads((Path(__file__).parent / 'legacy_dispatch.provenance.json').read_text(encoding='utf-8')),
        'routing_weight_dtype': 'float32 until weighted output is cast for accumulation',
        'expert_reduction_order': 'upstream cache yield order',
        'routed_expert_kernel': 'unfused DeepseekV2MLP gate_proj, up_proj, down_proj',
        'shared_expert_kernel': 'unfused DeepseekV2MLP after routed experts',
        'precision_adaptation': 'FP16 DeepSeek checkpoint instead of upstream HQQ-quantized Mixtral',
        'moe_layers': len(layers), 'cache_group_count': len(layers), 'expert_slot_bytes': cache.module_size,
        'resident_per_layer': resident, 'resident_experts_per_layer': resident,
        'offloaded_experts_per_layer': case.offload_per_layer,
        'resident_expert_slots': sum(len(group.main_infos) for group in cache.group_infos.values()),
        'offloaded_expert_slots': sum(len(group.offloaded_infos) for group in cache.group_infos.values()),
        'allocated_gpu_expert_slots': len(cache.main_modules),
        'allocated_host_expert_slots': len(cache.offloaded_storages),
        'unused_gpu_expert_slots': sum(info is None for info in cache.main_infos),
        'unused_host_expert_slots': sum(info is None for info in cache.offloaded_infos),
        'global_transfer_buffers': case.buffer_size,
    }
    return target, cache, metadata
