"""Checkpoint/model adapters for the retained SpecMoEOff adaptation.

Only model definitions, routing arithmetic, and flat expert storage are reused
from the artifact. Cache policy is supplied explicitly by the caller.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def load_component(module, state_path, weight_map, prefix=""):
    """Load exact tensors from both AE split files and standard HF shards.

    The AE expert files have local keys (e.g. gate_proj.weight), while the
    index retains full model paths. Ordinary HF files have full keys.
    Missing trunk weights fail instead of leaving random parameters behind.
    """
    from safetensors import safe_open
    groups = defaultdict(list)
    for local_name in module.state_dict():
        full_name = prefix + local_name
        if full_name not in weight_map:
            raise KeyError(f"Checkpoint index is missing {full_name}")
        groups[weight_map[full_name]].append((local_name, full_name))
    for filename, names in groups.items():
        with safe_open(str(Path(state_path) / filename), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            values = {}
            for local_name, full_name in names:
                key = full_name if full_name in keys else local_name
                if key not in keys:
                    raise KeyError(f"{filename} has neither {full_name} nor {local_name}")
                values[local_name] = handle.get_tensor(key)
            module.load_state_dict(values, strict=False)


class DeepseekGate(nn.Module):
    """The original adapted target's greedy router, with its four-value API."""
    def __init__(self, config):
        super().__init__()
        if config.topk_method != "greedy" or config.scoring_func != "softmax":
            raise ValueError("This DeepSeek adapter supports greedy softmax routing")
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            weights = weights * self.routed_scaling_factor
        return indices, weights, None, logits


def build_target(case, device, cache_cls=None):
    """Return (target, cache, metadata) without constructing a draft model."""
    from streamlined_execution_engine.expert_storage import ExpertWrapper
    from streamlined_execution_engine.expert_reorder import (
        Deepseekv2BlockSparseTop2MLP, PhiMoEBlockSparseTop2MLP,
        QwenmoeBlockSparseTop2MLP, SparseMoeWrapperDeepseekv2,
        SparseMoeWrapperPhimoe, SparseMoeWrapperShared,
    )
    from streamlined_execution_engine.tensor_utils import with_default_dtype
    if cache_cls is None:
        from .cache import ExpertCache
        cache_cls = ExpertCache
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Full baseline model execution requires a CUDA device")
    state_path = Path(case.state_path).expanduser().resolve()
    with (state_path / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    if case.name == "dsv2lite":
        from models.target.configuration_deepseek import DeepseekV2Config as Config
        from models.target.modeling_deepseek import DeepseekV2ForCausalLM as Model
        config = Config.from_pretrained(str(state_path))
        num_experts = config.n_routed_experts
        layers = [i for i in range(config.num_hidden_layers)
                  if i >= config.first_k_dense_replace and i % config.moe_layer_freq == 0]
        field = "n_routed_experts"
        def make_expert():
            return Deepseekv2BlockSparseTop2MLP(config, intermediate_size=config.moe_intermediate_size)
    elif case.name == "qwen2moe":
        from transformers.models.qwen2_moe import Qwen2MoeConfig as Config, Qwen2MoeForCausalLM as Model
        config = Config.from_pretrained(str(state_path), attn_implementation="eager")
        num_experts = config.num_experts
        if config.norm_topk_prob:
            raise ValueError("This Qwen adapter requires norm_topk_prob=False")
        layers = [i for i in range(config.num_hidden_layers)
                  if i not in config.mlp_only_layers and (i + 1) % config.decoder_sparse_step == 0]
        field = "num_experts"
        def make_expert():
            return QwenmoeBlockSparseTop2MLP(config)
    elif case.name == "phimoe":
        from transformers.models.phimoe import PhimoeConfig as Config, PhimoeForCausalLM as Model
        config = Config.from_pretrained(str(state_path), attn_implementation="eager")
        num_experts = config.num_local_experts
        layers = list(range(config.num_hidden_layers))
        field = "num_local_experts"
        def make_expert():
            return PhiMoEBlockSparseTop2MLP(config)
    else:
        raise ValueError(f"Unsupported baseline model: {case.name}")
    offload = int(case.offload_per_layer)
    if not 0 <= offload < num_experts:
        raise ValueError("offload_per_layer must leave at least one resident expert per MoE layer")
    if num_experts != case.num_experts:
        raise ValueError("Model case expert count differs from checkpoint configuration")
    stripped = deepcopy(config)
    setattr(stripped, field, 0)
    stripped.torch_dtype = torch.float16
    stripped._attn_implementation = "eager"
    with device, with_default_dtype(torch.float16):
        model = Model(stripped)
        for index in layers:
            layer = model.model.layers[index]
            if case.name == "dsv2lite":
                layer.mlp = SparseMoeWrapperDeepseekv2(
                    config, index, DeepseekGate(config),
                    Deepseekv2BlockSparseTop2MLP(config, intermediate_size=config.n_shared_experts * config.moe_intermediate_size),
                    None,
                )
            elif case.name == "qwen2moe":
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
                layer.mlp = SparseMoeWrapperShared(
                    config, index, nn.Linear(config.hidden_size, num_experts, bias=False),
                    Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size),
                    nn.Linear(config.hidden_size, 1, bias=False), None,
                )
            else:
                layer.block_sparse_moe = SparseMoeWrapperPhimoe(
                    config, index, nn.Linear(config.hidden_size, num_experts, bias=False), None,
                )
    model.config = model.model.config = config
    load_component(model, state_path, weight_map)

    def make_module():
        return ExpertWrapper(make_expert().half(), device)

    cache = cache_cls(make_module=make_module,
                      main_size=len(layers) * (num_experts - offload),
                      offload_size=len(layers) * offload,
                      buffer_size=case.buffer_size)
    for index in layers:
        layer = model.model.layers[index]
        name = "block_sparse_moe" if case.name == "phimoe" else "mlp"
        getattr(layer, name).experts = cache
        for expert_index in range(num_experts):
            expert = make_expert().half()
            prefix = f"model.layers.{index}.{name}.experts.{expert_index}."
            load_component(expert, state_path, weight_map, prefix)
            wrapper = ExpertWrapper(expert, device)
            cache.add_expert((index, expert_index), wrapper, eviction_group=index,
                             offload=expert_index < offload)
            del wrapper, expert
    if hasattr(cache, "seal_initial_state"):
        cache.seal_initial_state()
    model.eval()
    if case.name == "qwen2moe":
        from initialization.attention import validate_qwen_attention
        validate_qwen_attention(model)
    torch.cuda.synchronize(device)
    metadata = {
        "model": case.name, "target_dtype": "float16", "moe_layers": len(layers),
        "resident_experts_per_layer": num_experts - offload,
        "offloaded_experts_per_layer": offload, "global_transfer_buffers": case.buffer_size,
        "resident_expert_slots": len(layers) * (num_experts - offload),
        "offloaded_expert_slots": len(layers) * offload,
        "expert_reduction_order": "expert_id", "target_attention": "eager",
    }
    return model, cache, metadata
