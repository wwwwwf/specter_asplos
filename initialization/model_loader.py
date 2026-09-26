"""Shared model loading and routing setup; no experiment entrypoints."""
from __future__ import annotations
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

@dataclass(frozen=True)
class ModelCase:
    name: str
    state_path: str
    quant_path: str
    layer_num: int
    num_experts: int
    router_topk: int
    gamma: int
    offload_per_layer: int
    buffer_size: int
    skip_first_layer: bool = False


def model_cases(config=None) -> dict[str, ModelCase]:
    from configuration import get_config
    config = config or get_config()
    specs = {
        'dsv2lite': (26, 64, 6, 16, 48, 32, True),
        'qwen2moe': (24, 60, 4, 8, 15, 32, False),
        'phimoe': (32, 16, 2, 16, 15, 0, False),
    }
    return {
        name: ModelCase(name, str(config.path('models.' + name, 'target')),
                        str(config.path('models.' + name, 'draft')), *spec)
        for name, spec in specs.items()
    }


def register_qwen_fused(quant_path: str) -> None:
    sys.path.insert(0, str(Path(quant_path).resolve()))
    from configuration_qwen2_moe_fused import FusedQwen2MoeConfig
    from modeling_qwen2_moe_fused import FusedQwen2MoeForCausalLM

    try:
        AutoConfig.register("qwen2_moe_fused", FusedQwen2MoeConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(
            FusedQwen2MoeConfig, FusedQwen2MoeForCausalLM
        )
    except ValueError:
        pass


def load_models(
    case: ModelCase,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, torch.nn.Module]:
    from gptqmodel import GPTQModel
    from initialization.checkpoint_layout import localize_checkpoint
    case = replace(case, quant_path=localize_checkpoint(case.quant_path, case.name))
    from initialization.target_loader import build_offload_model
    from streamlined_execution_engine.draft_fusion import (
        weight_loader_deepseekv2,
        weight_loader_phimoe,
        weight_loader_qwenmoe,
    )

    for path in (case.state_path, case.quant_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    if case.name == "qwen2moe":
        register_qwen_fused(case.quant_path)

    tokenizer = AutoTokenizer.from_pretrained(
        case.state_path, trust_remote_code=True
    )
    quant_model = GPTQModel.load(
        case.quant_path,
        device_map=str(device),
        backend="marlin",
        trust_remote_code=True,
        **({"attn_implementation": "eager"} if case.name == "qwen2moe" else {}),
    )

    if case.name == "qwen2moe":
        from initialization.attention import validate_qwen_attention
        validate_qwen_attention(quant_model.model)

    if case.name == "qwen2moe":
        weight_loader_qwenmoe(quant_model, None, case.quant_path)
    elif case.name == "phimoe":
        weight_loader_phimoe(quant_model, None, case.quant_path)
    elif case.name == "dsv2lite":
        has_fused = hasattr(
            quant_model.model.model.layers[1].mlp, "fusedexperts"
        )
        if has_fused:
            weight_loader_deepseekv2(quant_model, None, case.quant_path)
        else:
            raise RuntimeError(
                "DeepSeek draft has no mlp.fusedexperts. "
                "Use the supported fused GPTQ checkpoint/package."
            )
    else:
        raise ValueError(case.name)

    target_model = build_offload_model(
        case.state_path,
        case.state_path,
        device,
        case.offload_per_layer,
        case.buffer_size,
    )
    quant_model.eval()
    target_model.eval()
    return tokenizer, quant_model, target_model




def install_hooks(
    case: ModelCase,
    quant_model: torch.nn.Module,
    target_model: torch.nn.Module,
    controller: Any,
) -> list[Any]:
    from predictive_io_orchestrator.routing_hooks import (
        set_hook_4_prefetch_noblock_dsv2lite,
        set_hook_4_prefetch_noblock_phimoe,
        set_hook_4_prefetch_noblock_qwenmoe,
    )

    hooks: list[Any] = []
    if case.name == "qwen2moe":
        set_hook_4_prefetch_noblock_qwenmoe(
            quant_model, target_model, hooks, controller
        )
    elif case.name == "dsv2lite":
        set_hook_4_prefetch_noblock_dsv2lite(
            quant_model, target_model, hooks, controller
        )
    elif case.name == "phimoe":
        set_hook_4_prefetch_noblock_phimoe(
            quant_model, target_model, hooks, controller
        )
    return hooks
