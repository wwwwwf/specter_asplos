"""Shared checkpoint format support; no shared draft/target parameter aliasing."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer
from initialization.model_loader import ModelCase, register_qwen_fused

def load_models(
    case: ModelCase,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, torch.nn.Module]:
    from gptqmodel import GPTQModel
    from initialization.checkpoint_layout import localize_checkpoint
    case = replace(case, quant_path=localize_checkpoint(case.quant_path, case.name))
    from .model_builder import build_target
    from .cache import ExpertCache
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

    target_model, _, _ = build_target(case, device, cache_cls=ExpertCache)
    quant_model.eval()
    target_model.eval()
    return tokenizer, quant_model, target_model
