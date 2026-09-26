"""Stable Specter kernel API; implementations are compiled in this tree."""
import torch
from .build import load_extension

def moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False):
    padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        padded = (padded + block_size - 1) // block_size * block_size
    ids = torch.full((padded,), topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    experts = torch.zeros(((padded + block_size - 1)//block_size,), dtype=torch.int32, device=topk_ids.device)
    count = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    load_extension().align_tokens(topk_ids, num_experts, block_size, ids, experts, count)
    if expert_map is not None:
        experts = expert_map[experts]
    return ids, experts, count

def moe_wna16_marlin_gemm(input, output, b_qweight, b_scales, global_scale,
        b_qzeros, g_idx, perm, workspace, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, is_ep, b_q_type, size_m, size_n, size_k,
        is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float):
    from .scalar_type import scalar_types
    if b_q_type.id != scalar_types.uint4b8.id:
        raise ValueError('Specter local build supports uint4b8 weights only')
    return load_extension().moe_gemm(input, output, b_qweight, b_scales,
        global_scale, b_qzeros, g_idx, perm, workspace, sorted_token_ids,
        expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
        top_k, mul_topk_weights, is_ep, b_q_type.id, size_m, size_n, size_k,
        is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float)

def moe_sum(input, output):
    return load_extension().sum_experts(input, output)

def gptq_marlin_repack(weight, perm, size_k, size_n, num_bits, pack_bits=32):
    return load_extension().repack(weight, perm, size_k, size_n, num_bits, pack_bits)
