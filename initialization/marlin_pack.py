import torch
from streamlined_execution_engine.kernels.ops import gptq_marlin_repack as repack_kernel

def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_moe_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output

"""
    modified from vllm fusedmoe
    layer.w13_qweight,
    layer.w13_g_idx_sort_indices,
    layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
    layer.w13_qweight.shape[2],
    self.quant_config.quant_type.size_bits,
"""

def repack(q_weight, perm, num_experts, hidden_dim, fused_inter_dim, num_bits):
    assert hidden_dim % 16 == 0
    output = torch.empty((num_experts, hidden_dim // 16, fused_inter_dim * (num_bits // 2)),
                         device=q_weight.device,
                         dtype=q_weight.dtype)
    for e in range(num_experts):
        output[e] = repack_kernel(q_weight[e], perm[e],
                                                    hidden_dim, fused_inter_dim, num_bits, 32)

    return output

