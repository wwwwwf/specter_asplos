import os
import re
import torch
from torch import nn
from safetensors.torch import load_file
from initialization.marlin_pack import marlin_moe_permute_scales, repack
from typing import Union

def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"

    # update tensor shape and stride
    dst.as_strided_(src.shape, src.stride())

    # If not the same underlying storage move tensor data
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
        del src
        
def replace_parameter(mod: torch.nn.Module, name: str,
                      new: Union[torch.Tensor, torch.nn.Parameter]) -> None:

    old = getattr(mod, name)
    if type(old) is type(new) and old.dtype == new.dtype and \
        old.untyped_storage().nbytes() == new.untyped_storage().nbytes():
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name,
                               torch.nn.Parameter(new, requires_grad=False))

def parse_param_name(name):
    parts = name.split(".")
    
    layer_idx = None
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                continue

    expert_idx = None
    for i, part in enumerate(parts):
        if part == "experts" and i + 1 < len(parts):
            try:
                expert_idx = int(parts[i + 1])
            except ValueError:
                continue

    module = None
    for part in parts:
        if part in ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"]:
            module = part
            break

    return layer_idx, expert_idx, module

def process_expert_weights_qwenmoe(model, name, param):
    # for qwen2moe
    if "shared" in name:
        return
    if name.endswith("g_idx") or name.endswith("qzeros"):
        return
    layer_idx, expert_idx, module = parse_param_name(name)
    if name.endswith("qweight"):
        #print("find qweight")
        param = param.to(torch.int32)
        if module == "w1" or module == "gate_proj": 
            #print("find w1")
            fused_qweight = model.model.model.layers[layer_idx].mlp.fusedexperts.w13_q_weight[expert_idx,:,:param.shape[1]]
            fused_qweight.copy_(param)
        elif module == "w3" or module == "up_proj":
            fused_qweight = model.model.model.layers[layer_idx].mlp.fusedexperts.w13_q_weight[expert_idx,:,param.shape[1]:]
            fused_qweight.copy_(param)
        elif module == "w2" or module == "down_proj":
            fused_qweight = model.model.model.layers[layer_idx].mlp.fusedexperts.w2_q_weight[expert_idx]
            fused_qweight.copy_(param)
    elif name.endswith("scales"):
        param_bf16 = param.to(torch.bfloat16) 
        if module == "w1" or module == "gate_proj": 
            fused_qweight = model.model.model.layers[layer_idx].mlp.fusedexperts.w13_scales[expert_idx,:,:param.shape[1]]
            fused_qweight.copy_(param_bf16)
        elif module == "w3" or module == "up_proj":
            fused_qweight = model.model.model.layers[layer_idx].mlp.fusedexperts.w13_scales[expert_idx,:,param.shape[1]:]
            fused_qweight.copy_(param_bf16)
        elif module == "w2" or module == "down_proj":
            fused_qweight = model.model.model.layers[layer_idx].mlp.fusedexperts.w2_scales[expert_idx]
            fused_qweight.copy_(param_bf16)

def process_expert_weights_phimoe(model, name, param):
    # for qwen2moe
    if "shared" in name:
        return
    if name.endswith("g_idx") or name.endswith("qzeros"):
        return
    layer_idx, expert_idx, module = parse_param_name(name)
    if name.endswith("qweight"):
        #print("find qweight")
        param = param.to(torch.int32)
        if module == "w1" or module == "gate_proj": 
            #print("find w1")
            fused_qweight = model.model.model.layers[layer_idx].block_sparse_moe.fusedexperts.w13_q_weight[expert_idx,:,:param.shape[1]]
            fused_qweight.copy_(param)
        elif module == "w3" or module == "up_proj":
            fused_qweight = model.model.model.layers[layer_idx].block_sparse_moe.fusedexperts.w13_q_weight[expert_idx,:,param.shape[1]:]
            fused_qweight.copy_(param)
        elif module == "w2" or module == "down_proj":
            fused_qweight = model.model.model.layers[layer_idx].block_sparse_moe.fusedexperts.w2_q_weight[expert_idx]
            fused_qweight.copy_(param)
    elif name.endswith("scales"):
        param_bf16 = param.to(torch.bfloat16) 
        if module == "w1" or module == "gate_proj": 
            fused_qweight = model.model.model.layers[layer_idx].block_sparse_moe.fusedexperts.w13_scales[expert_idx,:,:param.shape[1]]
            fused_qweight.copy_(param_bf16)
        elif module == "w3" or module == "up_proj":
            fused_qweight = model.model.model.layers[layer_idx].block_sparse_moe.fusedexperts.w13_scales[expert_idx,:,param.shape[1]:]
            fused_qweight.copy_(param_bf16)
        elif module == "w2" or module == "down_proj":
            fused_qweight = model.model.model.layers[layer_idx].block_sparse_moe.fusedexperts.w2_scales[expert_idx]
            fused_qweight.copy_(param_bf16)



def weight_loader_phimoe(model, param_list, param_path):
    # Collect all .safetensors files.
    files = [os.path.join(param_path, f) for f in os.listdir(param_path) if f.endswith(".safetensors")]
    
    # Merge the weights.
    state_dict = {}
    for f in files:
        state_dict.update(load_file(f))
    
    # # retype backward int32
    # for layer in model.model.model.layers:
    #     new_tensor = layer.block_sparse_moe.fusedexperts.w13_q_weight.to(torch.int32)
    #     layer.block_sparse_moe.fusedexperts.w13_q_weight = nn.Parameter(new_tensor, requires_grad=False)
    #     new_tensor = layer.block_sparse_moe.fusedexperts.w2_q_weight.to(torch.int32)
    #     layer.block_sparse_moe.fusedexperts.w2_q_weight = nn.Parameter(new_tensor, requires_grad=False)

    # layer = model.model.model.layers[1]
    # print(layer.block_sparse_moe.fusedexperts.w13_q_weight[0,:10,:10])
    # print(layer.block_sparse_moe.fusedexperts.w13_q_weight[0,:10,6400:6410])
    #     # Process each parameter.
    #     # load raw params
    #     print(f"loading layer: {layer.layer_idx}")    
    #     print(layer.self_attn.q_proj.qweight[:10,:10])
    #     print(layer.block_sparse_moe.gate.qweight[0:10, :])
    for name, param in state_dict.items():
        if "expert" in name:
            #print("find experts")
            process_expert_weights_phimoe(model, name, param)

    #check
    # layer = model.model.model.layers[1]
    # print(layer.block_sparse_moe.fusedexperts.w13_q_weight[0,:10,:10])
    # print(layer.block_sparse_moe.fusedexperts.w13_q_weight[0,:10,6400:6410])
    
    pack_factor = 8
    num_bits = 4
    group_size = 128
    num_experts = model.model.model.layers[1].block_sparse_moe.fusedexperts.w13_q_weight.shape[0]
    device = model.model.model.layers[1].block_sparse_moe.fusedexperts.w13_q_weight.device
    # print(f"device {device}")
    w13_hidden_dim = model.model.model.layers[1].block_sparse_moe.fusedexperts.w13_q_weight.shape[1] * pack_factor
    w13_fused_inter_dim = model.model.model.layers[1].block_sparse_moe.fusedexperts.w13_q_weight.shape[2]
    w2_hidden_dim = model.model.model.layers[1].block_sparse_moe.fusedexperts.w2_q_weight.shape[1] * pack_factor
    w2_inter_dim = model.model.model.layers[1].block_sparse_moe.fusedexperts.w2_q_weight.shape[2]
     


    for layer in model.model.model.layers:
        # print(layer.w13_g_idx.shape)
        # print(layer.w13_g_idx)
        
        # dummy params
        
        invalid_param_keys = ["w13_qzeros", "w2_qzeros"]
        for key in invalid_param_keys:
            # param = torch.nn.Parameter(torch.empty((0, ),
            #                             dtype=torch.int32,
            #                             device=device),
            #                             requires_grad=False)
            param = None
            layer.block_sparse_moe.fusedexperts.register_parameter(key, param)
        
        dummy_param_keys = ["w13_g_idx_sort_indices", "w2_g_idx_sort_indices", "w13_g_idx", "w2_g_idx"]
        for key in dummy_param_keys:
            param = torch.nn.Parameter(torch.empty((num_experts, 0),
                                        dtype=torch.int32,
                                        device=device),
                                        requires_grad=False)
            layer.block_sparse_moe.fusedexperts.register_parameter(key, param)

        
        w13_q_weight = layer.block_sparse_moe.fusedexperts.w13_q_weight
        w2_q_weight = layer.block_sparse_moe.fusedexperts.w2_q_weight
        w13_scales =  layer.block_sparse_moe.fusedexperts.w13_scales
        w2_scales = layer.block_sparse_moe.fusedexperts.w2_scales
        marlin_w13_q_weight = repack(w13_q_weight, layer.block_sparse_moe.fusedexperts.w13_g_idx_sort_indices, num_experts, w13_hidden_dim, w13_fused_inter_dim, num_bits)
        replace_parameter(layer.block_sparse_moe.fusedexperts, "w13_q_weight", marlin_w13_q_weight)
        marlin_w2_q_weight = repack(w2_q_weight, layer.block_sparse_moe.fusedexperts.w2_g_idx_sort_indices, num_experts, w2_hidden_dim, w2_inter_dim, num_bits)
        replace_parameter(layer.block_sparse_moe.fusedexperts, "w2_q_weight", marlin_w2_q_weight)
        marlin_w13_scales = marlin_moe_permute_scales(w13_scales, w13_scales.shape[1] * group_size, w13_scales.shape[2], group_size)
        replace_parameter(layer.block_sparse_moe.fusedexperts, "w13_scales", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(w2_scales, w2_scales.shape[1] * group_size, w2_scales.shape[2], group_size)
        replace_parameter(layer.block_sparse_moe.fusedexperts, "w2_scales", marlin_w2_scales)
        
    # layer = model.model.model.layers[1]
    # print(layer.block_sparse_moe.fusedexperts.w13_q_weight[0,:10,:10])
    # print(layer.block_sparse_moe.fusedexperts.w13_q_weight[0,:10,12800:12810])
    # print(layer.block_sparse_moe.fusedexperts.w13_scales[0,:10,:10])
    # print(layer.block_sparse_moe.fusedexperts.w13_scales[0,:10,6400:6410])
    # exit(0)

def weight_loader_qwenmoe(model, param_list, param_path):
    # Collect all .safetensors files.
    files = [os.path.join(param_path, f) for f in os.listdir(param_path) if f.endswith(".safetensors")]
    
    # Merge the weights.
    state_dict = {}
    for f in files:
        state_dict.update(load_file(f))
    
    # # retype backward int32
    # for layer in model.model.model.layers:
    #     new_tensor = layer.mlp.fusedexperts.w13_q_weight.to(torch.int32)
    #     layer.mlp.fusedexperts.w13_q_weight = nn.Parameter(new_tensor, requires_grad=False)
    #     new_tensor = layer.mlp.fusedexperts.w2_q_weight.to(torch.int32)
    #     layer.mlp.fusedexperts.w2_q_weight = nn.Parameter(new_tensor, requires_grad=False)

    # layer = model.model.model.layers[1]
    # print(layer.mlp.fusedexperts.w13_q_weight[0,:10,:10])
    # print(layer.mlp.fusedexperts.w13_q_weight[0,:10,6400:6410])
    #     # Process each parameter.
    #     # load raw params
    #     print(f"loading layer: {layer.layer_idx}")    
    #     print(layer.self_attn.q_proj.qweight[:10,:10])
    #     print(layer.mlp.gate.qweight[0:10, :])
    for name, param in state_dict.items():
        if "expert" in name:
            #print("find experts")
            process_expert_weights_qwenmoe(model, name, param)

    #check
    # layer = model.model.model.layers[1]
    # print(layer.mlp.fusedexperts.w13_q_weight[0,:10,:10])
    # print(layer.mlp.fusedexperts.w13_q_weight[0,:10,6400:6410])
    
    pack_factor = 8
    num_bits = 4
    group_size = 128
    num_experts = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.shape[0]
    device = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.device
    # print(f"device {device}")
    w13_hidden_dim = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.shape[1] * pack_factor
    w13_fused_inter_dim = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.shape[2]
    w2_hidden_dim = model.model.model.layers[1].mlp.fusedexperts.w2_q_weight.shape[1] * pack_factor
    w2_inter_dim = model.model.model.layers[1].mlp.fusedexperts.w2_q_weight.shape[2]
     


    for layer in model.model.model.layers:
        # print(layer.w13_g_idx.shape)
        # print(layer.w13_g_idx)
        
        # dummy params
        
        invalid_param_keys = ["w13_qzeros", "w2_qzeros"]
        for key in invalid_param_keys:
            # param = torch.nn.Parameter(torch.empty((0, ),
            #                             dtype=torch.int32,
            #                             device=device),
            #                             requires_grad=False)
            param = None
            layer.mlp.fusedexperts.register_parameter(key, param)
        
        dummy_param_keys = ["w13_g_idx_sort_indices", "w2_g_idx_sort_indices", "w13_g_idx", "w2_g_idx"]
        for key in dummy_param_keys:
            param = torch.nn.Parameter(torch.empty((num_experts, 0),
                                        dtype=torch.int32,
                                        device=device),
                                        requires_grad=False)
            layer.mlp.fusedexperts.register_parameter(key, param)

        
        w13_q_weight = layer.mlp.fusedexperts.w13_q_weight
        w2_q_weight = layer.mlp.fusedexperts.w2_q_weight
        w13_scales =  layer.mlp.fusedexperts.w13_scales
        w2_scales = layer.mlp.fusedexperts.w2_scales
        marlin_w13_q_weight = repack(w13_q_weight, layer.mlp.fusedexperts.w13_g_idx_sort_indices, num_experts, w13_hidden_dim, w13_fused_inter_dim, num_bits)
        replace_parameter(layer.mlp.fusedexperts, "w13_q_weight", marlin_w13_q_weight)
        marlin_w2_q_weight = repack(w2_q_weight, layer.mlp.fusedexperts.w2_g_idx_sort_indices, num_experts, w2_hidden_dim, w2_inter_dim, num_bits)
        replace_parameter(layer.mlp.fusedexperts, "w2_q_weight", marlin_w2_q_weight)
        marlin_w13_scales = marlin_moe_permute_scales(w13_scales, w13_scales.shape[1] * group_size, w13_scales.shape[2], group_size)
        replace_parameter(layer.mlp.fusedexperts, "w13_scales", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(w2_scales, w2_scales.shape[1] * group_size, w2_scales.shape[2], group_size)
        replace_parameter(layer.mlp.fusedexperts, "w2_scales", marlin_w2_scales)
        
    # layer = model.model.model.layers[1]
    # print(layer.mlp.fusedexperts.w13_q_weight[0,:10,:10])
    # print(layer.mlp.fusedexperts.w13_q_weight[0,:10,12800:12810])
    # print(layer.mlp.fusedexperts.w13_scales[0,:10,:10])
    # print(layer.mlp.fusedexperts.w13_scales[0,:10,6400:6410])
    # exit(0)
    
def weight_loader_deepseekv2(model, param_list, param_path):
    # Collect all .safetensors files.
    files = [os.path.join(param_path, f) for f in os.listdir(param_path) if f.endswith(".safetensors")]
    
    # Merge the weights.
    state_dict = {}
    for f in files:
        state_dict.update(load_file(f))
    
    for name, param in state_dict.items():
        if "expert" in name:
            #print("find experts")
            #TODO: Reuse temporarily.
            process_expert_weights_qwenmoe(model, name, param)
    
    pack_factor = 8
    num_bits = 4
    group_size = 128
    num_experts = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.shape[0]
    device = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.device
    # print(f"device {device}")
    w13_hidden_dim = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.shape[1] * pack_factor
    w13_fused_inter_dim = model.model.model.layers[1].mlp.fusedexperts.w13_q_weight.shape[2]
    w2_hidden_dim = model.model.model.layers[1].mlp.fusedexperts.w2_q_weight.shape[1] * pack_factor
    w2_inter_dim = model.model.model.layers[1].mlp.fusedexperts.w2_q_weight.shape[2]
     

    
    for layer in model.model.model.layers:
        # dummy params
        #skip layer 0
        if layer.layer_idx == 0:
            continue
        invalid_param_keys = ["w13_qzeros", "w2_qzeros"]
        for key in invalid_param_keys:
            param = None
            layer.mlp.fusedexperts.register_parameter(key, param)
        
        dummy_param_keys = ["w13_g_idx_sort_indices", "w2_g_idx_sort_indices", "w13_g_idx", "w2_g_idx"]
        for key in dummy_param_keys:
            param = torch.nn.Parameter(torch.empty((num_experts, 0),
                                        dtype=torch.int32,
                                        device=device),
                                        requires_grad=False)
            layer.mlp.fusedexperts.register_parameter(key, param)

        
        w13_q_weight = layer.mlp.fusedexperts.w13_q_weight
        w2_q_weight = layer.mlp.fusedexperts.w2_q_weight
        w13_scales =  layer.mlp.fusedexperts.w13_scales
        w2_scales = layer.mlp.fusedexperts.w2_scales
        marlin_w13_q_weight = repack(w13_q_weight, layer.mlp.fusedexperts.w13_g_idx_sort_indices, num_experts, w13_hidden_dim, w13_fused_inter_dim, num_bits)
        replace_parameter(layer.mlp.fusedexperts, "w13_q_weight", marlin_w13_q_weight)
        marlin_w2_q_weight = repack(w2_q_weight, layer.mlp.fusedexperts.w2_g_idx_sort_indices, num_experts, w2_hidden_dim, w2_inter_dim, num_bits)
        replace_parameter(layer.mlp.fusedexperts, "w2_q_weight", marlin_w2_q_weight)
        marlin_w13_scales = marlin_moe_permute_scales(w13_scales, w13_scales.shape[1] * group_size, w13_scales.shape[2], group_size)
        replace_parameter(layer.mlp.fusedexperts, "w13_scales", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(w2_scales, w2_scales.shape[1] * group_size, w2_scales.shape[2], group_size)
        replace_parameter(layer.mlp.fusedexperts, "w2_scales", marlin_w2_scales)
        