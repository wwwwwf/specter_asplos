import os
import json
from functools import cache
from dataclasses import dataclass
import typing as tp
import subprocess
import torch
from torch import nn
import math
import torch.nn.functional as F
import torch.nn.init as init
import psutil
from transformers import AutoConfig
from transformers.models.qwen2_moe import Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.models.phimoe.configuration_phimoe import PhimoeConfig
from transformers.models.phimoe.modeling_phimoe import PhimoeForCausalLM


# from .configuration_phimoe import PhimoeConfig
# from .modeling_phimoe import PhimoeForCausalLM
from models.target.modeling_deepseek import DeepseekV2ForCausalLM
from models.target.configuration_deepseek import DeepseekV2Config
from safetensors.torch import load_file

from torch import nn
from tqdm.auto import trange



from predictive_io_orchestrator.expert_cache import ExpertCache
from streamlined_execution_engine.expert_storage import ExpertWrapper
from streamlined_execution_engine.expert_reorder import (
    SparseMoeWrapperShared,
    SparseMoeWrapperPhimoe,
    QwenmoeBlockSparseTop2MLP,
    PhiMoEBlockSparseTop2MLP,
    SparseMoeWrapperDeepseekv2,
    Deepseekv2BlockSparseTop2MLP,
)
from streamlined_execution_engine.tensor_utils import with_default_dtype

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

@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int







def make_empty_expert_phimoe(
    model_config: PhimoeConfig
) -> PhiMoEBlockSparseTop2MLP:
    return PhiMoEBlockSparseTop2MLP(
        model_config,
    )

def make_and_load_expert_wrapper_phimoe( #load parameters
    config: PhimoeConfig,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> ExpertWrapper:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.weight"]

    state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = make_empty_expert_phimoe(config)
    expert.load_state_dict(state_dict, strict=True)
    expert.half()
    return ExpertWrapper(expert, device)


def replace_attn_layers_phimoe(
    model: PhimoeForCausalLM,
    config: PhimoeConfig,
    device: torch.device,
) -> None:

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads

    shapes = [
        (hidden_size, num_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (num_heads * head_dim, hidden_size),
    ]


    for layer in model.model.layers:
        layer.block_sparse_moe.gate = nn.Linear(  #Create the routing gate.
            config.hidden_size,
            config.num_local_experts,
            dtype=torch.float16,
            device=device,
            bias=False,
        )

def make_empty_expert_qwenmoe(
    model_config: Qwen2MoeConfig
) -> QwenmoeBlockSparseTop2MLP:
    return QwenmoeBlockSparseTop2MLP(
        model_config,
    )


def make_and_load_expert_wrapper_qwenmoe( #load parameters
    config: Qwen2MoeConfig,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> ExpertWrapper:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.gate_proj.weight"]

    state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = make_empty_expert_qwenmoe(config)
    expert.load_state_dict(state_dict, strict=True)
    expert.half()
    return ExpertWrapper(expert, device)

def replace_attn_layers_qwenmoe(
    model: Qwen2MoeForCausalLM,
    config: Qwen2MoeConfig,
    device: torch.device,
) -> None:

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads
    shared_expert_intermediate_size = config.shared_expert_intermediate_size
    
    shapes = [
        (hidden_size, num_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (num_heads * head_dim, hidden_size),
    ]

    #Create the routing gate.
    for layer in model.model.layers:
        layer.mlp.gate = nn.Linear(  
            config.hidden_size,
            config.num_experts,
            dtype=torch.float16,
            device=device,
            bias=False,
        )
        #SHARE_EXP
        layer.mlp.shared_expert = Qwen2MoeMLP(config, intermediate_size=shared_expert_intermediate_size)
        layer.mlp.shared_expert_gate = torch.nn.Linear(hidden_size, 1, bias=False)

def make_empty_expert_deepseekv2(
    model_config: DeepseekV2Config
) -> Deepseekv2BlockSparseTop2MLP:
    return Deepseekv2BlockSparseTop2MLP(
        model_config, intermediate_size=model_config.moe_intermediate_size
    )
    
def make_and_load_expert_wrapper_deepseekv2( #load parameters
    config: DeepseekV2Config,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> ExpertWrapper:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.gate_proj.weight"]

    state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = make_empty_expert_deepseekv2(config)
    expert.load_state_dict(state_dict, strict=True)
    expert.half()
    return ExpertWrapper(expert, device)

class MoEGate(nn.Module):
    def __init__(self, config, n_routed_experts):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        
        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        aux_loss = None
        return topk_idx, topk_weight, aux_loss, logits


def replace_attn_layers_deepseekv2(
    model: DeepseekV2ForCausalLM,
    config: DeepseekV2Config,
    device: torch.device,
) -> None:

    hidden_size = config.hidden_size
    shared_expert_intermediate_size = config.n_shared_experts * config.moe_intermediate_size
    #todo
    n_routed_experts = 64
    #Create the routing gate.
    for layer in model.model.layers:
        layer.mlp.gate = MoEGate(config, n_routed_experts)
        #SHARE_EXP
        layer.mlp.shared_experts = Deepseekv2BlockSparseTop2MLP(config, intermediate_size=shared_expert_intermediate_size)

def load_00_expert_state_dict(states_dir: str, device: torch.device,model_type:str):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        if model_type == "phimoe":
            module_idx = f"model.layers.0.block_sparse_moe.experts.0"
            state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.weight"]
        elif model_type == "qwen2_moe" or model_type == "deepseek_v2":
            #deepseek first layer dense 
            module_idx = f"model.layers.1.mlp.experts.0"
            state_fpath = json.load(f)["weight_map"][f"{module_idx}.gate_proj.weight"]
        else:
            print("not supported 1")
            exit(0)
    print(f"loading {os.path.join(states_dir, state_fpath)}")
    return load_file(os.path.join(states_dir, state_fpath), device=str(device))


def build_model(
    device: torch.device,
    offload_config: OffloadConfig,
    state_path: str,
    model_name: str,
    model_type: str,
):
    print("gpu info before build model")
    print_gpu_memory()
    state_dict_00 = load_00_expert_state_dict(state_path, device, model_type) #Load the initial expert parameters.
    print("gpu info after 00 expert")
    print_gpu_memory()
    def _make_module(model_type):
        if model_type=="qwen2_moe":
            config = Qwen2MoeConfig.from_pretrained(model_name)    
            expert = make_empty_expert_qwenmoe(config)
        elif model_type =="phimoe":
            config = PhimoeConfig.from_pretrained(model_name)    
            expert = make_empty_expert_phimoe(config)
        elif model_type=="deepseek_v2":
            config = DeepseekV2Config.from_pretrained(model_name)    
            expert = make_empty_expert_deepseekv2(config)
        else:
            print("not supported 2")
            exit(0)
        expert.load_state_dict(state_dict_00)
        expert.half()
        return ExpertWrapper(expert, device=device)
    
    if model_type=="qwen2_moe":
        model_config = Qwen2MoeConfig.from_pretrained(model_name)     
        with device, with_default_dtype(torch.float16):
            model = Qwen2MoeForCausalLM(
                Qwen2MoeConfig.from_pretrained(
                    model_name,
                    num_experts=0,
                    # The installed Qwen target selects eager attention.
                    # Its mask builder must use the same implementation.
                    attn_implementation="eager",
                    torch_dtype=torch.float16,
                    device_map=device,
                ),
            )
            #print(model.state_dict()["model.layers.0.mlp.shared_expert.up_proj.weight"][:10,:10])
        replace_attn_layers_qwenmoe(model, model_config, device)
        #print("————————————————————————————————————————————————————————————————————")
        #print(model.state_dict()["model.layers.0.mlp.shared_expert.up_proj.weight"][:10,:10])
    elif model_type=="phimoe":
        model_config = PhimoeConfig.from_pretrained(model_name)     
        with device, with_default_dtype(torch.float16):
            model = PhimoeForCausalLM(
                PhimoeConfig.from_pretrained(
                    model_name,
                    num_local_experts=0,
                    torch_dtype=torch.float16,
                    device_map=device,
                )
            )
            #print(model.state_dict()["model.layers.0.mlp.shared_expert.up_proj.weight"][:10,:10])
        replace_attn_layers_phimoe(model, model_config, device)
        # print("model.model._attn_implementation", model.model._attn_implementation)
        # exit(0)
    elif model_type=="deepseek_v2":
        model_config = DeepseekV2Config.from_pretrained(model_name, trust_remote_code=True)     
        with device, with_default_dtype(torch.float16):
            model = DeepseekV2ForCausalLM(
                DeepseekV2Config.from_pretrained(
                    model_name,
                    n_routed_experts=0,
                    torch_dtype=torch.float16,
                    device_map=device,
                ),
            )
            #print(model.state_dict()["model.layers.0.mlp.shared_expert.up_proj.weight"][:10,:10])
        replace_attn_layers_deepseekv2(model, model_config, device)        
    else:
        print("not supported 3")
        exit(0)


    state_index_path = os.path.join(state_path, "model.safetensors.index.json")
    with open(state_index_path) as f:
        weight_map = json.load(f)["weight_map"]

    trunk_state_path = os.path.join(
        state_path,
        weight_map["model.embed_tokens.weight"],
    )
    print(f"////load state dict//// {trunk_state_path}")
    # test fewer layers
    model.load_state_dict(load_file(trunk_state_path, device=str(device)), strict=False)
    #model.load_state_dict(load_file(trunk_state_path, device=str(device)), strict=True)
    print("gpu info after build non expert model")
    print_gpu_memory()
    print("////expert cache////")
    memory = psutil.virtual_memory()
    print(f"Available memory before expert catch: {memory.available / (1024**3):.2f} GB")
    expert_cache = ExpertCache(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
        model_type=model_type,
    )
    memory = psutil.virtual_memory()
    print(f"Available memory after expert catch: {memory.available / (1024**3):.2f} GB")
    print_gpu_memory()
    print("////do layer////") 
    memory = psutil.virtual_memory()
    print(f"Available memory before do layer: {memory.available / (1024**3):.2f} GB")
    
    if model_config.model_type=="qwen2_moe":
        for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"): #Process each layer.
            curr_layer = model.model.layers[layer_idx]
            # print(f"curr_layer.mlp.shared_expert {curr_layer.mlp.shared_expert.up_proj.weight}{curr_layer.mlp.shared_expert.up_proj.weight.device} {curr_layer.mlp.shared_expert_gate.weight.device}{curr_layer.mlp.shared_expert_gate.weight}")
            # exit(0)
            curr_layer.mlp = SparseMoeWrapperShared(
                model_config,
                layer_idx,
                curr_layer.mlp.gate.to(device, dtype=torch.float16),
                #SHARE_EXP
                curr_layer.mlp.shared_expert.to(device, dtype=torch.float16),
                curr_layer.mlp.shared_expert_gate.to(device, dtype=torch.float16),
                expert_cache,
            )

            for expert_idx in range(model_config.num_experts): # todo
                do_offload = True
                
                expert_wrapper = make_and_load_expert_wrapper_qwenmoe(
                    config=model_config,
                    states_dir=state_path,
                    expert_uid=(layer_idx, expert_idx),
                    device=device,
                )

                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    eviction_group=layer_idx,
                    offload=do_offload,
                )

                del expert_wrapper
                
                if expert_idx >= offload_config.offload_per_layer:
                    do_offload = False                
                    expert_wrapper = make_and_load_expert_wrapper_qwenmoe(
                        config=model_config,
                        states_dir=state_path,
                        expert_uid=(layer_idx, expert_idx),
                        device=device,
                    )

                    expert_cache.add_expert(
                        uid=(layer_idx, expert_idx),
                        module=expert_wrapper,
                        eviction_group=layer_idx,
                        offload=do_offload,
                    )

                    del expert_wrapper
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    elif model_config.model_type=="phimoe":
        for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"): #Process each layer.
            curr_layer = model.model.layers[layer_idx]
            curr_layer.block_sparse_moe = SparseMoeWrapperPhimoe(
                model_config,
                layer_idx,
                curr_layer.block_sparse_moe.gate,
                expert_cache,
            )

            for expert_idx in range(model_config.num_local_experts):
                do_offload = True
                expert_wrapper = make_and_load_expert_wrapper_phimoe(
                    config=model_config,
                    states_dir=state_path,
                    expert_uid=(layer_idx, expert_idx),
                    device=device,
                )                
                
                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    eviction_group=layer_idx,
                    offload=do_offload,
                )

                del expert_wrapper
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()                
                if expert_idx >= offload_config.offload_per_layer:
                    do_offload = False
                    expert_wrapper = make_and_load_expert_wrapper_phimoe(
                        config=model_config,
                        states_dir=state_path,
                        expert_uid=(layer_idx, expert_idx),
                        device=device,
                    )                
                    
                    expert_cache.add_expert(
                        uid=(layer_idx, expert_idx),
                        module=expert_wrapper,
                        eviction_group=layer_idx,
                        offload=do_offload,
                    )                    
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    elif model_config.model_type=="deepseek_v2":
        #skip denselayer begin at layer 1
        for layer_idx in trange(1, model_config.num_hidden_layers, desc="Loading experts"): #Process each layer.
            curr_layer = model.model.layers[layer_idx]
            # print(f"curr_layer.mlp.shared_expert {curr_layer.mlp.shared_expert.up_proj.weight}{curr_layer.mlp.shared_expert.up_proj.weight.device} {curr_layer.mlp.shared_expert_gate.weight.device}{curr_layer.mlp.shared_expert_gate.weight}")
            # exit(0)
            curr_layer.mlp = SparseMoeWrapperDeepseekv2(
                model_config,
                layer_idx,
                curr_layer.mlp.gate.to(device, dtype=torch.float16),
                #SHARE_EXP
                curr_layer.mlp.shared_experts.to(device, dtype=torch.float16),
                expert_cache,
            )

            for expert_idx in range(model_config.n_routed_experts): # todo
                #do_offload = expert_idx < offload_config.offload_per_layer
                if expert_idx >= offload_config.offload_per_layer:
                    do_offload = True
                    expert_wrapper = make_and_load_expert_wrapper_deepseekv2(
                        config=model_config,
                        states_dir=state_path,
                        expert_uid=(layer_idx, expert_idx),
                        device=device,
                    )

                    expert_cache.add_expert(
                        uid=(layer_idx, expert_idx),
                        module=expert_wrapper,
                        eviction_group=layer_idx,
                        offload=do_offload,
                    )

                    del expert_wrapper
                    do_offload = False
                    expert_wrapper = make_and_load_expert_wrapper_deepseekv2(
                        config=model_config,
                        states_dir=state_path,
                        expert_uid=(layer_idx, expert_idx),
                        device=device,
                    )

                    expert_cache.add_expert(
                        uid=(layer_idx, expert_idx),
                        module=expert_wrapper,
                        eviction_group=layer_idx,
                        offload=do_offload,
                    )

                    del expert_wrapper
                else:
                    do_offload = True
                    expert_wrapper = make_and_load_expert_wrapper_deepseekv2(
                        config=model_config,
                        states_dir=state_path,
                        expert_uid=(layer_idx, expert_idx),
                        device=device,
                    )

                    expert_cache.add_expert(
                        uid=(layer_idx, expert_idx),
                        module=expert_wrapper,
                        eviction_group=layer_idx,
                        offload=do_offload,
                    )

                    del expert_wrapper
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    else:
        print("not supported 4")
        exit(0)
    memory = psutil.virtual_memory()
    print(f"Available memory after do layer: {memory.available / (1024**3):.2f} GB")
    print("gpu info after build whole model")
    print_gpu_memory()
    return model
