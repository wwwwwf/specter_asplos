import time
import torch
import psutil
from transformers import AutoConfig
from initialization.target_builder import OffloadConfig, build_model

def build_offload_model(model_name, state_path, device, offload_per_layer, buffer_size):
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)


    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.2f} GB")
    #print(f"Available memory: {memory.available / (1024**3):.2f} GB")

    if config.model_type == "phimoe":
        num_experts = config.num_local_experts
    elif config.model_type=="qwen2_moe":
        num_experts = config.num_experts
    elif config.model_type=="deepseek_v2":
        num_experts = config.n_routed_experts 
    else:
        print("not supported")
        exit(0)
    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=buffer_size,
        offload_per_layer=offload_per_layer,
    )


    model_begin = time.time()
    model = build_model(
        device=device,
        offload_config=offload_config,
        state_path=state_path,
        model_name=model_name,
        model_type=config.model_type,
    )
    model_end = time.time()
    if config.model_type == 'qwen2_moe':
        from initialization.attention import validate_qwen_attention
        validate_qwen_attention(model)
    print(f"build model tiime:{model_end - model_begin}s")
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model

