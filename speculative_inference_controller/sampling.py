import torch
from torch.nn import functional as F
import numpy as np
import random

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def sample_token(logits, temperature, top_k, top_p):
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float('Inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_logits[cumulative_probs > top_p] = -float('Inf')
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
#     """

#     Args:
#         logits (torch.Tensor): shape (1, vocab)
#         temperature (float): temperature
#         top_k (float): top_k
#         top_p (float): top_p

#     Returns:
#         torch.Tensor: next token with shape as (batch,  1)
#     """
#     # # assert logits.dim() == 2

#     # logits = logits / temperature
#     # logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
#     # probs = F.softmax(logits, dim=1)
#     # return probs
#     return logits


def norm_logits(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): shape (..., vocab)
        temperature (float): temperature; <=0 means no temperature scaling
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: normalized probabilities with same shape as logits
    """
    if temperature is not None and temperature > 0:
        logits = logits / temperature

    # Uncomment to enable top-k/top-p sampling.
    # logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

    return F.softmax(logits, dim=-1)


def prune_cache(cache, token_len):
    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue

        layer = []
        for i in range(len(layer_cache)):
            tensor = layer_cache[i]
            new_tensor = tensor[:, :, :token_len, :]
            layer.append(new_tensor)
        new_cache.append(tuple(layer))

    return tuple(new_cache)

def check_quant_linear_types(module, prefix=""):
    for name, submodule in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        # Check for a quantized linear layer.
        if isinstance(submodule, torch.nn.Linear) and hasattr(submodule, 'quantization'):
            print(f"Layer {full_name} is of type: {type(submodule).__name__}")
            # Identify known quantized linear implementations by subclass name.
            if 'Exllama' in str(type(submodule)):
                print(f"  -> This layer uses ExllamaQuantLinear")
            elif 'GPTQ' in str(type(submodule)):
                print(f"  -> This layer uses GPTQQuantLinear")
            else:
                print(f"  -> This layer uses a different QuantLinear implementation")
        check_quant_linear_types(submodule, full_name)
