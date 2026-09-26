"""Attention-level regression: future tokens must not affect committed states."""
import torch
import pytest
from transformers import DynamicCache
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from initialization.attention import validate_qwen_attention


def model():
    torch.manual_seed(42)
    config = Qwen2MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=32, attn_implementation='eager')
    return Qwen2MoeForCausalLM(config).eval()


def forward_attention(m, x, cache, start):
    position = torch.arange(start, start+x.shape[1])
    mask = m.model._update_causal_mask(None, x, position, cache, False)
    rope = m.model.rotary_emb(x, position.unsqueeze(0))
    return m.model.layers[0].self_attn(x, attention_mask=mask,
        position_embeddings=rope, past_key_value=cache, use_cache=True)[0]


@torch.inference_mode()
def test_future_tokens_do_not_change_prefix():
    m = model()
    validate_qwen_attention(m)
    x = torch.randn(1, 6, 32)
    y = x.clone()
    y[:, 3:] = torch.randn_like(y[:, 3:]) * 5
    a = forward_attention(m, x, DynamicCache(), 0)
    b = forward_attention(m, y, DynamicCache(), 0)
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=0, rtol=0)


@torch.inference_mode()
def test_cached_chunking_preserves_attention():
    m = model()
    x = torch.randn(1, 6, 32)
    full = forward_attention(m, x, DynamicCache(), 0)
    cache = DynamicCache()
    a = forward_attention(m, x[:, :3], cache, 0)
    b = forward_attention(m, x[:, 3:], cache, 3)
    torch.testing.assert_close(full, torch.cat((a,b), 1), atol=1e-7, rtol=1e-5)


def test_mismatched_mask_configuration_is_rejected():
    m = model()
    m.config._attn_implementation = 'sdpa'
    with pytest.raises(RuntimeError, match='causal mask configured'):
        validate_qwen_attention(m)
