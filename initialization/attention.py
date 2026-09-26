"""Fail fast when Qwen attention and causal-mask configuration disagree."""


def validate_qwen_attention(model):
    configured = model.config._attn_implementation
    expected = {
        'eager': 'Qwen2MoeAttention',
        'sdpa': 'Qwen2MoeSdpaAttention',
        'flash_attention_2': 'Qwen2MoeFlashAttention2',
    }.get(configured)
    if expected is None:
        raise ValueError(f'Unsupported Qwen attention implementation: {configured}')
    for index, layer in enumerate(model.model.layers):
        actual = type(layer.self_attn).__name__
        if actual != expected:
            raise RuntimeError(
                f'Qwen layer {index}: causal mask configured for {configured}, '
                f'but actual attention is {actual}; expected {expected}'
            )
