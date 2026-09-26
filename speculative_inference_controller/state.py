"""SIC: target-authoritative committed KV and independent speculative extensions."""
import time
import torch
from transformers import DynamicCache
from speculative_inference_controller.shared_prefix_cache import SharedPrefixDraftCache
from speculative_inference_controller.model_wrapper import ModelWrapper

class TargetConsistentDecodingStateManager:
    def __init__(self, draft, target, draft_capacity=0):
        self.draft, self.target = draft, target
        self.draft_capacity = draft_capacity
        self._draft_cache = None

    @staticmethod
    def fork(cache, scratch_capacity=0):
        return SharedPrefixDraftCache(cache, scratch_capacity=scratch_capacity)

    def _share_committed(self):
        cache = self.target._past_key_values
        if self._draft_cache is None:
            self._draft_cache = self.fork(cache, self.draft_capacity)
        else:
            self._draft_cache.rebase(cache)
        self.draft._past_key_values = self._draft_cache
        self.draft._prob_history = self.target._prob_history

    def prefill(self, input_ids):
        # Leave one uncached token for both forward paths. Prefix KV comes
        # exclusively from the full precision target, including first round.
        if input_ids.shape[1] > 1:
            self.target._forward_with_kvcache(input_ids[:, :-1])
            self._share_committed()

    def commit(self, committed_length):
        self.target.rollback(committed_length)
        self._share_committed()

UNTRUNCATED_SAMPLING_CONFIG = {
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
}

# These values mirror the sampling fields in each checkpoint's
# generation_config.json. They are used only when profile="model".
MODEL_SAMPLING_CONFIGS = {
    "dsv2lite": {
        "temperature": 0.3,
        "top_k": 0,
        "top_p": 0.95,
    },
    "qwen2moe": {
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    },
    "phimoe": {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
    },
}

MODEL_TYPE_TO_SAMPLING_KEY = {
    "deepseek_v2": "dsv2lite",
    "qwen2_moe": "qwen2moe",
    "qwen2_moe_fused": "qwen2moe",
    "phimoe": "phimoe",
}

SAMPLING_STRATEGY_ALIASES = {
    "greedy": "greedy",
    "sampling": "sampling",
    "stochastic": "sampling",
    "random": "sampling",
}

SAMPLING_PROFILE_ALIASES = {
    "untruncated": "untruncated",
    "full": "untruncated",
    "model": "model",
    "custom": "custom",
}

SAMPLING_CONFIG_FIELDS = frozenset(UNTRUNCATED_SAMPLING_CONFIG)


def _resolve_sampling_model_key(model_name, target_model_proto):
    model_type = getattr(target_model_proto.config, "model_type", None)
    requested_key = model_name or model_type
    return MODEL_TYPE_TO_SAMPLING_KEY.get(requested_key, requested_key)


def _normalize_sampling_strategy(sampling_strategy):
    strategy = SAMPLING_STRATEGY_ALIASES.get(
        str(sampling_strategy).strip().lower()
    )
    if strategy is None:
        supported = ", ".join(sorted(SAMPLING_STRATEGY_ALIASES))
        raise ValueError(
            f"Unknown sampling strategy {sampling_strategy!r}; "
            f"supported values: {supported}"
        )
    return strategy


def _normalize_sampling_profile(sampling_profile):
    profile = SAMPLING_PROFILE_ALIASES.get(
        str(sampling_profile).strip().lower()
    )
    if profile is None:
        supported = ", ".join(sorted(SAMPLING_PROFILE_ALIASES))
        raise ValueError(
            f"Unknown sampling profile {sampling_profile!r}; "
            f"supported values: {supported}"
        )
    return profile


def _validate_sampling_config(sampling_config):
    config = dict(sampling_config)
    unknown = set(config) - SAMPLING_CONFIG_FIELDS
    missing = SAMPLING_CONFIG_FIELDS - set(config)
    if unknown:
        raise ValueError(
            f"Unknown sampling config fields: {sorted(unknown)}"
        )
    if missing:
        raise ValueError(
            f"Missing sampling config fields: {sorted(missing)}"
        )

    temperature = float(config["temperature"])
    top_k = int(config["top_k"])
    top_p = float(config["top_p"])
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    return {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }


def _resolve_sampling_config(
    model_name,
    target_model_proto,
    sampling_profile,
    sampling_config,
):
    config_key = _resolve_sampling_model_key(
        model_name, target_model_proto
    )
    profile = _normalize_sampling_profile(sampling_profile)

    if profile == "untruncated":
        if sampling_config is not None:
            raise ValueError(
                "sampling_config is only valid with profile='custom'"
            )
        resolved = UNTRUNCATED_SAMPLING_CONFIG
    elif profile == "model":
        if sampling_config is not None:
            raise ValueError(
                "sampling_config is only valid with profile='custom'"
            )
        if config_key not in MODEL_SAMPLING_CONFIGS:
            supported = ", ".join(sorted(MODEL_SAMPLING_CONFIGS))
            raise KeyError(
                f"No model sampling config for {config_key!r}; "
                f"supported keys: {supported}"
            )
        resolved = MODEL_SAMPLING_CONFIGS[config_key]
    else:
        if sampling_config is None:
            raise ValueError(
                "profile='custom' requires sampling_config"
            )
        overrides = dict(sampling_config)
        unknown = set(overrides) - SAMPLING_CONFIG_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown sampling config fields: {sorted(unknown)}"
            )
        resolved = {
            **UNTRUNCATED_SAMPLING_CONFIG,
            **overrides,
        }

    return config_key, profile, _validate_sampling_config(resolved)


def _filter_sampling_probs(probs, top_k, top_p):
    probs = probs.float()
    if top_k > 0:
        top_k = min(int(top_k), probs.shape[-1])
        threshold = torch.topk(probs, top_k, dim=-1).values[..., -1:]
        probs = torch.where(probs >= threshold, probs, 0.0)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(
            probs, dim=-1, descending=True
        )
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(
            -1, sorted_indices, sorted_probs
        )

    normalizer = probs.sum(dim=-1, keepdim=True)
    if not torch.isfinite(normalizer).all() or (normalizer <= 0).any():
        raise RuntimeError("Invalid sampling distribution")
    return probs / normalizer


def _select_token(probs, strategy):
    if strategy == "greedy":
        return torch.argmax(probs, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def _generate_draft_window(
    approx_model,
    input_ids,
    draft_len,
    strategy,
    top_k,
    top_p,
    prefetch_controller,
):
    draft_seq = input_ids
    proposal_probs = []
    if prefetch_controller is not None:
        prefetch_controller.begin_window(draft_len)
    try:
        for draft_index in range(draft_len):
            probs = approx_model._forward_with_kvcache(draft_seq)
            probs = _filter_sampling_probs(probs, top_k, top_p)
            proposal_probs.append(probs)
            next_token = _select_token(probs, strategy)
            draft_seq = torch.cat((draft_seq, next_token), dim=1)

            if prefetch_controller is not None:
                prefetch_controller.after_draft_step(draft_index)
        if prefetch_controller is not None:
            prefetch_controller.before_verify()
    except Exception:
        if prefetch_controller is not None:
            prefetch_controller.abort_window()
        raise
    return draft_seq, proposal_probs


def spec_inf(
    quant_model_proto,
    target_model_proto,
    input_ids,
    num_tokens,
    gamma,
    tokenizer,
    prefetch_controller=None,
    sampling_strategy="sampling",
    model_name=None,
    sampling_profile="untruncated",
    sampling_config=None,
    state_factory=None,
):
    """Run exact greedy or canonical stochastic speculative decoding."""
    assert input_ids.shape[0] == 1
    strategy = _normalize_sampling_strategy(sampling_strategy)
    if strategy == "greedy":
        if sampling_config is not None:
            raise ValueError(
                "sampling_config is not used by greedy decoding"
            )
        config_key = _resolve_sampling_model_key(
            model_name, target_model_proto
        )
        resolved_profile = "greedy"
        resolved_sampling_config = dict(
            UNTRUNCATED_SAMPLING_CONFIG
        )
    else:
        (
            config_key,
            resolved_profile,
            resolved_sampling_config,
        ) = _resolve_sampling_config(
            model_name,
            target_model_proto,
            sampling_profile,
            sampling_config,
        )
    temperature_smp = resolved_sampling_config["temperature"]
    top_k_smp = resolved_sampling_config["top_k"]
    top_p_smp = resolved_sampling_config["top_p"]
    if gamma < 1 or num_tokens < 0:
        raise ValueError("gamma must be positive and num_tokens nonnegative")
    max_gamma = gamma

    print(
        f"[decode.config] model={config_key} strategy={strategy} "
        f"profile={resolved_profile} "
        f"temperature={temperature_smp} top_k={top_k_smp} "
        f"top_p={top_p_smp}"
    )

    accept_statistic = torch.zeros((max_gamma + 1), dtype=torch.int32)
    approx_model = ModelWrapper(
        quant_model_proto,
        prefetch_controller,
        temperature_smp,
        top_k_smp,
        top_p_smp,
    )
    target_model = ModelWrapper(
        target_model_proto,
        None,
        temperature_smp,
        top_k_smp,
        top_p_smp,
    )

    token_len = input_ids.shape[1]
    total_len = token_len + num_tokens
    accept_total = 0
    checked_total = 0
    proposed_total = 0

    start_time_sd = time.time()
    assert (
        approx_model._model.config.vocab_size
        == target_model._model.config.vocab_size
    )

    step = 0
    t_prefill_start = time.time()
    # Each draft forward caches one token: the uncached correction/prompt tail,
    # then at most K-1 proposals. K slots cover even the full-length window.
    state_type = state_factory or TargetConsistentDecodingStateManager
    state = state_type(approx_model, target_model, max_gamma)
    state.prefill(input_ids)
    while token_len < total_len:
        remaining = total_len - token_len
        draft_len = min(max_gamma, max(0, remaining - 1))

        if draft_len == 0:
            target_probs = target_model._forward_with_kvcache(input_ids)
            target_probs = _filter_sampling_probs(
                target_probs, top_k_smp, top_p_smp
            )
            correction = _select_token(target_probs, strategy)
            input_ids = torch.cat((input_ids, correction), dim=1)
            token_len += 1
            break

        drft_seq, proposal_probs = _generate_draft_window(
            approx_model,
            input_ids,
            draft_len,
            strategy,
            top_k_smp,
            top_p_smp,
            prefetch_controller,
        )
        try:
            target_model._forward_with_kvcache(drft_seq)
        except Exception:
            abort_window = getattr(
                prefetch_controller,
                "abort_window",
                None,
            )
            if abort_window is not None:
                abort_window()
            raise
        else:
            end_window = getattr(
                prefetch_controller,
                "end_window",
                None,
            )
            if end_window is not None:
                end_window()
        if step == 0:
            print(f"prefill:{time.time() - t_prefill_start}")

        accept_len = 0
        for index in range(draft_len):
            draft_token = drft_seq[0, token_len + index]
            target_probs = target_model._prob_history[
                :, token_len + index - 1, :
            ]
            target_probs = _filter_sampling_probs(
                target_probs, top_k_smp, top_p_smp
            )

            if strategy == "greedy":
                target_token = torch.argmax(target_probs, dim=-1)
                if draft_token != target_token[0]:
                    break
            else:
                p_token = target_probs[0, draft_token]
                q_token = proposal_probs[index][0, draft_token]
                ratio = torch.clamp(
                    p_token
                    / q_token.clamp_min(torch.finfo(torch.float32).tiny),
                    max=1.0,
                )
                if torch.rand((), device=input_ids.device) > ratio:
                    break
            accept_len += 1

        correction_position = token_len + accept_len - 1
        correction_probs = target_model._prob_history[
            :, correction_position, :
        ]
        correction_probs = _filter_sampling_probs(
            correction_probs, top_k_smp, top_p_smp
        )
        if strategy == "sampling" and accept_len < draft_len:
            residual = torch.clamp(
                correction_probs - proposal_probs[accept_len], min=0.0
            )
            if residual.sum(dim=-1).min() <= torch.finfo(torch.float32).eps:
                residual = correction_probs
            correction = _select_token(
                residual / residual.sum(dim=-1, keepdim=True), strategy
            )
        else:
            correction = _select_token(correction_probs, strategy)

        accepted_prefix = drft_seq[
            :, token_len : token_len + accept_len
        ]
        input_ids = torch.cat(
            (input_ids, accepted_prefix, correction), dim=1
        )
        token_len = input_ids.shape[1]
        accept_total += accept_len
        checked_total += accept_len + int(accept_len < draft_len)
        proposed_total += draft_len
        accept_statistic[accept_len] += 1

        # The correction token is selected but has not been forwarded yet.
        state.commit(token_len - 1)
        step += 1

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    end_time_sd = time.time()
    print(
        f"speculative decoding: {generated_text}\n",
        f"time:{end_time_sd - start_time_sd}",
    )

    acceptance_rate = (
        accept_total / checked_total if checked_total else 0.0
    )
    window_utilization = (
        accept_total / proposed_total if proposed_total else 0.0
    )
    print(
        f"token_len:{token_len}, accept_total:{accept_total}, "
        f"total_len:{total_len}"
    )
    print(
        f"acceptance_rate:{acceptance_rate:.6f}, "
        f"window_utilization:{window_utilization:.6f}"
    )
    for i, count in enumerate(accept_statistic):
        print(f"accept len = {i}: {count}")
    print(f"step:{step}")
    target_model._past_key_values = None
    target_model._prob_history = None
    approx_model._past_key_values = None
    approx_model._prob_history = None
    torch.cuda.empty_cache()
    return input_ids
    
