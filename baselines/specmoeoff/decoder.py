"""Greedy speculative decoding with separate draft and target KV caches."""
import torch
from transformers import DynamicCache


class CachedModel:
    def __init__(self, model, controller=None):
        self.model = model
        self.controller = controller
        self.cache = DynamicCache()
        self.probabilities = None

    def forward(self, ids):
        length = self.cache.get_seq_length()
        if ids.shape[1] <= length:
            raise RuntimeError("A model forward requires an uncached token suffix")
        result = self.model(ids[:, length:], past_key_values=self.cache, use_cache=True)
        if not isinstance(result.past_key_values, DynamicCache):
            raise TypeError("Adapted model must return DynamicCache")
        self.cache = result.past_key_values
        if self.cache.get_seq_length() != ids.shape[1]:
            raise RuntimeError("Model did not append the full uncached suffix")
        if self.controller is not None:
            self.controller.after_draft_forward()
        # Same native-precision normalization as the supported target runtime;
        # greedy decisions are consistent with its target-only reference.
        probs = torch.softmax(result.logits, dim=-1)
        self.probabilities = probs if self.probabilities is None else torch.cat((self.probabilities, probs), dim=1)
        return self.probabilities[:, -1, :]

    def rollback(self, length):
        # An all-accepted draft has not forwarded its final proposal yet.
        # crop() therefore must never pad its shorter cache.
        retained = min(length, self.cache.get_seq_length())
        self.cache.crop(retained)
        self.probabilities = self.probabilities[:, :retained]


def greedy_decode(draft, target, input_ids, num_tokens, gamma=3, controller=None, clock=None,
                  target_only=False):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("SpecMoEOff adaptation supports nonempty B=1 inputs only")
    if num_tokens < 1 or gamma < 1:
        raise ValueError("Positive output length and gamma required")
    result = input_ids.clone()
    stop = result.shape[1] + num_tokens
    verifier = CachedModel(target)
    proposer = None if target_only else CachedModel(draft, controller)
    stats = {'accepted_tokens': 0, 'proposed_tokens': 0, 'sd_rounds': 0}

    def commit():
        if clock is not None:
            clock.commit(result.shape[1])

    while result.shape[1] < stop:
        prefix = result.shape[1]
        count = 0 if target_only else min(gamma, stop - prefix - 1)
        if count == 0:
            token = verifier.forward(result).argmax(-1, keepdim=True)
            result = torch.cat((result, token), dim=1)
            commit()
            continue
        proposal = result
        for index in range(count):
            token = proposer.forward(proposal).argmax(-1, keepdim=True)
            proposal = torch.cat((proposal, token), dim=1)
            if controller is not None:
                controller.after_draft_token(index)
        if controller is not None:
            controller.before_verify()
        verifier.forward(proposal)
        accepted = 0
        for index in range(count):
            choice = verifier.probabilities[:, prefix + index - 1].argmax(-1)
            if bool((proposal[:, prefix + index] != choice).any()):
                break
            accepted += 1
        correction = verifier.probabilities[:, prefix + accepted - 1].argmax(-1, keepdim=True)
        result = torch.cat((proposal[:, :prefix + accepted], correction), dim=1)
        proposer.rollback(result.shape[1] - 1)
        verifier.rollback(result.shape[1] - 1)
        if controller is not None:
            controller.finish_round()
        stats['accepted_tokens'] += accepted
        stats['proposed_tokens'] += count
        stats['sd_rounds'] += 1
        commit()
    stats['window_utilization'] = stats['accepted_tokens'] / stats['proposed_tokens'] if stats['proposed_tokens'] else 0.0
    return {'token_ids': result, 'stats': stats}
