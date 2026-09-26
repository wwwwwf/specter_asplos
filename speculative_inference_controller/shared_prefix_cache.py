"""One authoritative target prefix plus reusable, draft-only scratch storage.

Only speculative tokens occupy the reusable buffers. Attention still receives
a contiguous, temporary prefix + scratch concatenation; no second persistent
prefix is retained. Use this inference cache on the model's compute stream.
"""
import torch
from transformers import DynamicCache


class SharedPrefixDraftCache(DynamicCache):
    def __init__(self, committed, scratch_capacity=0):
        super().__init__()
        if not isinstance(scratch_capacity, int) or scratch_capacity < 0:
            raise ValueError('scratch_capacity must be a nonnegative integer')
        self.scratch_capacity = scratch_capacity
        self._scratch_key_buffers = []
        self._scratch_value_buffers = []
        self.rebase(committed)

    def rebase(self, committed):
        """Discard the old branch, alias current target KV, and reuse storage.

        Old buffer contents are invalid after this call. Each layer's visible
        scratch starts empty, so rejected or accepted draft KV is never reused
        as committed state. The state manager calls this between windows, after
        the previous draft's attention has been enqueued on the same stream.
        """
        if len(committed.key_cache) != len(committed.value_cache):
            raise ValueError('Committed key/value layer counts differ')
        self.key_cache = list(committed.key_cache)
        self.value_cache = list(committed.value_cache)
        layers = len(self.key_cache)
        for buffers in (self._scratch_key_buffers, self._scratch_value_buffers):
            if len(buffers) < layers:
                buffers.extend([None] * (layers - len(buffers)))
            elif len(buffers) > layers:
                del buffers[layers:]
        self.scratch_keys = [None] * layers
        self.scratch_values = [None] * layers
        self._scratch_lengths = [0] * layers
        self._seen_tokens = committed.get_seq_length()

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.key_cache):
            return 0
        prefix = self.key_cache[layer_idx].shape[-2]
        return prefix + self._scratch_lengths[layer_idx]

    @staticmethod
    def _validate_layout(prefix, states):
        if (states.ndim < 2 or prefix.ndim != states.ndim
                or prefix.shape[:-2] != states.shape[:-2]
                or prefix.shape[-1] != states.shape[-1]
                or prefix.device != states.device):
            raise ValueError('Draft KV must match its target prefix layout and device')

    def _append_scratch(self, buffer, states, used):
        needed = used + states.shape[-2]
        same_layout = (buffer is not None
            and buffer.shape[:-2] == states.shape[:-2]
            and buffer.shape[-1] == states.shape[-1]
            and buffer.device == states.device)
        dtype = states.dtype
        if same_layout and used and buffer.dtype != dtype:
            dtype = torch.promote_types(buffer.dtype, dtype)
        if not same_layout or buffer.dtype != dtype or buffer.shape[-2] < needed:
            # Explicit K reservation avoids growth in normal decoding; the
            # geometric fallback preserves the fork(cache) dynamic-cache API.
            previous_capacity = buffer.shape[-2] if same_layout else 0
            capacity = max(self.scratch_capacity, needed,
                previous_capacity * 2 if needed > previous_capacity else previous_capacity)
            shape = (*states.shape[:-2], capacity, states.shape[-1])
            replacement = torch.empty(shape, dtype=dtype, device=states.device)
            if used:
                replacement[..., :used, :].copy_(buffer[..., :used, :])
            buffer = replacement
        buffer[..., used:needed, :].copy_(states)
        return buffer, buffer[..., :needed, :]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx < 0 or layer_idx >= len(self.key_cache):
            raise RuntimeError('Draft branch must fork a fully initialized target prefix')
        self._validate_layout(self.key_cache[layer_idx], key_states)
        self._validate_layout(self.value_cache[layer_idx], value_states)
        if key_states.shape[-2] != value_states.shape[-2]:
            raise ValueError('Draft key/value token counts differ')
        used = self._scratch_lengths[layer_idx]
        self._scratch_key_buffers[layer_idx], self.scratch_keys[layer_idx] = self._append_scratch(
            self._scratch_key_buffers[layer_idx], key_states, used)
        self._scratch_value_buffers[layer_idx], self.scratch_values[layer_idx] = self._append_scratch(
            self._scratch_value_buffers[layer_idx], value_states, used)
        self._scratch_lengths[layer_idx] = used + key_states.shape[-2]
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        # Attention requires a contiguous view. Materialize only for this
        # layer's forward; the cache itself retains no duplicate prefix.
        return (torch.cat((self.key_cache[layer_idx], self.scratch_keys[layer_idx]), -2),
                torch.cat((self.value_cache[layer_idx], self.scratch_values[layer_idx]), -2))

    def crop(self, max_length):
        raise RuntimeError('Discard draft scratch and re-fork target committed KV; do not crop a draft branch')
