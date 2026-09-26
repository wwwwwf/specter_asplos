"""Scoped mechanism ablations that preserve target-authoritative decoding."""
from __future__ import annotations

from contextlib import AbstractContextManager


class ScopedPatches(AbstractContextManager):
    """Restore instance attributes, including inherited methods, on every exit."""

    def __init__(self):
        self._originals = []

    def set(self, instance, name, value):
        namespace = vars(instance)
        self._originals.append((instance, name, name in namespace, namespace.get(name)))
        setattr(instance, name, value)

    def __exit__(self, exc_type, exc, traceback):
        for instance, name, existed, original in reversed(self._originals):
            if existed:
                setattr(instance, name, original)
            else:
                delattr(instance, name)
        self._originals.clear()
        return False


def copied_state_factory(draft, target, draft_capacity=0):
    """Copy validated target KV into a fresh independent draft cache per window.

    This isolates shared-prefix storage and reusable draft scratch. Weight
    sharing, depth selection, acceptance, and target rollback remain enabled;
    it is not a removal of the complete speculative inference controller.
    """
    from transformers import DynamicCache
    from speculative_inference_controller.state import TargetConsistentDecodingStateManager

    class CopiedCommittedState(TargetConsistentDecodingStateManager):
        def _share_committed(self):
            committed = self.target._past_key_values
            if len(committed.key_cache) != len(committed.value_cache):
                raise ValueError("Committed key/value layer counts differ")
            branch = DynamicCache()
            for layer, (keys, values) in enumerate(zip(
                committed.key_cache, committed.value_cache
            )):
                branch.update(keys.clone(), values.clone(), layer)
            self._draft_cache = branch
            self.draft._past_key_values = branch
            self.draft._prob_history = self.target._prob_history

    return CopiedCommittedState(draft, target, draft_capacity)


class ScopedVariant(AbstractContextManager):
    """Apply a partial mechanism ablation to one manager for one decode pass.

    ``serial`` preserves request order and submits one expert at a time. Each
    transfer depends on all previously enqueued compute, preventing overlap
    of demand copies with earlier expert computation. Predictive prefetch is
    independently controlled, and fused draft kernels are still used.
    """

    def __init__(self, manager=None, *, kv_mode="shared", expert_mode="overlap"):
        if kv_mode not in {"shared", "copied"}:
            raise ValueError("kv_mode must be 'shared' or 'copied'")
        if expert_mode not in {"overlap", "serial"}:
            raise ValueError("expert_mode must be 'overlap' or 'serial'")
        self.manager = manager
        self.kv_mode = kv_mode
        self.expert_mode = expert_mode
        self._patches = None

    @property
    def state_factory(self):
        return copied_state_factory if self.kv_mode == "copied" else None

    @property
    def metadata(self):
        return {
            "kv_mode": self.kv_mode,
            "expert_mode": self.expert_mode,
            "scope": "partial mechanism ablation",
            "copied_kv_scope": "independent target-validated prefix and dynamic draft cache",
            "serial_scope": "one demand expert at a time; compute-to-transfer dependency; shared expert overlap disabled",
            "preserved": ["target verification", "canonical acceptance", "shared model weights", "fused draft kernels"],
            "full_sic_removal": False,
            "full_see_removal": False,
        }

    def __enter__(self):
        if self._patches is not None:
            raise RuntimeError("Variant context is already active")
        manager = self.manager
        if self.expert_mode == "serial":
            if manager is None or not getattr(manager, "specter_async_loading_enabled", False):
                raise ValueError("Serial demand requires an enabled asynchronous manager")
            if getattr(manager, "_async_demand_active", False):
                raise RuntimeError("Cannot change execution during an active demand")
            if getattr(manager, "_scoped_serial_variant", False):
                raise RuntimeError("Serial demand variant is already installed")
        patches = ScopedPatches()
        self._patches = patches
        try:
            if self.expert_mode == "serial":
                original = manager.load_experts_async

                def load_serial(*uids, unordered=True, prefetch=False):
                    del unordered
                    if len(set(uids)) != len(uids):
                        raise ValueError("duplicate demand expert UIDs")
                    groups = {manager.registered_experts[uid].eviction_group for uid in uids}
                    if len(groups) > 1:
                        raise ValueError("demand experts must share an eviction group")

                    def iterate():
                        for uid in uids:
                            compute_done = manager._new_async_event()
                            manager._async_record_event(compute_done, manager._async_current_stream())
                            manager._async_wait_event(manager.stream_tx, compute_done)
                            iterator = original(uid, unordered=False, prefetch=prefetch)
                            try:
                                yield from iterator
                            finally:
                                close = getattr(iterator, "close", None)
                                if close is not None:
                                    close()

                    return iterate()

                patches.set(manager, "_scoped_serial_variant", True)
                patches.set(manager, "load_experts_async", load_serial)
                patches.set(manager, "specter_async_overlap_shared_experts", False)
        except BaseException:
            patches.__exit__(None, None, None)
            self._patches = None
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        patches, self._patches = self._patches, None
        if patches is not None:
            return patches.__exit__(exc_type, exc, traceback)
        return False
