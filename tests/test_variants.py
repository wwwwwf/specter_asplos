"""State-preservation and exception-safety tests for mechanism ablations."""
import contextlib
import io
import unittest
from types import SimpleNamespace

from benchmarks.variants import ScopedPatches, ScopedVariant, copied_state_factory

try:
    import torch
    from speculative_inference_controller.model_wrapper import ModelWrapper
    from speculative_inference_controller.state import spec_inf
    from tests.test_state import ToyModel, Tokenizer
except ModuleNotFoundError:
    torch = None


class FakeManager:
    specter_async_loading_enabled = True
    _async_demand_active = False
    specter_async_overlap_shared_experts = True

    def __init__(self):
        self.registered_experts = {uid: SimpleNamespace(eviction_group=0) for uid in (0, 1, 2)}
        self.calls = []
        self.closed = []
        self.stream_tx = "transfer"

    def _new_async_event(self):
        return object()

    def _async_current_stream(self):
        return "compute"

    def _async_record_event(self, event, stream):
        self.calls.append(("record", stream))

    def _async_wait_event(self, stream, event):
        self.calls.append(("wait", stream))

    def load_experts_async(self, *uids, unordered=True, prefetch=False):
        self.calls.append(("load", uids, unordered))

        def iterator():
            try:
                for uid in uids:
                    yield uid, "expert"
            finally:
                self.closed.append(uids)

        return iterator()


class VariantScopeTests(unittest.TestCase):
    def test_default_leaves_manager_and_state_factory_unchanged(self):
        manager = FakeManager()
        original = dict(vars(manager))
        with ScopedVariant(manager) as variant:
            self.assertIsNone(variant.state_factory)
            self.assertEqual(vars(manager), original)
        self.assertEqual(vars(manager), original)

    def test_serial_request_order_and_stream_dependencies(self):
        manager = FakeManager()
        with ScopedVariant(manager, expert_mode="serial"):
            self.assertFalse(manager.specter_async_overlap_shared_experts)
            self.assertEqual(list(manager.load_experts_async(2, 0, 1)), [
                (2, "expert"), (0, "expert"), (1, "expert")
            ])
        self.assertEqual(manager.calls, [item for uid in (2, 0, 1) for item in (
            ("record", "compute"), ("wait", "transfer"), ("load", (uid,), False)
        )])
        self.assertEqual(manager.closed, [(2,), (0,), (1,)])
        self.assertNotIn("load_experts_async", vars(manager))
        self.assertNotIn("specter_async_overlap_shared_experts", vars(manager))
        self.assertTrue(manager.specter_async_overlap_shared_experts)

    def test_exception_restores_preexisting_instance_methods_and_flags(self):
        manager = FakeManager()
        original = manager.load_experts_async
        manager.load_experts_async = original
        manager.specter_async_overlap_shared_experts = "original"
        with self.assertRaisesRegex(ValueError, "decode failed"):
            with ScopedVariant(manager, expert_mode="serial"):
                raise ValueError("decode failed")
        self.assertIs(manager.load_experts_async, original)
        self.assertEqual(manager.specter_async_overlap_shared_experts, "original")
        self.assertNotIn("_scoped_serial_variant", vars(manager))

    def test_closing_serial_generator_releases_original_iterator(self):
        manager = FakeManager()
        with ScopedVariant(manager, expert_mode="serial"):
            iterator = manager.load_experts_async(1, 2)
            self.assertEqual(next(iterator), (1, "expert"))
            iterator.close()
        self.assertEqual(manager.closed, [(1,)])
        self.assertEqual(sum(call[0] == "load" for call in manager.calls), 1)

    def test_serial_validation_and_nested_context(self):
        manager = FakeManager()
        with ScopedVariant(manager, expert_mode="serial"):
            with self.assertRaises(ValueError):
                manager.load_experts_async(0, 0)
            with self.assertRaises(RuntimeError):
                with ScopedVariant(manager, expert_mode="serial"):
                    pass
        self.assertNotIn("_scoped_serial_variant", vars(manager))

    def test_patch_stack_restores_inherited_and_nested_values(self):
        manager = FakeManager()
        with self.assertRaises(RuntimeError):
            with ScopedPatches() as patches:
                patches.set(manager, "specter_async_overlap_shared_experts", False)
                patches.set(manager, "specter_async_overlap_shared_experts", None)
                raise RuntimeError("failure")
        self.assertNotIn("specter_async_overlap_shared_experts", vars(manager))


@unittest.skipIf(torch is None, "Torch and Transformers are required")
class CopiedKVTests(unittest.TestCase):
    def test_committed_copy_has_equal_values_and_independent_storage(self):
        draft, target = ModelWrapper(ToyModel(1)), ModelWrapper(ToyModel())
        state = copied_state_factory(draft, target, draft_capacity=8)
        with torch.inference_mode():
            state.prefill(torch.tensor([[1, 2, 3]]))
            draft._forward_with_kvcache(torch.tensor([[1, 2, 3, 1, 1]]))
            old_branch = draft._past_key_values
            target._forward_with_kvcache(torch.tensor([[1, 2, 3, 2, 2]]))
            state.commit(4)
            branch = draft._past_key_values
            self.assertIsNot(branch, old_branch)
            self.assertEqual(branch.get_seq_length(), 4)
            for field in ("key_cache", "value_cache"):
                copied = getattr(branch, field)[0]
                authoritative = getattr(target._past_key_values, field)[0]
                self.assertTrue(torch.equal(copied, authoritative))
                self.assertNotEqual(copied.data_ptr(), authoritative.data_ptr())
            draft._forward_with_kvcache(torch.tensor([[1, 2, 3, 2, 0]]))
            self.assertEqual(target._past_key_values.get_seq_length(), 4)
            self.assertEqual(target._past_key_values.key_cache[0].flatten().tolist(), [1, 2, 3, 2])

    def test_copied_mode_preserves_greedy_tokens_for_acceptance_and_rejection(self):
        for prompt in ([1], [1, 2, 3]):
            for offset in (0, 1):
                for depth in (1, 4, 16):
                    ids = torch.tensor([prompt])
                    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                        shared = spec_inf(ToyModel(offset), ToyModel(), ids.clone(), 9,
                                          depth, Tokenizer(), sampling_strategy="greedy")
                        with ScopedVariant(kv_mode="copied") as variant:
                            copied = spec_inf(ToyModel(offset), ToyModel(), ids.clone(), 9,
                                              depth, Tokenizer(), sampling_strategy="greedy",
                                              state_factory=variant.state_factory)
                    self.assertTrue(torch.equal(shared, copied), (prompt, offset, depth))

    def test_copied_mode_preserves_seeded_stochastic_decisions(self):
        from tests.test_state import ConstantModel
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            outputs = []
            for factory in (None, copied_state_factory):
                torch.manual_seed(19)
                outputs.append(spec_inf(
                    ConstantModel([0.4, 0.3, 0.2, 0.1]),
                    ConstantModel([0.1, 0.2, 0.3, 0.4]),
                    torch.tensor([[1, 2, 3]]), 17, 4, Tokenizer(),
                    sampling_strategy="sampling", state_factory=factory,
                ))
        self.assertTrue(torch.equal(*outputs))


if __name__ == "__main__":
    unittest.main()
