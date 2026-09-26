import contextlib
import io
import unittest
from types import SimpleNamespace
import torch
from transformers import DynamicCache
from speculative_inference_controller.state import TargetConsistentDecodingStateManager, spec_inf
from speculative_inference_controller.depth_profiler import LightweightSpeculativeDepthProfiler
from speculative_inference_controller.model_wrapper import ModelWrapper


class ToyModel:
    config = SimpleNamespace(vocab_size=4, model_type='deepseek_v2')

    def __init__(self, offset=0):
        self.offset = offset

    def __call__(self, ids, past_key_values, use_cache):
        history, _ = past_key_values.update(ids[:, None, :, None].float(), ids[:, None, :, None].float(), 0)
        sums = history[:, 0, :, 0].cumsum(-1)[:, -ids.shape[1]:].long()
        logits = torch.full((*ids.shape, 4), -4., device=ids.device)
        logits.scatter_(-1, ((sums + self.offset + 1) % 4).unsqueeze(-1), 4.)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class Tokenizer:
    def decode(self, ids, **kwargs):
        return str(ids.tolist())


class ConstantModel(ToyModel):
    def __init__(self, probabilities):
        self.logits = torch.tensor(probabilities).log()

    def __call__(self, ids, past_key_values, use_cache):
        states = ids[:, None, :, None].float()
        past_key_values.update(states, states, 0)
        return SimpleNamespace(logits=self.logits.expand(*ids.shape, 4).clone(), past_key_values=past_key_values)


class StateTests(unittest.TestCase):
    def test_fork_shares_prefix_without_cross_branch_mutation(self):
        cache = DynamicCache()
        prefix = torch.arange(3.).reshape(1, 1, 3, 1)
        cache.update(prefix, prefix.clone(), 0)
        fork = TargetConsistentDecodingStateManager.fork(cache)
        self.assertEqual(cache.key_cache[0].data_ptr(), fork.key_cache[0].data_ptr())
        extension = torch.ones(1, 1, 2, 1)
        fork.update(extension, extension, 0)
        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(fork.get_seq_length(), 5)
        self.assertTrue(torch.equal(cache.key_cache[0], prefix))

    def test_commit_replaces_draft_kv_with_target_prefix(self):
        draft, target = ModelWrapper(ToyModel(1)), ModelWrapper(ToyModel())
        state = TargetConsistentDecodingStateManager(draft, target, draft_capacity=8)
        ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            state.prefill(ids)
            draft._forward_with_kvcache(torch.tensor([[1, 2, 3, 1, 1]]))
            branch = draft._past_key_values
            scratch_pointer = branch._scratch_key_buffers[0].data_ptr()
            target._forward_with_kvcache(torch.tensor([[1, 2, 3, 2, 2]]))
            state.commit(4)
        self.assertEqual(draft._past_key_values.get_seq_length(), 4)
        self.assertEqual(target._past_key_values.get_seq_length(), 4)
        self.assertIs(draft._past_key_values.key_cache[0], target._past_key_values.key_cache[0])
        self.assertEqual(draft._past_key_values.key_cache[0].flatten().tolist(), [1, 2, 3, 2])
        self.assertIs(draft._past_key_values, branch)
        self.assertIsNone(branch.scratch_keys[0])
        with torch.inference_mode():
            draft._forward_with_kvcache(torch.tensor([[1, 2, 3, 2, 0]]))
        self.assertEqual(branch._scratch_key_buffers[0].data_ptr(), scratch_pointer)
        self.assertEqual(branch.scratch_keys[0].flatten().tolist(), [0.])
        self.assertEqual(target._past_key_values.get_seq_length(), 4)

    def test_greedy_accept_reject_short_tail_and_single_token_prompt(self):
        for prompt in [[1], [1, 2, 3]]:
            for offset in [0, 1]:
                for gamma in [1, 4, 16]:
                    ids = torch.tensor([prompt])
                    expected = ids.clone()
                    target = ModelWrapper(ToyModel())
                    with torch.inference_mode():
                        for _ in range(9):
                            p = target._forward_with_kvcache(expected)
                            expected = torch.cat((expected, p.argmax(-1, keepdim=True)), 1)
                        with contextlib.redirect_stdout(io.StringIO()):
                            actual = spec_inf(ToyModel(offset), ToyModel(), ids, 9, gamma, Tokenizer(), sampling_strategy='greedy')
                    self.assertTrue(torch.equal(actual, expected), (prompt, offset, gamma))

    def test_depth_selection_uses_mean_tpot(self):
        result = LightweightSpeculativeDepthProfiler().profile(lambda k, p, r: abs(k-8)+1+p+r, [0, 1], 3)
        self.assertEqual(result['selected_depth'], 8)
        self.assertEqual(len(result['records']), 54)

    def test_stochastic_acceptance_and_residual_preserve_target_marginal(self):
        target_probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
        counts = torch.zeros(4)
        torch.manual_seed(20260920)
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2000):
                result = spec_inf(ConstantModel([0.4, 0.3, 0.2, 0.1]), ConstantModel(target_probabilities),
                    torch.tensor([[1]]), 2, 1, Tokenizer(), sampling_strategy='sampling')
                counts[result[0, 1]] += 1
        self.assertTrue(torch.all((counts / 2000 - target_probabilities).abs() < 0.045), counts.tolist())

if __name__ == '__main__':
    unittest.main()
