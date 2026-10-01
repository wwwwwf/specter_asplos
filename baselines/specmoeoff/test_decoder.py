"""CPU regressions for exact-greedy B=1 acceptance, rollback and commitment."""
import unittest
from types import SimpleNamespace
import torch
from .decoder import greedy_decode


class ToyModel:
    def __init__(self, mode=0):
        self.mode = mode
        self.caches = []

    def __call__(self, ids, past_key_values, use_cache):
        self.caches.append(past_key_values)
        value = ids[:, None, :, None].float()
        history, _ = past_key_values.update(value, value.clone(), 0)
        sums = history[:, 0, :, 0].cumsum(-1)[:, -ids.shape[1]:].long()
        offset = self.mode if self.mode != 2 else (sums % 3 == 0).long()
        choice = (sums + offset + 1) % 7
        logits = torch.full((*ids.shape, 7), -8.)
        logits.scatter_(-1, choice.unsqueeze(-1), 8.)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class Clock:
    def __init__(self):
        self.lengths = []

    def commit(self, count):
        self.lengths.append(count)


class DecoderTests(unittest.TestCase):
    def test_accept_reject_partial_and_short_tail_match_target(self):
        for prefix in ([1], [1, 2, 3]):
            for tokens in (1, 2, 3, 4, 9, 19):
                for gamma in (1, 3, 8):
                    for mode in (0, 1, 2):
                        ids = torch.tensor([prefix])
                        reference = greedy_decode(None, ToyModel(), ids, tokens, target_only=True)['token_ids']
                        clock, draft, target = Clock(), ToyModel(mode), ToyModel()
                        actual = greedy_decode(draft, target, ids, tokens, gamma, clock=clock)
                        self.assertTrue(torch.equal(actual['token_ids'], reference), (prefix, tokens, gamma, mode))
                        self.assertEqual(clock.lengths[-1], len(prefix) + tokens)
                        self.assertTrue(all(a < b for a, b in zip([len(prefix)] + clock.lengths, clock.lengths)))
                        self.assertTrue(set(map(id, draft.caches)).isdisjoint(set(map(id, target.caches))))

    def test_greedy_generation_does_not_advance_rng(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        greedy_decode(ToyModel(2), ToyModel(), torch.tensor([[1, 2]]), 13, gamma=3)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_reject_invalid_workload(self):
        for ids, tokens, gamma in [(torch.ones((2, 1), dtype=torch.long), 4, 3),
                                   (torch.ones((1, 0), dtype=torch.long), 4, 3),
                                   (torch.ones((1, 1), dtype=torch.long), 0, 3),
                                   (torch.ones((1, 1), dtype=torch.long), 4, 0)]:
            with self.assertRaises(ValueError):
                greedy_decode(ToyModel(), ToyModel(), ids, tokens, gamma)


if __name__ == '__main__':
    unittest.main()
