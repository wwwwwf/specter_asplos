"""CPU-only regression tests; never loads checkpoints or initializes CUDA."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from transformers import DynamicCache

from .backend import Backend
from .model import _load_component as load_component


class LoaderTests(unittest.TestCase):
    def test_full_and_split_expert_keys_and_missing_weights(self):
        from safetensors.torch import save_file
        for local_keys in [False, True]:
            with TemporaryDirectory() as directory:
                module = torch.nn.Linear(3, 2, bias=False)
                full_name = "model.layers.1.mlp.experts.0.weight"
                expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
                save_file({"weight" if local_keys else full_name: expected}, str(Path(directory) / "expert.safetensors"))
                load_component(module, directory, {full_name: "expert.safetensors"}, "model.layers.1.mlp.experts.0.")
                self.assertTrue(torch.equal(module.weight, expected))
                with self.assertRaises(KeyError):
                    load_component(module, directory, {}, "model.layers.1.mlp.experts.0.")


class GenerationTests(unittest.TestCase):
    def test_greedy_exact_budget_fresh_kv_and_commit_boundaries(self):
        class Target:
            calls = []
            def __call__(self, input_ids, past_key_values, **kwargs):
                self.calls.append((input_ids.clone(), past_key_values))
                logits = torch.zeros((1, input_ids.shape[1], 8))
                logits[:, -1, (int(input_ids[0, -1]) + 1) % 8] = 1
                return SimpleNamespace(logits=logits, past_key_values=past_key_values)
        class Clock:
            def __init__(self): self.commits = []
            def commit(self, total): self.commits.append(total)
        target, clock = Target(), Clock()
        backend = Backend(None, target, SimpleNamespace(group_infos={}))
        result = backend.generate(torch.tensor([[1, 2]]), 5, clock=clock)
        self.assertEqual(result["token_ids"].tolist(), [[1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(clock.commits, [3, 4, 5, 6, 7])
        self.assertIsInstance(target.calls[0][1], DynamicCache)
        self.assertTrue(all(ids.shape[1] == 1 for ids, _ in target.calls[1:]))
        backend.generate(torch.tensor([[1, 2]]), 1)
        self.assertIsInstance(target.calls[-1][1], DynamicCache)
        self.assertIsNot(target.calls[-1][1], target.calls[0][1])


if __name__ == "__main__":
    unittest.main()
