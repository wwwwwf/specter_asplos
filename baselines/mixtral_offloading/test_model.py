"""CPU checks for the author's historical DeepSeek Mixtral-Offloading path."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from . import model
from .legacy_dispatch import SparseMoeWrapperDeepseekv2 as HistoricalMoe
from .model import DeepseekLegacyMoe, FP16ExpertWrapper, LegacyGate, _construct_target
from .upstream.src.expert_cache import ExpertCache as UpstreamExpertCache


class ReverseCache:
    """Deliberately yield experts in non-ID order, as a resident-first cache can."""
    def __init__(self, experts):
        self.experts = experts
        self.calls = []

    def load_experts(self, *uids, unordered=False):
        if not unordered:
            raise AssertionError('Historical wrapper should request unordered iteration')
        for uid in reversed(uids):
            self.calls.append(uid)
            yield uid, self.experts[uid[1]]


class ZeroShared(nn.Module):
    def forward(self, hidden):
        return torch.zeros_like(hidden)


class HalfWeightGate(nn.Module):
    """A deliberately changed path that the dispatch equivalence test rejects."""
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def forward(self, hidden):
        selected, weights, auxiliary, logits = self.gate(hidden)
        return selected, weights.half(), auxiliary, logits


def routing_config():
    from models.target.configuration_deepseek import DeepseekV2Config
    return DeepseekV2Config(hidden_size=3, intermediate_size=5, moe_intermediate_size=5,
                           n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
                           topk_method='greedy', norm_topk_prob=False)


class HistoricalDispatchTests(unittest.TestCase):
    def setUp(self):
        # Only annotation hooks are disabled. Routing, indexing, expert kernels,
        # weighting and reduction execute real PyTorch operations on CPU.
        self.push = patch.object(torch.cuda.nvtx, 'range_push', return_value=0)
        self.pop = patch.object(torch.cuda.nvtx, 'range_pop', return_value=0)
        self.push.start()
        self.pop.start()
        self.addCleanup(self.push.stop)
        self.addCleanup(self.pop.stop)

    def test_historical_dispatch_body_matches_pinned_manifest(self):
        manifest = json.loads((Path(__file__).parent / 'legacy_dispatch.provenance.json').read_text(encoding='utf-8'))
        source = inspect.getsource(HistoricalMoe).replace('\r\n', '\n')
        self.assertEqual(manifest['source_commit'], '82d75e3ad3d38fe71c86b0dba290d1e0ef9e66f0')
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), manifest['class_utf8_lf_sha256'])
        self.assertTrue(issubclass(DeepseekLegacyMoe, HistoricalMoe))

    def test_real_gate_preserves_reference_indices_and_float32_weights(self):
        from models.target.modeling_deepseek import MoEGate
        torch.manual_seed(29)
        config = routing_config()
        reference, adapted = MoEGate(config).half().eval(), LegacyGate(config).half().eval()
        adapted.load_state_dict(reference.state_dict())
        hidden = torch.randn(1, 7, 3).half()
        with torch.inference_mode():
            expected = reference(hidden)
            actual = adapted(hidden)
        self.assertEqual(len(actual), 4)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(actual[0].dtype, torch.int64)
        self.assertEqual(actual[1].dtype, torch.float32)
        self.assertIsNone(actual[3])
        self.assertFalse(torch.equal(actual[1], actual[1].half().float()))

    def test_dispatch_matches_historical_class_without_half_rounding_weights(self):
        torch.manual_seed(29)
        config = routing_config()
        gate = LegacyGate(config).half().eval()
        experts = [nn.Linear(3, 3, bias=False).half() for _ in range(4)]
        first, second, rounded = (ReverseCache(experts) for _ in range(3))
        historical = HistoricalMoe(config, 7, gate, ZeroShared(), first)
        adapted = DeepseekLegacyMoe(config, 7, gate, ZeroShared(), second)
        altered = HistoricalMoe(config, 7, HalfWeightGate(gate), ZeroShared(), rounded)
        hidden = torch.randn(1, 7, 3).half()
        with torch.inference_mode():
            expected, _ = historical(hidden)
            actual = adapted(hidden)
            wrong, _ = altered(hidden)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(expected, wrong), 'Fixture must detect prematurely rounded routing weights')
        self.assertEqual(first.calls, second.calls)
        self.assertEqual(second.calls, sorted(second.calls, reverse=True))

    def test_reduction_follows_cache_yield_order_before_shared_expert(self):
        class Gate(nn.Module):
            def forward(self, hidden):
                selected = torch.tensor([[0, 1, 2]], dtype=torch.int64)
                return selected, torch.ones(1, 3, dtype=torch.float32), None, None
        class ConstantExpert(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, hidden):
                return torch.full_like(hidden, self.value)
        class Shared(nn.Module):
            def forward(self, hidden):
                return torch.full_like(hidden, 2)
        config = SimpleNamespace(hidden_size=1, intermediate_size=1, n_routed_experts=3, num_experts_per_tok=3)
        # Reverse order is +1, -10000, +10000 -> 0 in FP16; ID order
        # is +10000, -10000, +1 -> 1. Adding shared expert last gives 2.
        cache = ReverseCache([ConstantExpert(10000), ConstantExpert(-10000), ConstantExpert(1)])
        adapted = DeepseekLegacyMoe(config, 1, Gate(), Shared(), cache)
        with torch.inference_mode():
            actual = adapted(torch.ones(1, 1, 1, dtype=torch.float16))
        self.assertEqual(cache.calls, [(1, 2), (1, 1), (1, 0)])
        self.assertEqual(actual.item(), 2)

    def test_raw_fp16_storage_preserves_unfused_expert_and_upstream_cache_methods(self):
        from models.target.modeling_deepseek import DeepseekV2MLP
        config = SimpleNamespace(hidden_size=3, intermediate_size=5, hidden_act='silu')
        expert = DeepseekV2MLP(config).half()
        expected = deepcopy(expert)
        wrapped = FP16ExpertWrapper(expert, torch.device('cpu'))
        hidden = torch.randn(4, 3).half()
        with torch.inference_mode():
            self.assertTrue(torch.equal(wrapped(hidden), expected(hidden)))
        self.assertEqual(set(dict(wrapped.expert.named_parameters())),
                         {'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'})
        cache = UpstreamExpertCache(lambda: FP16ExpertWrapper(DeepseekV2MLP(config).half(), torch.device('cpu')),
                                    main_size=1, offload_size=0, buffer_size=0)
        cache.add_expert((0, 0), wrapped, eviction_group=0, offload=False)
        self.assertIs(type(cache), UpstreamExpertCache)
        self.assertIs(cache.load_experts.__func__, UpstreamExpertCache.load_experts)
        self.assertIs(cache._swap.__func__, UpstreamExpertCache._swap)
        for _, loaded in cache.load_experts((0, 0), unordered=True):
            with torch.inference_mode():
                self.assertTrue(torch.equal(loaded(hidden), expected(hidden)))

    def test_tiny_cpu_target_has_separate_gate_and_up_projections(self):
        from models.target.configuration_deepseek import DeepseekV2Config
        from models.target.modeling_deepseek import DeepseekV2MLP
        config = DeepseekV2Config(vocab_size=16, hidden_size=8, intermediate_size=12,
                                  moe_intermediate_size=4, num_hidden_layers=2,
                                  num_attention_heads=2, num_key_value_heads=2,
                                  n_shared_experts=1, n_routed_experts=4, num_experts_per_tok=2,
                                  first_k_dense_replace=1, topk_method='greedy', kv_lora_rank=4,
                                  q_lora_rank=None, qk_rope_head_dim=2, qk_nope_head_dim=2,
                                  v_head_dim=4, bos_token_id=0, eos_token_id=1)
        target, layers = _construct_target(config, torch.device('cpu'))
        self.assertEqual(layers, [1])
        self.assertIsInstance(target.model.layers[0].mlp, DeepseekV2MLP)
        moe = target.model.layers[1].mlp
        self.assertIsInstance(moe, DeepseekLegacyMoe)
        self.assertIsInstance(moe.gate, LegacyGate)
        self.assertIsInstance(moe.shared_experts, DeepseekV2MLP)
        self.assertFalse(hasattr(moe.shared_experts, 'gate_up_proj'))
        moe.experts = ReverseCache([DeepseekV2MLP(config, intermediate_size=4).half() for _ in range(4)])
        from transformers import DynamicCache
        target.eval()
        with torch.inference_mode():
            first = target(torch.tensor([[2, 3]]), past_key_values=DynamicCache(), use_cache=True)
            second = target(torch.tensor([[4]]), past_key_values=first.past_key_values, use_cache=True,
                            attention_mask=torch.zeros(1, 1, 1, 3, dtype=torch.float16))
        self.assertEqual(second.logits.shape, (1, 1, 16))
        self.assertEqual(second.past_key_values.get_seq_length(), 3)
        self.assertTrue(torch.isfinite(second.logits).all())

    def test_adapter_has_no_specter_runtime_imports(self):
        imports = []
        for node in ast.walk(ast.parse(inspect.getsource(model))):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
            elif isinstance(node, ast.Import):
                imports.extend(item.name for item in node.names)
        forbidden = ('streamlined_execution_engine', 'predictive_io_orchestrator',
                     'speculative_inference_controller', 'model_builder')
        self.assertFalse(any(any(part in name for part in forbidden) for name in imports), imports)


if __name__ == '__main__':
    unittest.main()
