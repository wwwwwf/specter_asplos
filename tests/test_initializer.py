import unittest
from types import SimpleNamespace
import torch
from speculative_inference_controller.model_init import HybridPrecisionModelInitializer


class DummyFusedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_scales = torch.nn.Parameter(torch.tensor([0.125, 0.375], dtype=torch.bfloat16), requires_grad=False)
        self.w2_scales = torch.nn.Parameter(torch.tensor([0.25, 0.5], dtype=torch.bfloat16), requires_grad=False)
        self.w13_q_weight = torch.nn.Parameter(torch.tensor([17, -91, 1234567], dtype=torch.int32), requires_grad=False)
        self.register_buffer('floating_buffer', torch.tensor([0.625], dtype=torch.bfloat16))
        self.register_buffer('g_idx', torch.tensor([3, 1, 0], dtype=torch.int32))
        self.intermediate_cache13 = torch.tensor([0.75, -0.25], dtype=torch.bfloat16)
        self.intermediate_cache2 = torch.tensor([[0.125, 0.5]], dtype=torch.bfloat16)
        self.workspace = torch.tensor([0, 9], dtype=torch.int32)

    def forward(self, x):
        # Matching input/scales/scratch is required without any precision hook.
        assert x.dtype == self.w13_scales.dtype == self.intermediate_cache13.dtype
        return x * self.w13_scales + self.w2_scales


class DummyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(2, 2, bias=False).bfloat16()
        self.fusedexperts = DummyFusedExperts()


class InitializerTests(unittest.TestCase):
    def test_fp16_target_parameters_are_authoritative_and_shared(self):
        target = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).half()
        draft = SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).bfloat16())
        report = HybridPrecisionModelInitializer.share_non_expert_parameters(draft, target)
        self.assertIs(draft.model[0].weight, target[0].weight)
        self.assertEqual(draft.model[0].weight.dtype, torch.float16)
        self.assertEqual(report['shared_parameter_count'], 1)
        self.assertEqual(len(report['precision_conversions']), 1)

    def test_layout_mismatch_fails_explicitly(self):
        target = torch.nn.Sequential(torch.nn.Linear(3, 3, bias=False)).half()
        draft = SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).bfloat16())
        with self.assertRaisesRegex(ValueError, 'layout mismatch'):
            HybridPrecisionModelInitializer.share_non_expert_parameters(draft, target)

    def test_fused_fp16_preparation_preserves_quantized_state_and_shared_router(self):
        target = torch.nn.Module()
        target.gate = torch.nn.Linear(2, 2, bias=False).half()
        with torch.no_grad():
            target.gate.weight.copy_(torch.tensor([[0.625, -0.375], [0.25, 0.75]]))
        draft = SimpleNamespace(model=DummyDraft())
        fused = draft.model.fusedexperts
        integer_names = ['w13_q_weight', 'g_idx', 'workspace']
        integers = {name: (getattr(fused, name), getattr(fused, name).clone()) for name in integer_names}
        report = HybridPrecisionModelInitializer.share_non_expert_parameters(draft, target)
        self.assertIs(draft.model.gate.weight, target.gate.weight)
        self.assertEqual(report['fused_precision_adapter_count'], 0)
        self.assertEqual(len(report['fused_expert_precision']), 1)
        self.assertFalse(draft._specter_precision_hooks)
        self.assertFalse(fused._forward_pre_hooks)
        self.assertFalse(fused._forward_hooks)
        for name in ['w13_scales', 'w2_scales', 'floating_buffer', 'intermediate_cache13', 'intermediate_cache2']:
            self.assertEqual(getattr(fused, name).dtype, torch.float16, name)
        for name, (original, expected) in integers.items():
            self.assertIs(getattr(fused, name), original, name)
            torch.testing.assert_close(getattr(fused, name), expected, rtol=0, atol=0)
        x = torch.tensor([[0.375, -0.625]], dtype=torch.float16)
        with torch.inference_mode():
            routed = draft.model.gate(x)
            actual = fused(routed)
            expected = target.gate(x) * torch.tensor([0.125, 0.375], dtype=torch.float16) + torch.tensor([0.25, 0.5], dtype=torch.float16)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        first_scale = fused.w13_scales
        first_scratch = fused.intermediate_cache13
        HybridPrecisionModelInitializer.share_non_expert_parameters(draft, target)
        self.assertIs(fused.w13_scales, first_scale)
        self.assertIs(fused.intermediate_cache13, first_scratch)
        self.assertIs(draft.model.gate.weight, target.gate.weight)

    def test_reinitialization_removes_old_precision_adapters(self):
        target = torch.nn.Module()
        target.gate = torch.nn.Linear(2, 2, bias=False).half()
        draft = SimpleNamespace(model=DummyDraft())
        fused = draft.model.fusedexperts
        draft._specter_precision_hooks = [
            fused.register_forward_pre_hook(lambda module, args: (args[0].bfloat16(),)),
            fused.register_forward_hook(lambda module, args, output: output.half()),
        ]
        HybridPrecisionModelInitializer.share_non_expert_parameters(draft, target)
        self.assertFalse(fused._forward_pre_hooks)
        self.assertFalse(fused._forward_hooks)
        self.assertFalse(draft._specter_precision_hooks)

if __name__ == '__main__':
    unittest.main()
