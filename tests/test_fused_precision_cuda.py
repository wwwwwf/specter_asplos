"""Nonzero INT4 fused experts against an explicit FP16 two-projection reference.

These tests use small synthetic quantized weights, never load full checkpoints,
and exercise the real local Marlin repacker/kernel and all three model forwards.
"""
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from speculative_inference_controller.model_init import HybridPrecisionModelInitializer


@unittest.skipUnless(torch.cuda.is_available(), 'requires the local CUDA kernels')
class FusedPrecisionCudaTests(unittest.TestCase):
    def test_deepseek_native_fp16_int4(self):
        self.check_kernel('dsv2lite.modeling_deepseek', 'DeepseekV2Fusedexperts', 6, torch.float32)

    def test_qwen_native_fp16_int4(self):
        self.check_kernel('qwen2moe.modeling_qwen2_moe_fused', 'Qwen2MoeFusedExperts', 4, torch.float16)

    def test_phi_native_fp16_int4(self):
        self.check_kernel('phimoe.modeling_fusedphimoe', 'PhiMoEFusedExperts', 2, torch.float16)

    @staticmethod
    def packed_projection(generator, experts, k, n):
        from initialization.marlin_pack import repack, marlin_moe_permute_scales
        # Asymmetric, nonzero integer values and nonuniform scales detect wrong
        # layouts, ignored weights and silent all-zero outputs.
        q = torch.randint(1, 16, (experts, k, n), generator=generator, dtype=torch.int64)
        scales = (torch.rand(experts, k // 128, n, generator=generator) * 0.03125 + 0.015625).bfloat16()
        shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 1, 8, 1)
        packed = (q.reshape(experts, k // 8, 8, n) << shifts).sum(dim=2).to(torch.int32).cuda()
        perm = torch.empty(experts, 0, dtype=torch.int32, device='cuda')
        repacked = repack(packed, perm, experts, k, n, 4)
        permuted_scales = marlin_moe_permute_scales(scales.cuda(), k, n, 128)
        # BF16 scale values loaded by the existing weight loader are retained;
        # the reference represents their one-time FP16 conversion exactly.
        dequantized = ((q - 8).half() * scales.half().repeat_interleave(128, dim=1)).cuda()
        return repacked, permuted_scales, dequantized

    def check_kernel(self, module_path, class_name, topk, routing_dtype):
        # Qwen's upstream file uses an absolute config import.
        qwen_dir = str(Path(__file__).resolve().parents[1] / 'models/draft/qwen2moe')
        if qwen_dir not in sys.path:
            sys.path.insert(0, qwen_dir)
        module = importlib.import_module('models.draft.' + module_path)
        config = SimpleNamespace(hidden_size=256, moe_intermediate_size=256,
                                 intermediate_size=256, n_routed_experts=8,
                                 num_experts=8, num_local_experts=8, hidden_act='silu')
        expert = getattr(module, class_name)(config).cuda()
        generator = torch.Generator().manual_seed(20260924)
        e, h, f = 8, 256, 256
        w13, s13, reference13 = self.packed_projection(generator, e, h, f * 2)
        w2, s2, reference2 = self.packed_projection(generator, e, f, h)
        expert.w13_q_weight = torch.nn.Parameter(w13, requires_grad=False)
        expert.w2_q_weight = torch.nn.Parameter(w2, requires_grad=False)
        expert.w13_scales = torch.nn.Parameter(s13, requires_grad=False)
        expert.w2_scales = torch.nn.Parameter(s2, requires_grad=False)
        for name in ['w13_qzeros', 'w2_qzeros']:
            expert.register_parameter(name, None)
        for name in ['w13_g_idx', 'w2_g_idx', 'w13_g_idx_sort_indices', 'w2_g_idx_sort_indices']:
            expert.register_parameter(name, torch.nn.Parameter(torch.empty(e, 0, dtype=torch.int32, device='cuda'), requires_grad=False))
        draft_model = torch.nn.Module()
        draft_model.fusedexperts = expert
        draft_model.gate = torch.nn.Linear(h, e, bias=False, device='cuda', dtype=torch.bfloat16)
        target = torch.nn.Module()
        target.gate = torch.nn.Linear(h, e, bias=False, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            target.gate.weight.copy_(torch.randn(e, h, generator=generator).cuda() * 0.125)
        draft = SimpleNamespace(model=draft_model)
        original_qweights = [expert.w13_q_weight, expert.w2_q_weight]
        qweight_values = [weight.clone() for weight in original_qweights]
        HybridPrecisionModelInitializer.share_non_expert_parameters(draft, target)
        self.assertIs(draft_model.gate.weight, target.gate.weight)
        self.assertFalse(expert._forward_pre_hooks)
        self.assertFalse(expert._forward_hooks)
        for weight, original, expected in zip([expert.w13_q_weight, expert.w2_q_weight], original_qweights, qweight_values):
            self.assertIs(weight, original)
            torch.testing.assert_close(weight, expected, rtol=0, atol=0)

        with torch.inference_mode():
            for tokens in [1, 5]:
                with self.subTest(tokens=tokens):
                    x = (torch.randn(tokens, h, generator=generator) * 0.25).half().cuda()
                    # DS intentionally retains its existing FP32 gate arithmetic.
                    if routing_dtype == torch.float32:
                        logits = F.linear(x.float(), draft_model.gate.weight.float())
                    else:
                        logits = draft_model.gate(x)
                    weights, selected = logits.softmax(-1, dtype=torch.float32).topk(topk, dim=-1)
                    weights = (weights / weights.sum(-1, keepdim=True)).to(routing_dtype)
                    if tokens == 5:
                        # Force the growth/device-dtype checks to allocate fresh
                        # FP16 scratch, rather than only testing initial buffers.
                        expert.intermediate_cache13 = torch.empty(1, dtype=torch.bfloat16, device='cuda')
                        expert.intermediate_cache2 = torch.empty(1, 1, dtype=torch.bfloat16, device='cuda')
                    with patch.object(module.ops, 'moe_wna16_marlin_gemm', wraps=module.ops.moe_wna16_marlin_gemm) as calls:
                        actual = expert(x, logits, weights, selected)
                    self.assertEqual(calls.call_count, 2)
                    for call in calls.call_args_list:
                        self.assertEqual(call.args[0].dtype, torch.float16)
                        self.assertEqual(call.args[1].dtype, torch.float16)
                        self.assertEqual(call.args[3].dtype, torch.float16)
                    reference_rows = []
                    for row in range(tokens):
                        contributions = []
                        for slot in range(topk):
                            expert_id = int(selected[row, slot])
                            projected = x[row:row + 1] @ reference13[expert_id]
                            activated = F.silu(projected[:, :f]) * projected[:, f:]
                            contributions.append((activated @ reference2[expert_id]).squeeze(0))
                        contribution = torch.stack(contributions) * weights[row].unsqueeze(-1)
                        reference_rows.append(torch.sum(contribution, dim=0,
                                                        out=torch.empty(h, dtype=torch.float16, device='cuda')))
                    expected = torch.stack(reference_rows)
                    self.assertEqual(actual.dtype, torch.float16)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertGreater(float(expected.abs().max()), 0.01)
                    self.assertGreater(float(actual.abs().max()), 0.01)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0005)
                    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
                    self.assertLess(float(relative_l2), 0.015)
                    self.assertEqual(expert.intermediate_cache13.dtype, torch.float16)
                    self.assertEqual(expert.intermediate_cache2.dtype, torch.float16)
                    self.assertGreaterEqual(expert.intermediate_cache13.numel(), tokens * topk * max(2 * f, h))
                    self.assertGreaterEqual(expert.intermediate_cache2.shape[0], tokens * topk)
                    scratch = expert.intermediate_cache13
                    repeated = expert(x, logits, weights, selected)
                    self.assertIs(expert.intermediate_cache13, scratch)
                    torch.testing.assert_close(repeated, actual, rtol=0.02, atol=0.0005)
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
