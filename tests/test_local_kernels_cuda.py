"""Differential tests against the installed kernels used before vendoring."""
import unittest
from pathlib import Path
from configuration import get_config
import torch
from streamlined_execution_engine.kernels import ops
from streamlined_execution_engine.kernels.build import load_extension
from streamlined_execution_engine.kernels.scalar_type import scalar_types


class LocalKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_extension()
        expected = (get_config().cache_dir / 'kernels' / 'specter_cuda').resolve()
        if Path(cls.extension.__file__).resolve().parent != expected:
            raise AssertionError('Extension is not loaded from the configured build cache')

    def test_repack_matches_installed_gptq(self):
        import gptqmodel_marlin_kernels as original
        torch.manual_seed(71)
        for bits in [4, 8]:
            for permuted in [False, True]:
                k, n = 256, 256
                weight = torch.randint(-(2**31), 2**31-1, (k//(32//bits), n), device='cuda', dtype=torch.int32)
                perm = torch.randperm(k,device='cuda',dtype=torch.int32) if permuted else torch.empty(0,device='cuda',dtype=torch.int32)
                a = ops.gptq_marlin_repack(weight,perm,k,n,bits,32)
                b = original.gptq_marlin_repack(weight,perm,k,n,bits,32)
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_alignment_matches_installed_vllm(self):
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size as original
        torch.manual_seed(72)
        for m, topk, e, block in [(1,6,64,8),(17,6,64,8),(128,4,60,16),(513,2,16,32)]:
            routes = torch.randint(e,(m,topk),device='cuda',dtype=torch.int64)
            a = ops.moe_align_block_size(routes,block,e)
            b = original(routes,block,e)
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y,rtol=0,atol=0)

    def _check_alignment_semantics(self, routes, experts, block=8):
        ids, expert_ids, count = ops.moe_align_block_size(routes, block, experts)
        valid_count = int(count.item())
        valid_ids = ids[:valid_count]
        real = valid_ids[valid_ids < routes.numel()]
        torch.testing.assert_close(torch.sort(real).values,
            torch.arange(routes.numel(), device=routes.device, dtype=torch.int32))
        blocks = expert_ids[:valid_count // block].repeat_interleave(block)
        mask = valid_ids < routes.numel()
        torch.testing.assert_close(routes.flatten()[real.long()].to(torch.int32), blocks[mask])

    def test_shared_memory_cache_reuses_and_grows_without_shrinking(self):
        # E=16 fits the kernel default; E=128/192 require opt-in on A100.
        # Repeated smaller/larger requests must retain the largest capacity.
        routes = {e: torch.arange(34, device='cuda', dtype=torch.int64).view(17, 2) % e
                  for e in (16, 128, 192)}
        for e in (16, 128, 16, 192, 128):
            self._check_alignment_semantics(routes[e], e)
        warm = tuple(self.extension.shared_memory_cache_stats())
        self.assertGreater(warm[1], 0)
        for e in (192, 16, 128, 192):
            self._check_alignment_semantics(routes[e], e)
        self.assertEqual(warm, tuple(self.extension.shared_memory_cache_stats()))

    def test_shared_memory_cache_concurrent_host_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        device = torch.cuda.current_device()
        def run(experts):
            with torch.cuda.device(device):
                routes = torch.arange(34, device=f'cuda:{device}', dtype=torch.int64).view(17, 2) % experts
                for _ in range(3):
                    self._check_alignment_semantics(routes, experts)
                torch.cuda.synchronize(device)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, (16, 192, 128, 192)))
        warm = tuple(self.extension.shared_memory_cache_stats())
        # Fresh host threads may populate TLS, but need no new CUDA metadata calls.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, (192, 128, 16, 192)))
        self.assertEqual(warm, tuple(self.extension.shared_memory_cache_stats()))

    def test_moe_sum_matches_installed_vllm(self):
        from vllm import _custom_ops as original
        torch.manual_seed(73)
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            x=torch.randn(17,6,256,device='cuda',dtype=dtype)
            a=torch.empty(17,256,device='cuda',dtype=dtype)
            b=torch.empty_like(a)
            ops.moe_sum(x,a)
            original.moe_sum(x,b)
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_random_moe_gemm_matches_installed_vllm(self):
        from vllm import _custom_ops as original
        torch.manual_seed(74)
        for dtype in [torch.float16,torch.bfloat16]:
            for m in [1,17,128]:
                with self.subTest(dtype=dtype,m=m):
                    e,k,n,topk=8,256,256,2
                    q=torch.randint(-(2**31),2**31-1,(e,k//8,n),device='cuda',dtype=torch.int32)
                    perm=torch.empty(0,device='cuda',dtype=torch.int32)
                    packed=torch.stack([ops.gptq_marlin_repack(w,perm,k,n,4) for w in q])
                    scales=torch.rand(e,k//128,n,device='cuda',dtype=dtype)*0.02
                    ids=torch.randint(e,(m,topk),device='cuda',dtype=torch.int64)
                    sorted_ids,expert_ids,count=ops.moe_align_block_size(ids,8,e)
                    x=torch.randn(m,k,device='cuda',dtype=dtype)
                    weights=torch.softmax(torch.randn(m,topk,device='cuda'),-1)
                    def run(fn):
                        workspace=torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count*4,device='cuda',dtype=torch.int32)
                        output=torch.empty(m*topk,n,device='cuda',dtype=dtype)
                        return fn(x,output,packed,scales,None,None,None,None,workspace,sorted_ids,expert_ids,count,weights,
                            moe_block_size=8,top_k=topk,mul_topk_weights=False,is_ep=False,b_q_type=scalar_types.uint4b8,
                            size_m=m,size_n=n,size_k=k,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
                    a,b=run(ops.moe_wna16_marlin_gemm),run(original.moe_wna16_marlin_gemm)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
                    warm = tuple(self.extension.shared_memory_cache_stats())
                    repeated = run(ops.moe_wna16_marlin_gemm)
                    torch.cuda.synchronize()
                    self.assertEqual(warm, tuple(self.extension.shared_memory_cache_stats()))
                    torch.testing.assert_close(repeated,b,rtol=0,atol=0)

if __name__=='__main__':
    unittest.main()
