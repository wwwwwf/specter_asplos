import unittest
from types import SimpleNamespace
import torch
from tests.test_async_expert_loading_cuda import CudaAsyncExpertCache
from predictive_io_orchestrator.prefetch_planner import LayerAwarePrefetchPlanner


class PlannerCudaTests(unittest.TestCase):
    def test_three_phases_and_over_capacity_authoritative_verification(self):
        manager = CudaAsyncExpertCache(capacity=2, total_experts=8, staging_buffers=0, elements=256)
        planner = LayerAwarePrefetchPlanner(1, 8, 2, 8, 1, manager, device='cuda:0', initial_fraction=0.25)
        route = SimpleNamespace(_selected_experts=torch.tensor([[4]], device='cuda'))
        planner.bind_draft_routes([route])
        planner.begin_window(8)
        planner.update_from_bound_routes()
        planner.after_draft_step(0)
        self.assertEqual(planner.phase, 'cold_start')
        self.assertEqual(len(planner.boundaries), 0)
        planner.update_from_bound_routes()
        planner.after_draft_step(1)
        self.assertEqual(planner.phase, 'prefetch_overlap')
        route._selected_experts = torch.tensor([[7]], device='cuda')
        planner.update_from_bound_routes()
        planner.after_draft_step(2)
        self.assertEqual(planner.phase, 'final_cleanup')
        for step in range(3, 8):
            planner.update_from_bound_routes()
            planner.after_draft_step(step)
        planner.before_verify()
        self.assertEqual(int(planner.activation_counters.sum()), 0)
        actual = []
        for uid, expert in manager.load_experts_async((0, 4), (0, 7), (0, 6), (0, 0), unordered=True):
            actual.append((uid, expert.storage.clone()))
        torch.cuda.synchronize()
        for uid, values in actual:
            self.assertTrue(torch.all(values == uid[1]), uid)
        planner.end_window()
        self.assertEqual(planner.phase, 'idle')
        self.assertEqual(len(planner.boundaries), 2)

if __name__ == '__main__':
    unittest.main()
