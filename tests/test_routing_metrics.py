"""Route metric semantics and observer lifetime without model weights."""
import copy
import json
from types import SimpleNamespace
import unittest

from benchmarks.routing import (ForwardRouteCapture, RouteCapture, matched_route_fidelity,
                                _target_ids, route_specs, route_statistics, window_recall)

try:
    import torch
except ImportError:
    torch = None


class Tensor:
    cpu_calls = 0

    def __init__(self, rows):
        self.rows = rows

    def detach(self):
        return self

    def clone(self):
        return Tensor(copy.deepcopy(self.rows))

    def cpu(self):
        Tensor.cpu_calls += 1
        return self

    def tolist(self):
        return self.rows


class Module:
    def __init__(self):
        self.pre, self.post = [], []

    def register_forward_hook(self, callback):
        self.post.append(callback)
        return SimpleNamespace(remove=lambda: self.post.remove(callback))

    def register_forward_pre_hook(self, callback):
        self.pre.append(callback)
        return SimpleNamespace(remove=lambda: self.pre.remove(callback))

    def __call__(self, rows):
        for hook in self.pre:
            hook(self, (rows,))
        output = self.forward(rows)
        for hook in self.post:
            hook(self, (rows,), output)
        return output

    def forward(self, rows):
        return Tensor(rows), None, None, None


class Block(Module):
    def __init__(self, manager, target=False):
        super().__init__()
        self.manager, self.target = manager, target
        self.gate = Module()

    def forward(self, rows):
        self._selected_experts = Tensor(rows)
        if self.target:
            self.gate(rows)
            count = len({expert for row in rows for expert in row})
            self.manager.stats['demand_requests'] += count
            self.manager.stats['demand_hits'] += 1
            self.manager.stats['demand_misses'] += count - 1
        return None


class Controller:
    def __init__(self):
        self.expert_manager = SimpleNamespace(stats={'demand_requests': 0, 'demand_hits': 0, 'demand_misses': 0})
        self.expert_manager.get_async_loading_stats = lambda: dict(self.expert_manager.stats)

    def begin_window(self, draft_len):
        pass

    def end_window(self):
        pass

    def abort_window(self):
        pass


def fixture():
    case = SimpleNamespace(name='dsv2lite', layer_num=2, num_experts=8, skip_first_layer=True)
    controller = Controller()
    draft_blocks = [Block(controller.expert_manager) for _ in range(3)]
    target_blocks = [Block(controller.expert_manager, True) for _ in range(3)]
    draft = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(
        layers=[SimpleNamespace(mlp=block) for block in draft_blocks])))
    target = SimpleNamespace(model=SimpleNamespace(
        layers=[SimpleNamespace(mlp=block) for block in target_blocks]))
    return case, controller, draft, target, draft_blocks, target_blocks


class RoutingMetricsTests(unittest.TestCase):
    def test_window_recall_is_not_matched_token_fidelity(self):
        draft, target = [[0, 1], [2, 3]], [[2, 3], [0, 1]]
        self.assertEqual(window_recall(draft, target)['target_recall'], 1)
        self.assertEqual(matched_route_fidelity({2: draft}, {2: target})['mean_target_recall'], 0)
        self.assertIsNone(window_recall([], [])['target_recall'])
        with self.assertRaises(ValueError):
            matched_route_fidelity({1: draft}, {2: target})
        with self.assertRaises(ValueError):
            matched_route_fidelity({1: draft}, {1: target[:1]})

    def test_diversity_counts_and_empty_results(self):
        result = route_statistics([[1, 2], [2, 3]], 8)
        self.assertEqual(result['expert_counts'], {'1': 1, '2': 2, '3': 1})
        self.assertEqual(result['working_set_fraction'], 3 / 8)
        self.assertEqual(result['assignments'], 4)
        self.assertEqual(result['adjacent_token_jaccard_mean'], 1 / 3)
        self.assertEqual(route_statistics([], 8)['effective_experts'], 0)
        self.assertIsNone(matched_route_fidelity({}, {})['mean_target_recall'])
        for rows in ([[1, 1]], [[1, 2], [3]], [[-1]], [[9]], [1]):
            with self.assertRaises(ValueError):
                route_statistics(rows, 8)

    def test_window_capture_excludes_prefill_tail_and_preserves_shapes(self):
        case, controller, draft, target, db, tb = fixture()
        Tensor.cpu_calls = 0
        with RouteCapture(case, draft, target, controller) as capture:
            for block in tb[1:]:
                block([[7], [7]])
            controller.begin_window(2)
            db[1]([[1]])
            db[2]([[3]])
            db[1]([[1]])
            db[2]([[3]])
            tb[1]([[1], [2]])
            tb[2]([[3], [3]])
            controller.end_window()
            for block in tb[1:]:
                block([[7]])
            self.assertEqual(Tensor.cpu_calls, 0)
            with self.assertRaises(RuntimeError):
                capture.result()
        result = capture.result()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result['summary']['target_recall']['mean'], 0.75)
        self.assertEqual(result['summary']['demand']['demand_requests'], 3)
        self.assertEqual(result['summary']['demand']['hit_rate'], 2 / 3)
        layers = result['windows'][0]['layers']
        self.assertEqual([row['layer_id'] for row in layers], [1, 2])
        self.assertEqual(layers[0]['draft_ids'], [[1], [1]])
        self.assertEqual(layers[0]['target_ids'], [[1], [2]])

    def test_empty_and_aborted_capture_restore_every_patch(self):
        case, controller, draft, target, db, tb = fixture()
        original = dict(vars(controller))
        with RouteCapture(case, draft, target, controller) as empty:
            pass
        self.assertIsNone(empty.result()['summary']['target_recall']['mean'])
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with RouteCapture(case, draft, target, controller) as capture:
                controller.begin_window(2)
                db[1]([[1]])
                raise RuntimeError('injected')
        self.assertEqual(vars(controller), original)
        self.assertEqual(capture.result()['summary']['aborted_windows'], 1)
        self.assertEqual(capture.result()['summary']['layer_window_count'], 0)
        self.assertTrue(all(not x.pre and not x.post and not x.gate.post for x in db + tb))

    def test_forward_capture_clones_ids_and_checks_layer_order(self):
        case, controller, draft, target, db, tb = fixture()
        rows = [[1, 2]]
        with ForwardRouteCapture(case, draft) as capture:
            db[1](rows)
            db[2](rows)
            rows[0][0] = 7
        self.assertEqual(capture.result()['calls'][0]['layers'][0]['ids'], [[1, 2]])
        with ForwardRouteCapture(case, target, role='target') as capture:
            tb[2]([[1]])
        with self.assertRaisesRegex(ValueError, 'reordered'):
            capture.result()
        case.layer_num = 3
        with self.assertRaisesRegex(ValueError, 'routed layers'):
            route_specs(case, target)

    def test_failed_enter_removes_partial_hooks(self):
        case, controller, draft, target, db, tb = fixture()
        del tb[2].gate
        original = dict(vars(controller))
        with self.assertRaises(AttributeError):
            with RouteCapture(case, draft, target, controller):
                pass
        self.assertEqual(vars(controller), original)
        self.assertTrue(all(not x.post for x in db))
        self.assertFalse(tb[1].gate.post)

    def test_completed_window_cannot_omit_target_layer(self):
        case, controller, draft, target, db, tb = fixture()
        with RouteCapture(case, draft, target, controller) as capture:
            controller.begin_window(1)
            db[1]([[1]])
            db[2]([[2]])
            tb[1]([[1]])
            controller.end_window()
        with self.assertRaisesRegex(ValueError, 'missing'):
            capture.result()


@unittest.skipIf(torch is None, 'PyTorch is unavailable')
class TorchRouteTests(unittest.TestCase):
    def test_gate_route_rules_include_ties_and_low_precision(self):
        logits = torch.tensor([[1, 1, 1, 0], [100, 100, 99, -100]], dtype=torch.float16)
        case = SimpleNamespace(name='qwen2moe')
        block = SimpleNamespace(top_k=2)
        expected = torch.topk(torch.softmax(logits, dim=1, dtype=torch.float), 2, dim=-1).indices
        self.assertTrue(torch.equal(_target_ids(case, block, logits), expected))
        case.name = 'phimoe'
        self.assertTrue(torch.equal(_target_ids(case, block, logits), torch.tensor([[0, 1], [0, 1]])))
        case.name = 'dsv2lite'
        self.assertIs(_target_ids(case, block, (expected, None, None, logits)), expected)

    def test_real_forward_hooks_preserve_output_and_restore_after_exception(self):
        class Routed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Identity()
                self.top_k = 2

            def forward(self, value):
                return self.gate(value)

        block = Routed()
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=block)]))
        case = SimpleNamespace(name='qwen2moe', layer_num=1, num_experts=4)
        logits = torch.tensor([[0., 3., 2., 1.], [0., 1., 2., 3.]])
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with ForwardRouteCapture(case, model, 'target') as capture:
                self.assertTrue(torch.equal(block(logits), logits))
                raise RuntimeError('injected')
        self.assertFalse(block.gate._forward_hooks)
        self.assertEqual(capture.result()['calls'][0]['layers'][0]['ids'], [[1, 2], [3, 2]])


if __name__ == '__main__':
    unittest.main()
