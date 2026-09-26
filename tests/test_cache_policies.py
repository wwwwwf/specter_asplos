"""Prefetch timing, event ordering, and scoped restoration without a GPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.cache_policies import PrefetchPolicy


class Event:
    def record(self, stream):
        stream.events.append(('record', self))


class Stream:
    def __init__(self):
        self.events = []

    def wait_event(self, event):
        self.events.append(('wait', event))


class Manager:
    def __init__(self):
        self.async_loading_stats = {'h2d_bytes': 0}
        self.specter_async_demand_policy = 'resident'
        self.calls = []
        self.ready = Event()

    def prefetch_experts_async(self, *uids, unordered):
        self.calls.append((uids, unordered))
        self.async_loading_stats['h2d_bytes'] += len(uids) * 100
        return [(uid, None, self.ready) for uid in uids]


class Controller:
    def __init__(self):
        self.expert_manager = Manager()
        self.stream = Stream()
        self.verify_stream = Stream()
        self._lookahead_ready = Event()
        self.max_cached_experts = 2
        self.boundaries = []
        self.phase = 'idle'
        self.mutex = None
        self.steps, self.plans, self.original_steps = 0, [], []

    def clear(self):
        self.steps = 0

    def begin_window(self, draft_len):
        self.clear()
        self.draft_len = draft_len
        self.phase = 'cold_start'
        self._predicted = [set()]

    def update_from_bound_routes(self):
        if self.phase != 'final_cleanup':
            self.steps += 1

    def get_all_layer_topm_uids(self, count):
        self.plans.append(self.steps)
        return [[(1, 2), (1, 3)][:count]]

    def after_draft_step(self, index):
        self.original_steps.append(index)

    def before_verify(self):
        if self.mutex is not None:
            self.verify_stream.wait_event(self.mutex)
        self.clear()

    def end_window(self):
        self.phase = 'idle'

    def abort_window(self):
        self.before_verify()
        self.phase = 'idle'

    def get_prefetch_policy_stats(self):
        return {'policy': 'original', 'boundaries': self.boundaries}


class CachePolicyTests(unittest.TestCase):
    def setUp(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, stream=lambda stream: nullcontext()))
        self.patch = patch.dict('sys.modules', {'torch': torch})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_early_and_late_submit_once_at_different_boundaries(self):
        for name, expected in [('early', 1), ('late', 3)]:
            controller = Controller()
            original_methods = {key: key in vars(controller) for key in (
                'begin_window', 'after_draft_step', 'before_verify', 'end_window', 'abort_window')}
            with PrefetchPolicy(controller, name) as policy:
                controller.begin_window(3)
                for step in range(3):
                    controller.update_from_bound_routes()
                    controller.after_draft_step(step)
                if name == 'late':
                    self.assertFalse(controller.expert_manager.calls)
                controller.before_verify()
                self.assertEqual(controller.plans, [expected])
                self.assertEqual(len(controller.expert_manager.calls), 1)
                self.assertEqual(controller.stream.events[0], ('wait', controller._lookahead_ready))
                self.assertIn(('wait', controller.expert_manager.ready), controller.stream.events)
                self.assertEqual(controller.verify_stream.events[-1], ('wait', controller.mutex))
                self.assertEqual(controller.get_prefetch_policy_stats()['policy'], name)
                self.assertEqual(policy.metadata()['demand_policy'], 'resident')
                controller.end_window()
            self.assertEqual(controller.get_prefetch_policy_stats()['policy'], 'original')
            self.assertEqual({key: key in vars(controller) for key in original_methods}, original_methods)

    def test_none_preserves_demand_cache_and_submits_nothing(self):
        controller = Controller()
        with PrefetchPolicy(controller, 'none'):
            controller.begin_window(3)
            for step in range(3):
                controller.update_from_bound_routes()
                controller.after_draft_step(step)
            controller.before_verify()
            self.assertEqual(controller.steps, 0)
            controller.end_window()
        self.assertFalse(controller.plans)
        self.assertFalse(controller.expert_manager.calls)
        self.assertEqual(controller.expert_manager.specter_async_demand_policy, 'resident')

    def test_adaptive_calls_original_implementation(self):
        controller = Controller()
        original = dict(vars(controller))
        with PrefetchPolicy(controller, 'adaptive'):
            controller.begin_window(2)
            controller.after_draft_step(1)
            controller.end_window()
        self.assertEqual(controller.original_steps, [1])
        self.assertNotIn('_benchmark_prefetch_policy', vars(controller))
        self.assertEqual(controller.begin_window.__func__, Controller.begin_window)

    def test_abort_does_not_submit_late_prediction_and_restores_methods(self):
        for explicit in (False, True):
            controller = Controller()
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with PrefetchPolicy(controller, 'late'):
                    controller.begin_window(4)
                    controller.update_from_bound_routes()
                    controller.after_draft_step(0)
                    if explicit:
                        controller.abort_window()
                    raise RuntimeError('injected')
            self.assertFalse(controller.plans)
            self.assertEqual(controller.phase, 'idle')
            self.assertNotIn('before_verify', vars(controller))
            self.assertNotIn('_benchmark_prefetch_policy', vars(controller))

    def test_multiple_windows_reset_and_invalid_policy(self):
        controller = Controller()
        with PrefetchPolicy(controller, 'early'):
            for _ in range(2):
                controller.begin_window(1)
                controller.update_from_bound_routes()
                controller.after_draft_step(0)
                controller.before_verify()
                controller.end_window()
        self.assertEqual(controller.plans, [1, 1])
        with self.assertRaises(ValueError):
            PrefetchPolicy(controller, 'lru-s')


if __name__ == '__main__':
    unittest.main()
