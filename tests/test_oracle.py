"""Offline prediction perturbations and scoped replay validation."""
import copy
from types import SimpleNamespace
import unittest

from benchmarks.oracle import OraclePrefetch, build_oracle_plans


def windows():
    return [{'window_id': 0, 'draft_len': 2, 'status': 'complete', 'layers': [
        {'layer_id': 1, 'target_ids': [[0, 1], [0, 2], [0, 3]]},
        {'layer_id': 2, 'target_ids': [[2, 3], [2, 4], [2, 5]]},
    ]}]


class Controller:
    def __init__(self):
        self.layer_num = 2
        self.skip_fisrt_layer = True
        self.max_cached_experts = 2
        self.num_experts = 8
        self.started = []

    def begin_window(self, draft_len):
        self.started.append(draft_len)

    def get_all_layer_topm_uids(self, count):
        return [('original', count)]


class OracleTests(unittest.TestCase):
    def test_frequency_selection_is_unique_bounded_and_deterministic(self):
        source = windows()
        original = copy.deepcopy(source)
        plans = build_oracle_plans(source, num_experts=8, resident=2, seed=42, error_rate=0)
        self.assertEqual(source, original)
        self.assertEqual(plans[0]['layers'][0]['selected'], [0, 1])
        self.assertEqual(plans[0]['layers'][1]['selected'], [2, 3])
        self.assertTrue(all(layer['accuracy'] == 1 for layer in plans[0]['layers']))
        noisy = build_oracle_plans(source, 8, 2, 42, 1)
        self.assertEqual(noisy, build_oracle_plans(source, 8, 2, 42, 1))
        for layer in noisy[0]['layers']:
            self.assertLessEqual(len(layer['selected']), 2)
            self.assertEqual(len(set(layer['selected'])), len(layer['selected']))
            self.assertTrue(all(0 <= expert < 8 for expert in layer['selected']))

    def test_uniform_replacement_can_remain_correct(self):
        source = [{'window_id': 0, 'draft_len': 1, 'status': 'complete',
                   'layers': [{'layer_id': 0, 'target_ids': [[0], [0]]}]}]
        plan = build_oracle_plans(source, 1, 1, 1, 1)[0]['layers'][0]
        self.assertEqual(plan['replacements'], [[0, 0]])
        self.assertEqual(plan['selected'], [0])
        self.assertEqual(plan['accuracy'], 1)

    def test_malformed_rates_capacities_and_expert_ids_are_rejected(self):
        for rate in (float('nan'), float('inf'), -0.01, 1.01):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                build_oracle_plans(windows(), 8, 2, 1, rate)
        for capacity in (0, -1, 9):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                build_oracle_plans(windows(), 8, capacity, 1, 0)
        for expert in (-1, 8):
            source = windows()
            source[0]['layers'][0]['target_ids'][0][0] = expert
            with self.subTest(expert=expert), self.assertRaises(ValueError):
                build_oracle_plans(source, 8, 2, 1, 0)

    def test_empty_source_is_rejected_and_unused_plan_has_no_accuracy(self):
        with self.assertRaisesRegex(ValueError, 'completed'):
            build_oracle_plans([], 8, 2, 1, 0)
        with OraclePrefetch(Controller(), []) as oracle:
            pass
        self.assertIsNone(oracle.result()['prefetch_accuracy'])
        self.assertEqual(oracle.result()['used_windows'], [])

    def test_layer_indices_and_repeated_plan_reads_preserve_denominators(self):
        controller = Controller()
        plans = build_oracle_plans(windows(), 8, 2, 42, 0)
        with OraclePrefetch(controller, plans) as oracle:
            controller.begin_window(2)
            self.assertEqual(controller.get_all_layer_topm_uids(2), [[(1, 0), (1, 1)], [(2, 2), (2, 3)]])
            controller.get_all_layer_topm_uids(2)
        result = oracle.result()
        self.assertEqual(result['used_windows'], [0])
        self.assertEqual(result['prefetch_accuracy'], 1)
        self.assertNotIn('begin_window', vars(controller))

    def test_source_windows_require_complete_unique_expected_layers(self):
        source = windows()
        source[0]['status'] = 'aborted'
        with self.assertRaisesRegex(ValueError, 'aborted'):
            build_oracle_plans(source, 8, 2, 1, 0)
        source = windows()
        source[0]['layers'].append(copy.deepcopy(source[0]['layers'][0]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            build_oracle_plans(source, 8, 2, 1, 0)
        for bad_layers in ([], [windows()[0]['layers'][0]]):
            source = windows()
            source[0]['layers'] = bad_layers
            with self.subTest(layers=bad_layers), self.assertRaises(ValueError):
                build_oracle_plans(source, 8, 2, 1, 0, expected_layers=[1, 2])
        source = windows() + windows()
        source[1]['layers'].pop()
        with self.assertRaisesRegex(ValueError, 'expected'):
            build_oracle_plans(source, 8, 2, 1, 0)

    def test_source_rows_must_be_nonempty_token_by_topk_ids(self):
        for rows in ([], [[0, 0]], [[0, 1], [2]], [0, 1], [[True]], [[1.5]]):
            source = windows()
            source[0]['layers'][0]['target_ids'] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                build_oracle_plans(source, 8, 2, 1, 0)

    def test_replay_requires_begin_and_matches_recorded_draft_length(self):
        plans = build_oracle_plans(windows(), 8, 2, 42, 0)
        controller = Controller()
        with OraclePrefetch(controller, plans):
            with self.assertRaisesRegex(RuntimeError, 'active'):
                controller.get_all_layer_topm_uids(2)
        with self.assertRaisesRegex(RuntimeError, 'length'):
            with OraclePrefetch(controller, plans):
                controller.begin_window(3)
        self.assertFalse(controller.started)
        self.assertNotIn('begin_window', vars(controller))
        self.assertNotIn('get_all_layer_topm_uids', vars(controller))

    def test_exception_restores_owned_and_inherited_methods(self):
        controller = Controller()
        owned = lambda count: [('owned', count)]
        controller.get_all_layer_topm_uids = owned
        plans = build_oracle_plans(windows(), 8, 2, 42, 0)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with OraclePrefetch(controller, plans):
                controller.begin_window(2)
                raise RuntimeError('injected')
        self.assertIs(controller.get_all_layer_topm_uids, owned)
        self.assertNotIn('begin_window', vars(controller))

    def test_exceeding_recorded_windows_is_rejected_and_restored(self):
        controller = Controller()
        plans = build_oracle_plans(windows(), 8, 2, 42, 0)
        with self.assertRaisesRegex(RuntimeError, 'exceeded'):
            with OraclePrefetch(controller, plans):
                controller.begin_window(2)
                controller.begin_window(2)
        self.assertEqual(controller.started, [2])
        self.assertNotIn('begin_window', vars(controller))

    def test_context_reuse_starts_new_trace_after_success_or_error(self):
        controller = Controller()
        oracle = OraclePrefetch(controller, build_oracle_plans(windows(), 8, 2, 42, 0))
        with oracle:
            controller.begin_window(2)
            controller.get_all_layer_topm_uids(2)
        self.assertEqual(oracle.result()['used_windows'], [0])
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with oracle:
                self.assertEqual(oracle.result()['visited_windows'], 0)
                self.assertEqual(oracle.result()['used_windows'], [])
                controller.begin_window(2)
                raise RuntimeError('injected')
        with oracle:
            controller.begin_window(2)
            controller.get_all_layer_topm_uids(2)
        self.assertEqual(oracle.result()['visited_windows'], 1)
        self.assertEqual(oracle.result()['used_windows'], [0])
        self.assertEqual(controller.started, [2, 2, 2])
        self.assertNotIn('begin_window', vars(controller))


if __name__ == '__main__':
    unittest.main()
