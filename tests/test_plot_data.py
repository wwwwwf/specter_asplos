"""Statistical and completeness checks for plotting measured result files."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from benchmarks.plot_experiments import (
    describe, js_distance, load_results, main, prepare_tables,
    render_tables, route_observations, routing_distances,
)


def synthetic_record(dataset='GK', index=0):
    case = {'label': dataset, 'dataset': dataset, 'depth': 2, 'resident': 4,
            'policy': 'adaptive', 'kv_mode': 'shared', 'expert_mode': 'overlap'}
    return {'case': case, 'sample_index': index, 'repeat': 0, 'role': 'timed',
            'tpot_ms': 5.0, 'token_ids': [1, 2, 3]}


def synthetic_diagnostic(record):
    return {**deepcopy(record), 'role': 'diagnostic', 'diagnostic': 'routing',
            'routing': {'windows': [{'status': 'complete', 'layers': [
                {'layer_id': 0, 'draft_ids': [[0, 1], [0, 1]], 'target_ids': [[0, 1], [1, 2]]},
                {'layer_id': 1, 'draft_ids': [[0, 1], [2, 3]], 'target_ids': [[0, 1], [0, 1]]},
            ]}]}}


class PlotStatisticsTests(unittest.TestCase):
    def test_distribution_uses_all_layer_window_samples_and_linear_quantiles(self):
        result = describe([1, 2, 3, 10])
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['mean'], 4)
        self.assertAlmostEqual(result['p10'], 1.3)
        self.assertAlmostEqual(result['p90'], 7.9)
        for invalid in ([], [float('nan')], [-1], [True]):
            with self.assertRaises(ValueError):
                describe(invalid)

    def test_js_identity_disjoint_support_symmetry_and_scale_invariance(self):
        self.assertEqual(js_distance({0: 2, 1: 3}, {0: 20, 1: 30}), 0)
        self.assertEqual(js_distance({0: 2}, {1: 300}), 1)
        left, right = {0: 2, 1: 1}, {0: 1, 2: 2}
        self.assertAlmostEqual(js_distance(left, right), js_distance(right, left))
        self.assertAlmostEqual(js_distance(left, right), js_distance({0: 20, 1: 10}, right))
        self.assertGreater(js_distance(left, right), 0)
        self.assertLess(js_distance(left, right), 1)
        for invalid in ({}, {0: 0}, {0: -1}):
            with self.assertRaises(ValueError):
                js_distance(invalid, {1: 1})

    def test_layerwise_js_has_equal_layer_weight_despite_unequal_assignments(self):
        # Layer zero dominates raw assignments but is identical across domains.
        observations = [
            {'case': {'dataset': dataset}, 'layer': layer,
             'counts': {'draft': counts, 'target': counts}}
            for dataset, layer, counts in [('A', 0, {0: 1000}), ('A', 1, {1: 1}),
                                           ('B', 0, {0: 10}), ('B', 1, {0: 5})]
        ]
        frequencies, layers, distances = routing_distances(observations)
        target = [row for row in distances if row['role'] == 'target' and row['dataset_a'] == 'A' and row['dataset_b'] == 'B'][0]
        self.assertEqual(target['distance'], 0.5)
        per_layer = [row['distance'] for row in layers if row['role'] == 'target' and row['dataset_a'] == 'A' and row['dataset_b'] == 'B']
        self.assertEqual(per_layer, [0, 1])
        self.assertTrue(all(row['probability'] == 1 for row in frequencies))

    def test_frequency_pooling_is_assignment_weighted_within_each_layer(self):
        observations = [
            {'case': {'dataset': 'A'}, 'layer': 0, 'counts': {'draft': counts, 'target': counts}}
            for counts in ({0: 9}, {1: 1})
        ]
        frequencies, _, _ = routing_distances(observations)
        values = {row['expert']: row['probability'] for row in frequencies if row['role'] == 'target'}
        self.assertEqual(values, {0: 0.9, 1: 0.1})

    def test_working_sets_and_recall_are_derived_from_actual_ids(self):
        rows = route_observations([synthetic_diagnostic(synthetic_record())])
        self.assertEqual([len(row['counts']['target']) for row in rows], [3, 2])
        self.assertEqual([row['recall'] for row in rows], [2 / 3, 1])
        bundle = {'experiment': 'routing', 'records': [synthetic_record()],
                  'diagnostics': [synthetic_diagnostic(synthetic_record())]}
        table = prepare_tables(bundle)['working_set']['rows']
        self.assertEqual({row['role']: row['mean'] for row in table}, {'draft': 3, 'target': 2.5})
        self.assertEqual({row['count'] for row in table}, {2})

    def test_route_abort_missing_layers_empty_rows_and_duplicate_ids_fail(self):
        for mutation in ('abort', 'missing', 'empty', 'duplicate'):
            diagnostic = synthetic_diagnostic(synthetic_record())
            windows = diagnostic['routing']['windows']
            if mutation == 'abort':
                windows[0]['status'] = 'aborted'
            elif mutation == 'missing':
                windows.append(deepcopy(windows[0]))
                windows[1]['layers'].pop()
            elif mutation == 'empty':
                windows[0]['layers'][0]['draft_ids'] = []
            else:
                windows[0]['layers'][0]['draft_ids'] = [[0, 0]]
            with self.assertRaises(ValueError):
                route_observations([diagnostic])

    def test_oracle_does_not_replace_missing_measured_precision_with_injected_rate(self):
        record = synthetic_record()
        record['case']['error_rate'] = 0.25
        record['oracle_output_matches'] = True
        record['prefetch_accuracy'] = None
        with self.assertRaisesRegex(ValueError, 'actual prefetch precision'):
            prepare_tables({'experiment': 'oracle', 'records': [record],
                            'diagnostics': [synthetic_diagnostic(record)]})

    def test_latency_uses_raw_unions_and_excludes_prefill_and_tail(self):
        diagnostic = synthetic_diagnostic(synthetic_record())
        events = [
            {'kind': kind, 'iteration': iteration, 'phase': phase,
             'start_ms': begin, 'end_ms': end}
            for kind, iteration, phase, begin, end in [
                ('h2d', -1, 'prefill', 0, 100),
                ('draft_window', 0, 'draft', 100, 110),
                ('target_verify', 0, 'verify', 110, 130),
                ('h2d', 0, 'draft', 105, 115),
                ('h2d', 0, 'verify', 110, 120),
                ('demand_wait', 0, 'verify', 110, 115),
                ('prefetch_wait', 0, 'prefetch_barrier', 109, 112),
                ('h2d', 0, 'between_windows', 130, 500),
            ]
        ]
        diagnostic['latency'] = {'failed': False, 'records': events}
        rows = prepare_tables({'experiment': 'latency', 'records': [synthetic_record()],
                               'diagnostics': [diagnostic]})['latency']['rows']
        self.assertEqual({row['metric']: row['mean'] for row in rows}, {
            'Draft window': 10, 'Target forward': 20, 'H2D copies': 15, 'Compute-stream waits': 6,
        })

    def test_memory_retained_and_peak_are_separate_measurements(self):
        record = synthetic_record()
        record['peak_allocated_bytes'] = 7 * 2**30
        diagnostic = synthetic_diagnostic(record)
        report = {'complete': True, 'by_device': {'cuda:0': 2**30, 'cpu': 9 * 2**30},
                  'models': {'shared_by_device': {'cuda:0': 2**29}},
                  'cache': {'by_device': {'cuda:0': 2**20}, 'shared_by_device': {'cuda:0': 2**19}}}
        diagnostic['memory'] = [{'label': stage, 'report': deepcopy(report)}
                                for stage in ('loaded', 'prefill', 'first_commit')]
        tables = prepare_tables({'experiment': 'memory', 'records': [record], 'diagnostics': [diagnostic]})
        self.assertEqual({row['mean'] for row in tables['memory_storage']['rows']}, {1})
        self.assertEqual({row['mean'] for row in tables['memory_kv']['rows']}, {1})
        self.assertEqual(tables['memory_peak']['rows'][0]['mean'], 7)
        diagnostic['memory'][1]['report']['complete'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            prepare_tables({'experiment': 'memory', 'records': [record], 'diagnostics': [diagnostic]})

    def test_ablation_names_preserve_cumulative_changes(self):
        record = synthetic_record()
        record['case'].update(policy='none', kv_mode='copied', expert_mode='serial')
        rows = prepare_tables({'experiment': 'ablation', 'records': [record],
                               'diagnostics': [synthetic_diagnostic(record)]})['ablation']['rows']
        self.assertEqual(rows[0]['variant'], 'No prefetch + copied KV + serial demand')

    def test_fidelity_histogram_weights_token_layer_pairs(self):
        record = synthetic_record()
        record['fidelity'] = {'token_layer_count': 4, 'layers': [
            {'layer_id': 0, 'token_count': 2, 'token_target_recall': [0, 1]},
            {'layer_id': 1, 'token_count': 2, 'token_target_recall': [1, 1]},
        ]}
        tables = prepare_tables({'experiment': 'fidelity', 'records': [record], 'diagnostics': []})
        histogram = {row['recall']: row['fraction'] for row in tables['fidelity_histogram']['rows']}
        self.assertEqual(histogram, {0: 0.25, 1: 0.75})
        record['fidelity']['token_layer_count'] = 5
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            prepare_tables({'experiment': 'fidelity', 'records': [record], 'diagnostics': []})

    def test_numeric_tick_labels_and_legend_do_not_overlap_the_plot(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Matplotlib is required for the rendering regression')
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = [{'recall': index / 6, 'fraction': 0.2, 'dataset': dataset}
                for index in range(7) for dataset in ('A', 'B')]
        payload = {'numeric_labels': {'rows': rows, 'notes': ['Synthetic rendering fixture.'],
                   'plot': {'kind': 'bars', 'x': 'recall', 'y': 'fraction', 'group': 'dataset',
                            'xlabel': 'Matched-token recall', 'ylabel': 'Fraction of pairs'}}}
        close = plt.close
        inspected = []

        def inspect_and_close(figure):
            if not hasattr(figure, 'canvas'):
                return close(figure)
            try:
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                axes = figure.axes[0]
                boxes = [label.get_window_extent(renderer) for label in axes.get_xticklabels()]
                self.assertTrue(all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:])))
                self.assertGreater(axes.get_legend().get_window_extent(renderer).y0,
                                   axes.get_window_extent(renderer).y1)
                inspected.append(True)
            finally:
                close(figure)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'preview'
            with mock.patch.object(plt, 'close', side_effect=inspect_and_close):
                render_tables(payload, output)
            for suffix in ('pdf', 'png', 'json', 'csv'):
                self.assertGreater((output / f'numeric_labels.{suffix}').stat().st_size, 100)
        self.assertEqual(inspected, [True])

    def test_module_import_does_not_load_torch_or_matplotlib(self):
        result = subprocess.run([sys.executable, '-c',
                                 "import sys; import benchmarks.plot_experiments; assert 'torch' not in sys.modules; assert 'matplotlib' not in sys.modules"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PlotInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        record = synthetic_record()
        self.files = {
            'config.json': {'experiment': 'routing', 'expected_records': 1,
                            'cases': [record['case']], 'args': {'num_data': 1, 'repeats': 1}},
            'summary.json': {'complete': True, 'expected_records': 1, 'completed_records': 1,
                             'rows': [{'label': 'GK', 'samples': 1}]},
            'records.jsonl': [record], 'diagnostics.jsonl': [synthetic_diagnostic(record)],
        }

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        for filename, value in self.files.items():
            content = ''.join(json.dumps(row) + '\n' for row in value) if filename.endswith('jsonl') else json.dumps(value)
            (self.directory / filename).write_text(content, encoding='utf-8')

    def test_complete_paired_run_loads(self):
        self.write()
        self.assertEqual(load_results(self.directory)['experiment'], 'routing')

    def test_incomplete_summary_truncated_diagnostics_and_absent_data_fail(self):
        for mutation in ('incomplete', 'truncated', 'empty', 'missing_pair', 'mismatch'):
            with self.subTest(mutation=mutation):
                original = deepcopy(self.files)
                if mutation == 'incomplete':
                    self.files['summary.json']['complete'] = False
                elif mutation == 'truncated':
                    self.files['summary.json']['expected_records'] = 2
                elif mutation == 'empty':
                    self.files['records.jsonl'] = []
                elif mutation == 'missing_pair':
                    self.files['diagnostics.jsonl'] = []
                else:
                    self.files['diagnostics.jsonl'][0]['token_ids'] = [9]
                self.write()
                with self.assertRaises(ValueError):
                    load_results(self.directory)
                self.files = original

    def test_shifted_sample_keys_and_nonfinite_json_fail(self):
        self.files['records.jsonl'][0]['sample_index'] = 2
        self.files['diagnostics.jsonl'][0]['sample_index'] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, 'sample/repeat'):
            load_results(self.directory)
        self.files['records.jsonl'][0]['tpot_ms'] = float('nan')
        self.write()
        with self.assertRaisesRegex(ValueError, 'Invalid JSON number'):
            load_results(self.directory)

    def test_existing_output_is_never_overwritten(self):
        self.write()
        output = self.directory / 'figures'
        output.mkdir()
        sentinel = output / 'keep.txt'
        sentinel.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            main(['--input', str(self.directory)])
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')

    def test_incomplete_kernel_report_fails(self):
        (self.directory / 'results.json').write_text(json.dumps({'complete': False, 'results': []}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            load_results(self.directory)


if __name__ == '__main__':
    unittest.main()
