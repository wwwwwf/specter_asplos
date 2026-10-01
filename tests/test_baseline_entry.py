"""Workload parity and result-pairing checks without loading GPU models."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baselines import compare, run, run_tpot
from baselines.protocol import TPOT_DEFINITION
from benchmarks import run_tpot as specter_runner


class BaselineEntryTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        directory = tempfile.TemporaryDirectory(prefix='baseline-entry-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / 'local.toml'
        self.config.write_text('[paths]\ncache="cache"\noutput="results"\n', encoding='utf-8')

    def test_default_commands_share_workloads_and_keep_separate_backends(self):
        # 24 is an explicit test input, not an inferred or historical low preset.
        args = run.build_parser().parse_args(['--memory', 'both', '--mixtral-low-resident-per-layer', '24'])
        commands = run.build_commands(args, self.config, self.root / 'results')
        self.assertEqual(len(commands), 6)
        for tier, subset in zip(('high', 'low'), (commands[:3], commands[3:])):
            self.assertEqual([label for label, _ in subset], [tier + '/' + method for method in run.METHODS])
            parsed = []
            for _, command in subset:
                parser = specter_runner.build_parser() if command[3] == 'benchmarks.run_tpot' else run_tpot.build_parser()
                parsed.append(parser.parse_args(command[4:]))
            for field in ('model', 'datasets', 'num_data', 'repeats', 'tokens', 'prefix_tokens'):
                self.assertTrue(all(getattr(arg, field) == getattr(parsed[0], field) for arg in parsed))
            self.assertEqual(parsed[0].offload_per_layer, 48 if tier == 'high' else 60)
            self.assertEqual(parsed[1].method, 'mixtral-offloading')
            self.assertEqual(parsed[2].method, 'specmoeoff')
            self.assertTrue(parsed[0].record_process_memory)
            self.assertIsNone(parsed[1].memory_reference)
            self.assertEqual(parsed[1].resident_per_layer, 32 if tier == 'high' else 24)
            self.assertEqual(parsed[1].buffer_size, 4)
            self.assertTrue(parsed[1].record_process_memory)
            self.assertEqual(parsed[2].memory_reference, self.root / 'results' / tier / 'specter')

    def test_high_mixtral_defaults_to_fixed32_without_a_reference(self):
        args = run.build_parser().parse_args(['--methods', 'mixtral-offloading'])
        [(label, command)] = run.build_commands(args, self.config, self.root)
        self.assertEqual(label, 'high/mixtral-offloading')
        parsed = run_tpot.build_parser().parse_args(command[4:])
        self.assertEqual(parsed.resident_per_layer, 32)
        self.assertEqual(parsed.buffer_size, 4)
        self.assertIsNone(parsed.memory_reference)
        self.assertTrue(parsed.record_process_memory)

    def test_mixtral_explicit_single_tier_override_and_separate_low_count(self):
        for tier in ('high', 'low'):
            args = run.build_parser().parse_args(['--methods', 'mixtral-offloading',
                '--memory', tier, '--mixtral-resident-per-layer', '28'])
            command = run.build_commands(args, self.config, self.root)[0][1]
            self.assertEqual(command[command.index('--resident-per-layer') + 1], '28')
        args = run.build_parser().parse_args(['--memory', 'both', '--mixtral-resident-per-layer', '30',
            '--mixtral-low-resident-per-layer', '24'])
        self.assertEqual(run.mixtral_residencies(args), {'high': 30, 'low': 24})

    def test_mixtral_rejects_inferred_low_and_unsupported_models_before_launch(self):
        invalid = [(['--memory', 'low'], 'explicit'), (['--memory', 'both'], 'mixtral-low'),
                   (['--model', 'qwen2moe'], 'only dsv2lite'), (['--model', 'phimoe'], 'only dsv2lite'),
                   (['--mixtral-resident-per-layer', '0'], 'between 1 and 64'),
                   (['--mixtral-resident-per-layer', '65'], 'between 1 and 64')]
        for arguments, message in invalid:
            with self.subTest(arguments=arguments):
                args = run.build_parser().parse_args(arguments)
                with self.assertRaisesRegex(ValueError, message):
                    run.build_commands(args, self.config, self.root)
                with patch.object(run.subprocess, 'run') as child, \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    run.main(['--config', str(self.config), *arguments])
                child.assert_not_called()
                self.assertFalse((self.root / 'results').exists())

    def test_specmoeoff_keeps_low_and_other_models_when_mixtral_is_omitted(self):
        args = run.build_parser().parse_args(['--methods', 'specter', 'specmoeoff',
                                             '--memory', 'both', '--model', 'qwen2moe'])
        commands = run.build_commands(args, self.config, self.root)
        self.assertEqual(len(commands), 4)
        self.assertIn('--memory-reference', commands[-1][1])

    def test_matched_memory_runs_specter_first_even_if_methods_are_reordered(self):
        args = run.build_parser().parse_args(['--methods', 'specmoeoff', 'specter'])
        commands = run.build_commands(args, self.config, self.root)
        self.assertEqual(commands[0][0], 'high/specter')
        args = run.build_parser().parse_args(['--methods', 'specmoeoff'])
        with self.assertRaisesRegex(ValueError, 'reference-root'):
            run.build_commands(args, self.config, self.root)
        args.memory_policy = 'same-cache'
        self.assertNotIn('--memory-reference', run.build_commands(args, self.config, self.root)[0][1])

    def test_fixed_specmoeoff_capacity_records_memory_without_calibration(self):
        args = run.build_parser().parse_args([
            '--methods', 'specter', 'specmoeoff', '--memory-policy', 'fixed',
            '--specmoeoff-resident-per-layer', '12'])
        commands = run.build_commands(args, self.config, self.root)
        self.assertEqual([label for label, _ in commands], ['high/specter', 'high/specmoeoff'])
        specter = specter_runner.build_parser().parse_args(commands[0][1][4:])
        baseline = run_tpot.build_parser().parse_args(commands[1][1][4:])
        self.assertEqual(specter.offload_per_layer, 48)
        self.assertTrue(specter.record_process_memory)
        self.assertEqual((baseline.resident_per_layer, baseline.buffer_size), (12, 32))
        self.assertTrue(baseline.record_process_memory)
        self.assertIsNone(baseline.memory_reference)
        for field in ('model', 'datasets', 'num_data', 'repeats', 'tokens', 'prefix_tokens'):
            self.assertEqual(getattr(specter, field), getattr(baseline, field))
        args.methods = ['specmoeoff']
        self.assertEqual(len(run.build_commands(args, self.config, self.root)), 1)

    def test_fixed_specmoeoff_requires_explicit_valid_single_tier_capacity(self):
        invalid = [
            ['--memory-policy', 'fixed'],
            ['--specmoeoff-resident-per-layer', '12'],
            ['--memory-policy', 'same-cache', '--specmoeoff-resident-per-layer', '12'],
            ['--memory-policy', 'fixed', '--methods', 'specter', '--specmoeoff-resident-per-layer', '12'],
            ['--memory-policy', 'fixed', '--memory', 'both', '--specmoeoff-resident-per-layer', '12'],
            ['--memory-policy', 'fixed', '--specmoeoff-resident-per-layer', '0'],
            ['--memory-policy', 'fixed', '--specmoeoff-resident-per-layer', '65'],
            ['--memory-policy', 'fixed', '--model', 'qwen2moe', '--specmoeoff-resident-per-layer', '61'],
            ['--memory-policy', 'fixed', '--model', 'phimoe', '--specmoeoff-resident-per-layer', '17'],
        ]
        for arguments in invalid:
            arguments = ['--methods', 'specter', 'specmoeoff', *arguments]
            with self.subTest(arguments=arguments):
                args = run.build_parser().parse_args(arguments)
                with self.assertRaises(ValueError):
                    run.build_commands(args, self.config, self.root)
                with patch.dict(os.environ), patch.object(run.subprocess, 'run') as child, \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    run.main(['--config', str(self.config), *arguments])
                child.assert_not_called()
        self.assertFalse((self.root / 'results').exists())

    def test_fixed_capacity_bounds_use_each_models_expert_count(self):
        for model, maximum in (('dsv2lite', 64), ('qwen2moe', 60), ('phimoe', 16)):
            for resident in (1, maximum):
                with self.subTest(model=model, resident=resident):
                    args = run.build_parser().parse_args([
                        '--methods', 'specmoeoff', '--model', model, '--memory', 'low',
                        '--memory-policy', 'fixed', '--specmoeoff-resident-per-layer', str(resident)])
                    [(label, command)] = run.build_commands(args, self.config, self.root)
                    parsed = run_tpot.build_parser().parse_args(command[4:])
                    self.assertEqual(label, 'low/specmoeoff')
                    self.assertEqual(parsed.resident_per_layer, resident)
                    self.assertIsNone(parsed.memory_reference)

    def test_fixed_run_plan_labels_both_baselines_as_explicit_residency(self):
        output = self.root / 'results' / 'fixed'
        fake_loader = SimpleNamespace(prepare_datasets=lambda *args: {})
        with patch.dict(os.environ), patch.dict(sys.modules, {'data.loader': fake_loader}), \
             patch.object(run.subprocess, 'run') as child, patch.object(compare, 'compare_root') as comparison, \
             contextlib.redirect_stdout(io.StringIO()):
            run.main(['--config', str(self.config), '--output', str(output),
                      '--memory-policy', 'fixed', '--specmoeoff-resident-per-layer', '12'])
        plan = json.loads((output / 'plan.json').read_text())
        self.assertEqual(plan['comparison'], {'mixtral-offloading': 'fixed-residency',
                                              'specmoeoff': 'fixed-residency'})
        self.assertEqual(child.call_count, 3)
        comparison.assert_called_once_with(output, None)

    def test_quick_workload_is_forwarded_identically(self):
        args = run.build_parser().parse_args(['--datasets', 'GK', '--num-data', '1', '--repeats', '1', '--tokens', '33'])
        for _, command in run.build_commands(args, self.config, self.root):
            self.assertIn('33', command)
            self.assertNotIn('--protocol-case', command)
            self.assertEqual(command[command.index('--num-data') + 1], '1')

    def test_dry_run_does_not_import_model_libraries_or_write_output(self):
        code = ('import sys; from baselines.run import main; main(sys.argv[1:]); '
                'assert "torch" not in sys.modules; assert "datasets" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-c', code, '--config', str(self.config),
                                 '--memory', 'both', '--mixtral-low-resident-per-layer', '24', '--dry-run'], cwd=run.ROOT,
                                capture_output=True, text=True, check=True)
        self.assertIn('[low/specmoeoff]', result.stdout)
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / 'cache').exists())

    def test_conflicting_output_rejected_before_data_or_children(self):
        (self.root / 'results').mkdir()
        with patch.dict(os.environ), patch.object(run.subprocess, 'run') as child:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                run.main(['--config', str(self.config), '--output', str(self.root / 'results')])
        child.assert_not_called()

    def write_run(self, method, times=(10, 20), parent=None):
        directory = (parent or self.root) / method
        directory.mkdir(parents=True, exist_ok=True)
        config = {'args': {'method': method, 'model': 'dsv2lite', 'datasets': ['GK'],
                          'tokens': 33, 'num_data': 2, 'repeats': 1, 'prefix_tokens': 1},
                  'expected_records': 2, 'tpot_definition': TPOT_DEFINITION,
                  'model_case': {'num_experts': 64, 'offload_per_layer': 48,
                                 'gamma': 16 if method == 'specter' else 3, 'buffer_size': 32},
                  'runtime_environment': {}, 'resource_configuration_toml': 'same paths'}
        inputs = [{'dataset': 'GK', 'index': index, 'text': str(index), 'input_ids': [index]}
                  for index in range(2)]
        records = [dict(kind=method, dataset='GK', prompt_index=index, repeat=0, seed=42+index*1009,
                        generated_tokens=33, strategy='greedy', prefix_tokens=1,
                        token_ids=[index] + [2]*33, peak_allocated_bytes=1000, tpot_ms=times[index])
                   for index in range(2)]
        (directory / 'config.json').write_text(json.dumps(config))
        (directory / 'inputs.json').write_text(json.dumps(inputs))
        (directory / 'records.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
        return directory

    def write_reference(self):
        reference = self.root / 'reference'
        directory = self.write_run('specter', parent=reference / 'high')
        path = directory / 'config.json'
        config = json.loads(path.read_text())
        config['resource_configuration_toml'] = self.config.read_text(encoding='utf-8')
        path.write_text(json.dumps(config))
        return reference, directory

    def fixed_reference_arguments(self, reference, output):
        return ['--config', str(self.config), '--output', str(output),
                '--methods', 'specmoeoff', '--memory-policy', 'fixed',
                '--specmoeoff-resident-per-layer', '12', '--reference-root', str(reference),
                '--datasets', 'GK', '--num-data', '2', '--repeats', '1',
                '--tokens', '33', '--prefix-tokens', '1']

    def test_reference_mismatches_fail_before_children_or_output_creation(self):
        mutations = [('model', 'qwen2moe'), ('datasets', ['WT']), ('num_data', 3),
                     ('repeats', 2), ('tokens', 128), ('prefix_tokens', 16),
                     ('gamma', 8), ('offload_per_layer', 56), ('buffer_size', 4),
                     ('resource_configuration_toml', 'different TOML'),
                     ('incomplete', None), ('kind', 'target')]
        output = self.root / 'results' / 'reused'
        for field, value in mutations:
            with self.subTest(field=field):
                reference, directory = self.write_reference()
                path = directory / 'config.json'
                config = json.loads(path.read_text())
                if field in ('model', 'datasets', 'num_data', 'repeats', 'tokens', 'prefix_tokens'):
                    config['args'][field] = value
                elif field in ('gamma', 'offload_per_layer', 'buffer_size'):
                    config['model_case'][field] = value
                elif field == 'resource_configuration_toml':
                    config[field] = value
                path.write_text(json.dumps(config))
                if field in ('incomplete', 'kind'):
                    path = directory / 'records.jsonl'
                    records = [json.loads(line) for line in path.read_text().splitlines()]
                    if field == 'incomplete':
                        records.pop()
                    else:
                        records[0]['kind'] = value
                    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
                with patch.dict(os.environ), patch.object(run.subprocess, 'run') as child, \
                     contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    run.main(self.fixed_reference_arguments(reference, output))
                child.assert_not_called()
                self.assertFalse(output.exists())
                self.assertFalse((self.root / 'results').exists())

    def test_reference_does_not_silently_trim_five_inputs_to_three(self):
        args = run.build_parser().parse_args([
            '--methods', 'specmoeoff', '--memory-policy', 'fixed',
            '--specmoeoff-resident-per-layer', '12', '--reference-root', str(self.root / 'reference'),
            '--num-data', '3'])
        reference_config = {'args': vars(args).copy()}
        reference_config['args']['num_data'] = 5
        with patch.object(compare, 'read_run', return_value=(reference_config, [], {})), \
             self.assertRaisesRegex(ValueError, 'workload mismatch: num_data'):
            run.validate_references(args, self.config)

    def test_default_low_rejects_eight_resident_reference_before_any_child(self):
        reference = self.root / 'reference'
        directory = self.write_run('specter', parent=reference / 'low')
        path = directory / 'config.json'
        config = json.loads(path.read_text())
        config['resource_configuration_toml'] = self.config.read_text(encoding='utf-8')
        config['model_case']['offload_per_layer'] = 56
        path.write_text(json.dumps(config))
        output = self.root / 'results' / 'low_reference'
        arguments = self.fixed_reference_arguments(reference, output) + [
            '--memory', 'low', '--specmoeoff-resident-per-layer', '2']
        with patch.dict(os.environ), patch.object(run.subprocess, 'run') as child, \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(ValueError, 'Reference low configuration mismatch: offload_per_layer'):
            run.main(arguments)
        child.assert_not_called()
        self.assertFalse(output.exists())
        config['model_case']['offload_per_layer'] = 60
        path.write_text(json.dumps(config))
        run.validate_references(run.build_parser().parse_args(arguments), self.config)

    def test_all_requested_reference_tiers_are_checked_before_any_child(self):
        reference, _ = self.write_reference()
        output = self.root / 'results' / 'both'
        with patch.dict(os.environ), patch.object(run.subprocess, 'run') as child, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
            run.main(['--config', str(self.config), '--output', str(output),
                      '--methods', 'specmoeoff', '--memory', 'both', '--memory-policy', 'matched',
                      '--reference-root', str(reference), '--datasets', 'GK', '--num-data', '2',
                      '--repeats', '1', '--tokens', '33', '--prefix-tokens', '1'])
        child.assert_not_called()
        self.assertFalse(output.exists())

    def test_valid_fixed_reference_runs_baseline_and_compares_without_memory_budget(self):
        reference, _ = self.write_reference()
        output = self.root / 'results' / 'fixed_reference'

        def finish_baseline(command, **kwargs):
            self.assertEqual(command[3], 'baselines.run_tpot')
            self.assertNotIn('--memory-reference', command)
            directory = self.write_run('specmoeoff', (20, 40), parent=output / 'high')
            path = directory / 'config.json'
            config = json.loads(path.read_text())
            config['model_case']['offload_per_layer'] = 52
            config['resource_configuration_toml'] = self.config.read_text(encoding='utf-8')
            path.write_text(json.dumps(config))

        fake_loader = SimpleNamespace(prepare_datasets=lambda *args: {})
        with patch.dict(os.environ), patch.dict(sys.modules, {'data.loader': fake_loader}), \
             patch.object(run.subprocess, 'run', side_effect=finish_baseline) as child, \
             contextlib.redirect_stdout(io.StringIO()):
            run.main(self.fixed_reference_arguments(reference, output))
        self.assertEqual(child.call_count, 1)
        payload = json.loads((output / 'comparison.json').read_text())
        [result] = payload['comparisons']
        self.assertEqual(result['ratio_of_macro_tpot'], 2)
        self.assertEqual(result['baseline_resident_per_layer'], 12)
        self.assertIsNone(result['memory_matching'])

    def test_comparison_computes_measured_ratio_and_output_agreement(self):
        specter = self.write_run('specter')
        baseline = self.write_run('specmoeoff', (20, 40))
        result = compare.compare_pair(specter, baseline)
        self.assertEqual(result['ratio_of_macro_tpot'], 2)
        self.assertEqual(result['datasets'][0]['identical_output_records'], 2)

    def test_fixed_spec12_comparison_keeps_all_records_above_100ms_without_budget_gate(self):
        specter = self.write_run('specter', (110, 220))
        baseline = self.write_run('specmoeoff', (220, 440))
        config_file = baseline / 'config.json'
        config = json.loads(config_file.read_text())
        config['args']['resident_per_layer'] = 12
        config['model_case']['offload_per_layer'] = 52
        self.assertNotIn('memory_budget', config)
        config_file.write_text(json.dumps(config))
        for directory, footprint in ((specter, 10000), (baseline, 12000)):
            path = directory / 'records.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                record['gpu_memory'] = {'runtime_gpu_bytes': footprint}
            path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        result = compare.compare_pair(specter, baseline)
        self.assertEqual(result['specter_resident_per_layer'], 16)
        self.assertEqual(result['baseline_resident_per_layer'], 12)
        self.assertEqual(result['macro_specter_tpot_ms'], 165)
        self.assertEqual(result['macro_baseline_tpot_ms'], 330)
        self.assertEqual(result['ratio_of_macro_tpot'], 2)
        self.assertEqual(result['datasets'][0]['records_per_method'], 2)
        self.assertEqual(result['datasets'][0]['identical_output_records'], 2)
        self.assertEqual(result['datasets'][0]['baseline_runtime_gpu_gib'], 12000 / 2**30)
        self.assertIsNone(result['memory_matching'])

    def test_comparison_rejects_old_timing_changed_inputs_and_incomplete_runs(self):
        specter = self.write_run('specter')
        for issue in ('timing', 'inputs', 'incomplete', 'duplicate', 'seeds', 'environment'):
            with self.subTest(issue=issue):
                baseline = self.write_run('specmoeoff')
                config_file = baseline / 'config.json'
                config = json.loads(config_file.read_text())
                if issue == 'timing':
                    config['tpot_definition'] = 'end-to-end divided by N'
                elif issue == 'environment':
                    config['runtime_environment']['cpu_affinity'] = [99]
                config_file.write_text(json.dumps(config))
                if issue == 'inputs':
                    (baseline / 'inputs.json').write_text('[]')
                if issue in ('incomplete', 'duplicate', 'seeds'):
                    path = baseline / 'records.jsonl'
                    records = [json.loads(line) for line in path.read_text().splitlines()]
                    if issue == 'incomplete':
                        records.pop()
                    elif issue == 'duplicate':
                        records[1] = records[0]
                    else:
                        records[0]['seed'] += 1
                    path.write_text(''.join(json.dumps(record)+'\n' for record in records))
                with self.assertRaises(ValueError):
                    compare.compare_pair(specter, baseline)

    def test_comparison_rechecks_actual_memory_records(self):
        specter = self.write_run('specter')
        baseline = self.write_run('specmoeoff')
        for directory, value in ((specter, 10000), (baseline, 9999)):
            path = directory / 'records.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                record['gpu_memory'] = {'runtime_gpu_bytes': value}
            path.write_text(''.join(json.dumps(record)+'\n' for record in records))
        path = baseline / 'config.json'
        config = json.loads(path.read_text())
        config['memory_budget'] = {'budget_bytes': 10000, 'metric': 'test'}
        path.write_text(json.dumps(config))
        self.assertTrue(compare.compare_pair(specter, baseline)['memory_matching']['within_budget'])
        path = baseline / 'records.jsonl'
        path.write_text(path.read_text().replace('9999', '10001'))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            compare.compare_pair(specter, baseline)


if __name__ == '__main__':
    unittest.main()
