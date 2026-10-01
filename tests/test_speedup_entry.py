"""CPU-only checks for the one-command, single-model speedup comparison."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from baselines import run as baseline_run
from baselines.protocol import DATASETS
from benchmarks import run_ae


class SpeedupEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='specter-speedup-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'results' / 'ae'
        self.config = self.root / 'local.toml'
        self.config.write_text(
            '[paths]\ncache="cache"\noutput="results"\n'
            '[datasets]\nGK="missing-dataset"\n', encoding='utf-8')

    def command(self, **options):
        commands = run_ae.build_commands('speedup', self.config, self.output, **options)
        self.assertEqual(len(commands), 1)
        label, command = commands[0]
        self.assertEqual(label, 'speedup')
        self.assertEqual(command[:4], [sys.executable, '-B', '-m', 'baselines.run'])
        return baseline_run.build_parser().parse_args(command[4:])

    def test_default_is_fixed_deepseek_high_three_method_comparison(self):
        parser = run_ae.build_parser()
        self.assertEqual(parser.parse_args([]).experiment, 'main')
        selected = parser.parse_args(['--experiment', 'speedup'])
        self.assertEqual((selected.model, selected.memory), ('dsv2lite', 'high'))
        args = self.command()
        self.assertEqual(args.methods, ['specter', 'mixtral-offloading', 'specmoeoff'])
        self.assertEqual((args.model, args.memory, args.memory_policy),
                         ('dsv2lite', 'high', 'fixed'))
        self.assertEqual(args.specmoeoff_resident_per_layer, 12)
        self.assertEqual(args.datasets, list(DATASETS))
        self.assertEqual((args.num_data, args.repeats, args.tokens, args.prefix_tokens),
                         (5, 3, 128, 16))
        self.assertEqual(len(args.datasets) * args.num_data * args.repeats, 75)
        self.assertEqual(args.config, str(self.config))
        self.assertEqual(args.output, str(self.output / 'speedup'))
        children = baseline_run.build_commands(args, self.config, Path(args.output))
        self.assertEqual([label for label, _ in children],
                         ['high/specter', 'high/mixtral-offloading', 'high/specmoeoff'])
        specter, mixtral, specmoeoff = [command for _, command in children]
        self.assertEqual(specter[specter.index('--offload-per-layer') + 1], '48')
        self.assertEqual(specter[specter.index('--gamma') + 1], '16')
        for command, resident, buffers in ((mixtral, '32', '4'), (specmoeoff, '12', '32')):
            self.assertEqual(command[command.index('--resident-per-layer') + 1], resident)
            self.assertEqual(command[command.index('--buffer-size') + 1], buffers)
        for _, command in children:
            self.assertIn('--record-process-memory', command)
            self.assertNotIn('--memory-reference', command)

    def test_other_models_and_tiers_require_explicit_selection_without_mixtral(self):
        for model in ('phimoe',):
            for memory in ('high', 'low', 'both'):
                with self.subTest(model=model, memory=memory):
                    args = self.command(model=model, memory=memory)
                    self.assertEqual((args.model, args.memory), (model, memory))
                    self.assertEqual(args.methods, ['specter', 'specmoeoff'])
                    self.assertEqual(args.memory_policy, 'matched')
                    self.assertIsNone(args.specmoeoff_resident_per_layer)
                    children = baseline_run.build_commands(args, self.config, Path(args.output))
                    tiers = ['high', 'low'] if memory == 'both' else [memory]
                    self.assertEqual([label for label, _ in children],
                                     [f'{tier}/{method}' for tier in tiers
                                      for method in ('specter', 'specmoeoff')])

    def test_deepseek_reference_reuses_specter_with_both_fixed_baselines(self):
        reference = self.root / 'completed comparison'
        args = self.command(reference_root=reference)
        self.assertEqual(args.methods, ['mixtral-offloading', 'specmoeoff'])
        self.assertEqual(args.reference_root, reference)
        self.assertEqual(args.memory_policy, 'fixed')
        children = baseline_run.build_commands(args, self.config, Path(args.output))
        self.assertEqual([label for label, _ in children],
                         ['high/mixtral-offloading', 'high/specmoeoff'])
        for (_, command), resident in zip(children, ('32', '12')):
            self.assertEqual(command[command.index('--resident-per-layer') + 1], resident)
            self.assertNotIn('--memory-reference', command)

    def test_other_model_reference_reuses_specter_and_calibrates_requested_tiers(self):
        reference = self.root / 'completed comparison'
        args = self.command(model='phimoe', memory='both', reference_root=reference)
        self.assertEqual(args.methods, ['specmoeoff'])
        self.assertEqual(args.reference_root, reference)
        children = baseline_run.build_commands(args, self.config, Path(args.output))
        self.assertEqual([label for label, _ in children], ['high/specmoeoff', 'low/specmoeoff'])
        for tier, (_, command) in zip(('high', 'low'), children):
            self.assertEqual(command[command.index('--memory-reference') + 1],
                             str(reference / tier / 'specter'))

    def test_deepseek_low_uses_fixed4_2_20_and_the_full_workload(self):
        args = self.command(memory='low')
        self.assertEqual((args.model, args.memory, args.memory_policy),
                         ('dsv2lite', 'low', 'fixed'))
        self.assertEqual(args.methods, ['specter', 'mixtral-offloading', 'specmoeoff'])
        self.assertEqual(args.datasets, list(DATASETS))
        self.assertEqual((args.num_data, args.repeats, args.tokens, args.prefix_tokens),
                         (5, 3, 128, 16))
        self.assertEqual((args.specmoeoff_resident_per_layer, args.mixtral_resident_per_layer), (2, 20))
        children = baseline_run.build_commands(args, self.config, Path(args.output))
        self.assertEqual([label for label, _ in children],
                         ['low/specter', 'low/mixtral-offloading', 'low/specmoeoff'])
        specter, mixtral, specmoeoff = [command for _, command in children]
        self.assertEqual(specter[specter.index('--offload-per-layer') + 1], '60')
        self.assertEqual(specter[specter.index('--gamma') + 1], '16')
        for command, resident, buffers in ((mixtral, '20', '4'), (specmoeoff, '2', '32')):
            self.assertEqual(command[command.index('--resident-per-layer') + 1], resident)
            self.assertEqual(command[command.index('--buffer-size') + 1], buffers)
        for _, command in children:
            self.assertIn('--record-process-memory', command)
            self.assertNotIn('--memory-reference', command)

    def test_deepseek_both_runs_fixed_tiers_with_distinct_outputs(self):
        commands = run_ae.build_commands('speedup', self.config, self.output, memory='both',
                                        check_numa=True, require_greedy_match=True, num_data=3)
        self.assertEqual([label for label, _ in commands], ['speedup_high', 'speedup_low'])
        for (label, command), tier, resident, mixtral in zip(commands, ('high', 'low'), (12, 2), (32, 20)):
            args = baseline_run.build_parser().parse_args(command[4:])
            self.assertEqual(command[:4], [sys.executable, '-B', '-m', 'baselines.run'])
            self.assertEqual(args.memory, tier)
            self.assertEqual(args.memory_policy, 'fixed')
            self.assertEqual(args.specmoeoff_resident_per_layer, resident)
            self.assertEqual(args.mixtral_resident_per_layer, mixtral)
            self.assertEqual(args.output, str(self.output / label))
            self.assertEqual(args.datasets, list(DATASETS))
            self.assertEqual((args.num_data, args.repeats, args.tokens, args.prefix_tokens), (3, 3, 128, 16))
            children = baseline_run.build_commands(args, self.config, Path(args.output))
            self.assertEqual(len(children), 3)
            for child_label, child in children:
                self.assertIn('--record-process-memory', child)
                self.assertNotIn('--memory-reference', child)
                self.assertIn('--check-numa', child)
                self.assertIn('--require-greedy-match', child)
                self.assertEqual(child[child.index('--output') + 1], str(self.output / label / child_label))
                self.assertEqual(child[child.index('--num-data') + 1], '3')

    def test_deepseek_both_reuses_each_reference_tier_without_specter(self):
        reference = self.root / 'completed comparison'
        commands = run_ae.build_commands('speedup', self.config, self.output, memory='both',
                                        reference_root=reference)
        for (_, command), tier in zip(commands, ('high', 'low')):
            args = baseline_run.build_parser().parse_args(command[4:])
            self.assertEqual(args.reference_root, reference)
            self.assertEqual(args.memory, tier)
            self.assertEqual(args.methods, ['mixtral-offloading', 'specmoeoff'])
            children = baseline_run.build_commands(args, self.config, Path(args.output))
            self.assertEqual([label for label, _ in children],
                             [f'{tier}/mixtral-offloading', f'{tier}/specmoeoff'])
            for _, child in children:
                self.assertNotIn('--memory-reference', child)

    def test_qwen_high_uses_fixed15_and6_without_calibration(self):
        args = self.command(model='qwen2moe')
        self.assertEqual(args.methods, ['specter', 'specmoeoff'])
        self.assertEqual(args.memory_policy, 'fixed')
        self.assertEqual(args.specmoeoff_resident_per_layer, 6)
        children = baseline_run.build_commands(args, self.config, Path(args.output))
        self.assertEqual([label for label, _ in children], ['high/specter', 'high/specmoeoff'])
        specter, baseline = [command for _, command in children]
        self.assertEqual(specter[specter.index('--offload-per-layer') + 1], '45')
        self.assertEqual(specter[specter.index('--gamma') + 1], '8')
        self.assertEqual(baseline[baseline.index('--resident-per-layer') + 1], '6')
        for _, command in children:
            self.assertIn('--record-process-memory', command)
            self.assertNotIn('--memory-reference', command)
        reused = self.command(model='qwen2moe', reference_root=self.root / 'reference')
        self.assertEqual(reused.methods, ['specmoeoff'])
        for tier in ('low', 'both'):
            with self.subTest(memory=tier), self.assertRaises(ValueError):
                self.command(model='qwen2moe', memory=tier)

    def test_compact_workload_keeps_all_domains_and_same_sampling_for_both_methods(self):
        args = self.command(model='qwen2moe', num_data=3)
        self.assertEqual(args.datasets, list(DATASETS))
        self.assertEqual((args.num_data, args.repeats, args.tokens, args.prefix_tokens), (3, 3, 128, 16))
        children = baseline_run.build_commands(args, self.config, Path(args.output))
        for _, command in children:
            self.assertEqual(command[command.index('--num-data') + 1], '3')
            self.assertEqual(command[command.index('--repeats') + 1], '3')
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.command(num_data=value)

    def test_numa_and_strict_output_check_reach_all_measured_methods(self):
        args = self.command(check_numa=True, require_greedy_match=True)
        self.assertTrue(args.check_numa)
        self.assertTrue(args.require_greedy_match)
        for _, command in baseline_run.build_commands(args, self.config, Path(args.output)):
            self.assertIn('--check-numa', command)
            self.assertIn('--require-greedy-match', command)

    def test_speedup_options_are_rejected_for_other_experiments(self):
        options = [dict(model='qwen2moe'), dict(model='phimoe'),
                   dict(reference_root=self.root / 'old'), dict(require_greedy_match=True), dict(num_data=3)]
        for experiment in ('main', 'all', 'depth', 'prefix', 'kernel', 'routing'):
            for option in options:
                with self.subTest(experiment=experiment, option=option), self.assertRaises(ValueError):
                    run_ae.build_commands(experiment, self.config, self.output, **option)
        for arguments in (['--model', 'qwen2moe'], ['--reference-root', str(self.root / 'old')],
                          ['--require-greedy-match'], ['--num-data', '3']):
            with self.subTest(arguments=arguments), patch.dict(os.environ), \
                 patch.object(run_ae.subprocess, 'run') as child, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_ae.main(['--config', str(self.config), *arguments])
            child.assert_not_called()
        self.assertFalse((self.root / 'results').exists())

    def test_dry_run_imports_no_models_reads_no_datasets_and_writes_no_output(self):
        code = (
            'import sys; from benchmarks.run_ae import main; main(sys.argv[1:]); '
            'assert not {"torch", "transformers", "datasets", "data.loader"} & set(sys.modules)'
        )
        result = subprocess.run(
            [sys.executable, '-B', '-c', code, '--experiment', 'speedup',
             '--config', str(self.config), '--dry-run'],
            cwd=run_ae.ROOT, capture_output=True, text=True, check=True)
        self.assertIn('[speedup]', result.stdout)
        self.assertIn('baselines.run', result.stdout)
        self.assertIn('--methods specter mixtral-offloading specmoeoff', result.stdout)
        self.assertIn('--memory-policy fixed', result.stdout)
        self.assertIn('--specmoeoff-resident-per-layer 12', result.stdout)
        self.assertNotIn('qwen2moe', result.stdout)
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / 'cache').exists())
        compact = subprocess.run(
            [sys.executable, '-B', '-c', code, '--experiment', 'speedup',
             '--config', str(self.config), '--model', 'qwen2moe', '--num-data', '3', '--dry-run'],
            cwd=run_ae.ROOT, capture_output=True, text=True, check=True)
        self.assertIn('--methods specter specmoeoff', compact.stdout)
        self.assertIn('--num-data 3', compact.stdout)
        self.assertIn('--specmoeoff-resident-per-layer 6', compact.stdout)
        self.assertNotIn('mixtral-offloading', compact.stdout)
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / 'cache').exists())
        both = subprocess.run(
            [sys.executable, '-B', '-c', code, '--experiment', 'speedup',
             '--config', str(self.config), '--memory', 'both', '--dry-run'],
            cwd=run_ae.ROOT, capture_output=True, text=True, check=True)
        self.assertIn('[speedup_high]', both.stdout)
        self.assertIn('[speedup_low]', both.stdout)
        self.assertEqual(both.stdout.count('--memory-policy fixed'), 2)
        self.assertIn('--specmoeoff-resident-per-layer 12', both.stdout)
        self.assertIn('--specmoeoff-resident-per-layer 2', both.stdout)
        self.assertIn('--mixtral-resident-per-layer 20', both.stdout)
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / 'cache').exists())

    def test_existing_speedup_output_is_rejected_before_any_child(self):
        (self.output / 'speedup').mkdir(parents=True)
        with patch.dict(os.environ), patch.object(run_ae.subprocess, 'run') as child, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            run_ae.main(['--experiment', 'speedup', '--config', str(self.config)])
        child.assert_not_called()

    def test_existing_low_output_rejects_both_before_starting_high(self):
        (self.output / 'speedup_low').mkdir(parents=True)
        with patch.dict(os.environ), patch.object(run_ae.subprocess, 'run') as child, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            run_ae.main(['--experiment', 'speedup', '--memory', 'both', '--config', str(self.config)])
        child.assert_not_called()
        self.assertFalse((self.output / 'speedup_high').exists())

    def test_all_keeps_its_existing_workflows_without_adding_speedup(self):
        commands = run_ae.build_commands('all', self.config, self.output)
        self.assertEqual([label for label, _ in commands],
                         ['main', 'depth', 'prefix_8', 'prefix_16', 'prefix_32'])
        self.assertTrue(all('baselines.run' not in command for _, command in commands))


if __name__ == '__main__':
    unittest.main()
