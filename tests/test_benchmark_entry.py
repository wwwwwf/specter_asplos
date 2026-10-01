"""CPU-only contract checks for the AE entry and its recorded workload."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import numa, run_ae, run_tpot
from cli import profile
from speculative_inference_controller.depth_profiler import LightweightSpeculativeDepthProfiler


class BenchmarkEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='specter-ae-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'results' / 'ae'
        self.config = self.root / 'config.toml'
        self.config.write_text(
            '[paths]\ncache = "cache"\noutput = "results"\n'
            '[datasets]\nGK = "missing-gsm8k"\n', encoding='utf-8')

    def commands(self, experiment='all', check_numa=False, memory='high'):
        return run_ae.build_commands(experiment, self.config, self.output, check_numa, memory)

    def test_entry_defaults_to_main_and_default_command_satisfies_frozen_protocol(self):
        self.assertEqual(run_ae.build_parser().parse_args([]).experiment, 'main')
        label, command = self.commands('main')[0]
        self.assertEqual(label, 'main')
        parser = run_tpot.build_parser()
        args = parser.parse_args(command[4:])
        protocol, case = run_tpot.validate_arguments(parser, args)
        self.assertEqual(case['model'], 'dsv2lite')
        self.assertEqual(case['gamma'], 16)
        self.assertEqual(case['offload_per_layer'], 48)
        self.assertEqual(len(args.datasets) * args.num_data * args.repeats, 75)
        self.assertEqual(args.tokens, 128)
        self.assertEqual(args.prefix_tokens, protocol['prefix_max_tokens'])
        self.assertFalse(args.check_numa)

    def test_low_memory_command_satisfies_protocol_and_uses_separate_output(self):
        label, command = self.commands('main', memory='low')[0]
        parser = run_tpot.build_parser()
        args = parser.parse_args(command[4:])
        _, case = run_tpot.validate_arguments(parser, args)
        self.assertEqual(label, 'main_low')
        self.assertEqual(args.output, str(self.output / label))
        self.assertEqual(args.protocol_case, 'ds_low4')
        self.assertEqual(case['resident_per_layer'], 4)
        self.assertEqual(args.offload_per_layer, 60)
        self.assertEqual(args.gamma, 16)
        self.assertEqual(len(args.datasets) * args.num_data * args.repeats, 75)

    def test_prior_low8_protocol_remains_available_explicitly(self):
        _, command = self.commands('main', memory='low')[0]
        parser = run_tpot.build_parser()
        args = parser.parse_args(command[4:])
        args.protocol_case = 'ds_low8'
        args.offload_per_layer = 56
        _, case = run_tpot.validate_arguments(parser, args)
        self.assertEqual(case['resident_per_layer'], 8)
        self.assertEqual(case['offload_per_layer'], 56)

    def test_both_memory_tiers_keep_identical_workloads_and_supporting_experiments(self):
        commands = self.commands(memory='both', check_numa=True)
        self.assertEqual([label for label, _ in commands],
                         ['main', 'main_low', 'depth', 'prefix_8', 'prefix_16', 'prefix_32'])
        main_args = []
        for _, command in commands[:2]:
            parser = run_tpot.build_parser()
            args = parser.parse_args(command[4:])
            run_tpot.validate_arguments(parser, args)
            main_args.append(vars(args))
        differing = {key for key in main_args[0] if main_args[0][key] != main_args[1][key]}
        self.assertEqual(differing, {'protocol_case', 'offload_per_layer', 'output'})
        self.assertEqual(commands[2:], self.commands(check_numa=True)[1:])
        self.assertTrue(all('--check-numa' in command for _, command in commands))

    def test_memory_option_rejects_unrelated_experiments(self):
        for experiment in ('depth', 'prefix', 'routing', 'sensitivity', 'kernel'):
            with self.subTest(experiment=experiment), self.assertRaisesRegex(ValueError, 'main or all'):
                self.commands(experiment, memory='low')

    def test_supporting_commands_have_81_depth_and_27_prefix_measurements(self):
        commands = self.commands()
        self.assertEqual([label for label, _ in commands], ['main', 'depth', 'prefix_8', 'prefix_16', 'prefix_32'])
        depth = profile.build_parser().parse_args(commands[1][1][4:])
        self.assertEqual(depth.prefix_tokens, 16)
        self.assertEqual(depth.seed, 42)
        self.assertEqual(len(LightweightSpeculativeDepthProfiler.candidates) * 3 * depth.repeats, 81)
        count = 0
        for expected_prefix, (_, command) in zip((8, 16, 32), commands[2:]):
            parser = run_tpot.build_parser()
            args = parser.parse_args(command[4:])
            self.assertEqual(run_tpot.validate_arguments(parser, args), (None, None))
            self.assertEqual(args.datasets, ['GK'])
            self.assertEqual(args.prefix_tokens, expected_prefix)
            self.assertEqual(args.gamma, 16)
            count += len(args.datasets) * args.num_data * args.repeats
        self.assertEqual(count, 27)
        self.assertIsNone(profile.build_parser().parse_args(['--prompts-json', 'prompts.json']).prefix_tokens)

    def test_protocol_rejects_subset_or_changed_prefix(self):
        _, command = self.commands('main')[0]
        for field, value in [('datasets', ['GK']), ('prefix_tokens', 32)]:
            parser = run_tpot.build_parser()
            args = parser.parse_args(command[4:])
            setattr(args, field, value)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_tpot.validate_arguments(parser, args)

    def test_numa_is_opt_in_for_every_child_and_missing_tool_is_metadata(self):
        self.assertTrue(all('--check-numa' in command for _, command in self.commands(check_numa=True)))
        self.assertTrue(all('--check-numa' not in command for _, command in self.commands()))
        with patch.object(numa.subprocess, 'check_output', side_effect=FileNotFoundError('numactl')):
            self.assertIn('unavailable:', numa.probe_numa())

    def test_subset_summary_is_complete_without_all_five_datasets(self):
        record = dict(dataset='GK', kind='specter', tpot_ms=10, ttft_ms=20,
            post_first_batch_ms_per_token=11, e2e_ms_per_token=12, first_commit_tokens=2)
        run_tpot.summarize([record] * 9, self.root, 9, ['GK'])
        summary = json.loads((self.root / 'summary.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['macro_specter_tpot_ms'], 10)
        run_tpot.summarize([record] * 8, self.root, 9, ['GK'])
        self.assertFalse(json.loads((self.root / 'summary.json').read_text())['complete'])

    def test_recorded_gk_selection_is_indexed_and_rejects_missing_or_duplicate_inputs(self):
        inputs = [dict(dataset='GK', index=index, text=f'GK {index}') for index in (3, 2, 0, 4, 1)]
        inputs.append(dict(dataset='WT', index=0, text='Unrelated'))
        self.assertEqual(run_ae.select_main_prompts(inputs), ['GK 0', 'GK 1', 'GK 2'])
        for invalid in (inputs[:3], inputs + [dict(dataset='GK', index=0, text='Duplicate')]):
            with self.assertRaises(ValueError):
                run_ae.select_main_prompts(invalid)

    def test_depth_prompt_preparation_reuses_saved_inputs_without_dataset_import(self):
        (self.output / 'main').mkdir(parents=True)
        inputs = [dict(dataset='GK', index=index, text=f'GK {index}') for index in range(5)]
        (self.output / 'main' / 'inputs.json').write_text(json.dumps(inputs), encoding='utf-8')
        # None config proves no dataset fallback is attempted for saved inputs.
        run_ae.prepare_depth_prompts(None, self.output)
        self.assertEqual(json.loads((self.output / 'depth' / 'prompts.json').read_text()), ['GK 0', 'GK 1', 'GK 2'])

    def test_dry_run_imports_no_torch_or_dataset_and_writes_no_outputs(self):
        code = (
            'import sys; from benchmarks.run_ae import main; '
            'main(sys.argv[1:]); '
            'assert "torch" not in sys.modules; '
            'assert "data.loader" not in sys.modules; '
            'assert "datasets" not in sys.modules'
        )
        result = subprocess.run([sys.executable, '-B', '-c', code, '--experiment', 'all',
            '--config', str(self.config), '--dry-run'], cwd=run_ae.ROOT,
            capture_output=True, text=True, check=True)
        self.assertIn('[main]', result.stdout)
        self.assertIn('[depth]', result.stdout)
        self.assertIn('[prefix_32]', result.stdout)
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / 'cache').exists())

    def test_existing_output_stops_before_any_child_process(self):
        (self.output / 'prefix_32').mkdir(parents=True)
        with patch.dict(os.environ), patch.object(run_ae.subprocess, 'run') as child:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                run_ae.main(['--experiment', 'all', '--config', str(self.config)])
        child.assert_not_called()

    def test_existing_low_output_stops_both_tiers_before_any_child_process(self):
        (self.output / 'main_low').mkdir(parents=True)
        with patch.dict(os.environ), patch.object(run_ae.subprocess, 'run') as child:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                run_ae.main(['--memory', 'both', '--config', str(self.config)])
        child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
