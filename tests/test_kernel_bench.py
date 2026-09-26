"""CLI contracts and actual local CUDA projection correctness checks."""
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import kernel_bench

try:
    import torch
except ImportError:
    torch = None


class KernelEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='specter-kernel-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / 'config.toml'
        self.config.write_text('[paths]\ncache = "cache"\noutput = "results"\n'
                               '[runtime]\ncpu_node = 2\nmemory_node = 3\n', encoding='utf-8')

    def test_invalid_shapes_and_counts_are_rejected(self):
        case = kernel_bench.KernelCase()
        for change in ({'tokens': 0}, {'hidden': 192}, {'features': 65},
                       {'features': 64}, {'features': 192},
                       {'top_k': 9}, {'warmup': 0}, {'trials': -1},
                       {'device': 'cpu'}, {'device': 'cudabogus'}, {'seed': -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(case, **change).validate()
        self.assertIs(case.validate(), case)
        self.assertEqual(replace(case, hidden=2048, features=1408, experts=64, top_k=6).validate().features, 1408)

    def test_dry_run_imports_no_torch_and_creates_no_output(self):
        code = ('import sys; from benchmarks.kernel_bench import main; '
                'main(sys.argv[1:]); assert "torch" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-c', code,
                                 '--config', str(self.config), '--dry-run', '--check-numa'],
                                cwd=Path(kernel_bench.__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual([case['tokens'] for case in plan['cases']], [1, 16, 128])
        self.assertEqual(plan['paths'][0], 'local_w4a16_batched')
        self.assertTrue(plan['check_numa'])
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / 'cache').exists())

    def test_ae_forwards_numa_flag_to_kernel_and_optional_experiments(self):
        from benchmarks import run_ae
        from benchmarks.experiment_plan import EXPERIMENTS
        for experiment in ('kernel', *EXPERIMENTS):
            for checked in (False, True):
                with self.subTest(experiment=experiment, checked=checked):
                    [(label, command)] = run_ae.build_commands(
                        experiment, self.config, self.root / 'ae', check_numa=checked)
                    self.assertEqual(label, experiment)
                    self.assertEqual('--check-numa' in command, checked)
                    if experiment == 'kernel':
                        args = kernel_bench.build_parser().parse_args(command[4:])
                        self.assertEqual(args.check_numa, checked)

    def test_numa_failure_precedes_output_creation_and_benchmark(self):
        from benchmarks import numa
        with patch.dict(os.environ), patch.object(kernel_bench, 'run_case') as run:
            with patch.object(numa, 'check_numa', side_effect=RuntimeError('wrong placement')) as check:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'wrong placement'):
                    kernel_bench.main(['--config', str(self.config), '--check-numa'])
        check.assert_called_once_with({'cpu_node': 2, 'memory_node': 3})
        run.assert_not_called()
        self.assertFalse((self.root / 'results').exists())

    def test_numa_is_opt_in_and_records_best_effort_placement(self):
        from benchmarks import numa
        for checked in (False, True):
            with self.subTest(checked=checked), patch.dict(os.environ):
                with patch.object(numa, 'check_numa') as check, patch.object(numa, 'probe_numa', return_value='placement metadata'):
                    with patch.object(kernel_bench, 'run_case', return_value={'checked': True}):
                        with contextlib.redirect_stdout(io.StringIO()):
                            result = kernel_bench.main(['--config', str(self.config), '--tokens', '1',
                                '--output', f'numa_{checked}'] + (['--check-numa'] if checked else []))
                self.assertEqual(check.call_count, int(checked))
                self.assertEqual(result['plan']['numa_policy'], 'placement metadata')
                saved = json.loads((self.root / 'results' / f'numa_{checked}' / 'plan.json').read_text())
                self.assertEqual(saved['check_numa'], checked)
                self.assertEqual(saved['numa_policy'], 'placement metadata')

    def test_existing_output_fails_before_benchmark(self):
        (self.root / 'results' / 'kernel').mkdir(parents=True)
        with patch.dict(os.environ), patch.object(kernel_bench, 'run_case') as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                kernel_bench.main(['--config', str(self.config)])
        run.assert_not_called()

    def test_partial_failure_never_sets_complete(self):
        with patch.dict(os.environ), patch.object(kernel_bench, 'run_case', side_effect=AssertionError('bad output')):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
                kernel_bench.main(['--config', str(self.config), '--tokens', '1'])
        output = self.root / 'results' / 'kernel'
        self.assertEqual(json.loads((output / 'failure.json').read_text())['completed_cases'], 0)
        self.assertFalse((output / 'results.json').exists())


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires the local CUDA kernels')
class KernelCudaTests(unittest.TestCase):
    def test_real_paths_match_reference_for_sparse_and_multirow_routes(self):
        for tokens in (1, 17):
            with self.subTest(tokens=tokens):
                case = kernel_bench.KernelCase(tokens=tokens, warmup=1, iterations=2, trials=2)
                result = kernel_bench.run_case(case)
                self.assertEqual(sum(result['routing']['expert_counts']), tokens * 2)
                self.assertTrue(all(check['passed'] for check in result['correctness'].values()))
                self.assertEqual(len(result['timings']), 3)
                for timing in result['timings'].values():
                    self.assertEqual(len(timing['samples_ms']), 2)
                    self.assertGreater(timing['median_ms'], 0)

    def test_correctness_failure_happens_before_timing(self):
        case = kernel_bench.KernelCase(tokens=1, warmup=1, iterations=1, trials=1)
        with patch.object(kernel_bench, '_compare', side_effect=AssertionError('incorrect')):
            with patch.object(torch.cuda, 'Event') as event:
                with self.assertRaisesRegex(AssertionError, 'incorrect'):
                    kernel_bench.run_case(case)
                event.assert_not_called()

    def test_fixed_seed_recreates_identical_inputs_and_routes(self):
        case = kernel_bench.KernelCase(tokens=3)
        with torch.cuda.device(case.device), torch.inference_mode():
            first = kernel_bench._prepare(case)
            expected = first['paths']['local_w4a16_batched']().clone()
            second = kernel_bench._prepare(case)
            actual = second['paths']['local_w4a16_batched']().clone()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(first['routing'], second['routing'])


if __name__ == '__main__':
    unittest.main()
