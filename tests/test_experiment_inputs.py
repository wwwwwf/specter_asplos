"""Experiment candidate pools stay distinct from measured sample counts."""
import contextlib
import csv
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

from benchmarks.experiment_plan import build_cases, select_inputs
from benchmarks.run_experiments import build_parser, load_candidate_prompts


class Tokenizer:
    def encode(self, text):
        return [int(token) for token in text.split()]


class ExperimentInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='specter-experiment-inputs-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write_json(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding='utf-8')
        return path

    def release_datasets(self, qualifying=3):
        # Match the shipped formats and counts without downloading data/models.
        prompts = {name: ['1'] * count for name, count in
                   (('C4', 1000), ('HE', 164), ('GP', 448))}
        for values in prompts.values():
            for index in [20, 75, len(values) - 1][:qualifying]:
                values[index] = f' {index} 2 3 4 '
        paths = {'C4': self.write_json('c4.json', prompts['C4']),
                 'HE': self.root / 'humaneval', 'GP': self.root / 'gpqa.csv'}
        paths['HE'].mkdir(exist_ok=True)
        with paths['GP'].open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['Question', 'Pre-Revision Question'])
            writer.writeheader()
            for prompt in prompts['GP']:
                writer.writerow({'Question': 'Wrong column', 'Pre-Revision Question': prompt})
        saved = {'test': [{'prompt': prompt} for prompt in prompts['HE']]}
        dataset_patch = patch.dict(sys.modules, {'datasets': SimpleNamespace(load_from_disk=lambda _: saved)})
        dataset_patch.start()
        self.addCleanup(dataset_patch.stop)
        return SimpleNamespace(path=lambda section, name: paths[name]), prompts

    def arguments(self, experiment, *options):
        return build_parser().parse_args([
            '--experiment', experiment, '--prefix-tokens', '2',
            '--prefix-lengths', '2', '4', *options,
        ])

    def load(self, config, args, cases):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_candidate_prompts(config, args, cases)

    def test_release_pools_below_1024_keep_three_later_qualifying_inputs(self):
        config, expected = self.release_datasets()
        for experiment in ('sensitivity', 'routing'):
            args = self.arguments(experiment, '--datasets', 'C4', 'HE', 'GP')
            cases = build_cases(args)
            raw = self.load(config, args, cases)
            self.assertEqual(list(raw), ['C4', 'HE', 'GP'])
            for dataset, prompts in raw.items():
                with self.subTest(experiment=experiment, dataset=dataset):
                    self.assertEqual(prompts, expected[dataset])
                    required = max(case.prefix_tokens for case in cases if case.dataset == dataset)
                    selected = select_inputs(prompts, Tokenizer(), args.num_data, required)
                    self.assertEqual(len(selected), 3)
                    self.assertEqual([sample['source_index'] for sample in selected],
                                     [20, 75, len(prompts) - 1])

    def test_final_selection_still_requires_three_length_qualifying_inputs(self):
        config, _ = self.release_datasets(qualifying=2)
        for experiment in ('sensitivity', 'routing'):
            args = self.arguments(experiment, '--datasets', 'C4', 'HE', 'GP')
            cases = build_cases(args)
            for dataset, prompts in self.load(config, args, cases).items():
                required = max(case.prefix_tokens for case in cases if case.dataset == dataset)
                with self.subTest(experiment=experiment, dataset=dataset), self.assertRaisesRegex(
                        ValueError, rf'Need 3 nonempty inputs with at least {required} tokens; found 2'):
                    select_inputs(prompts, Tokenizer(), args.num_data, required)

    def test_candidate_pool_limit_is_preserved_for_larger_datasets(self):
        prompts = [f'{index} 2 3 4' for index in range(1100)]
        path = self.write_json('large.json', prompts)
        config = SimpleNamespace(path=lambda section, name: path)
        args = self.arguments('routing', '--datasets', 'C4')
        raw = self.load(config, args, build_cases(args))
        self.assertEqual(raw['C4'], prompts[:1024])

    def test_small_pools_use_configured_count_without_repeating_inputs(self):
        for experiment in ('sensitivity', 'routing'):
            for count in (3, 5):
                with self.subTest(experiment=experiment, count=count):
                    prompts = [f'{index} 2 3 4' for index in range(count)]
                    path = self.write_json('small.json', prompts)
                    config = SimpleNamespace(path=lambda section, name: path)
                    args = self.arguments(experiment, '--datasets', 'C4',
                                          '--num-data', str(count))
                    cases = build_cases(args)
                    raw = self.load(config, args, cases)
                    required = max(case.prefix_tokens for case in cases)
                    selected = select_inputs(raw['C4'], Tokenizer(), count, required)
                    self.assertEqual([sample['text'] for sample in selected], prompts)
                    self.assertEqual([sample['source_index'] for sample in selected],
                                     list(range(count)))

    def test_small_pools_preserve_later_qualifying_source_indices(self):
        for count in (3, 5):
            prompts = ['1', '2', *[f'{index} 2 3 4' for index in range(count)]]
            path = self.write_json('small-offset.json', prompts)
            config = SimpleNamespace(path=lambda section, name: path)
            args = self.arguments('sensitivity', '--datasets', 'C4',
                                  '--num-data', str(count))
            cases = build_cases(args)
            raw = self.load(config, args, cases)
            required = max(case.prefix_tokens for case in cases)
            selected = select_inputs(raw['C4'], Tokenizer(), count, required)
            self.assertEqual([sample['source_index'] for sample in selected],
                             list(range(2, count + 2)))

    def test_custom_json_retains_original_indices_and_text(self):
        prompts = ['', '1', '   ', ' 1 2 3 4 ', '2 3 4 5', '3 4 5 6']
        path = self.write_json('custom.json', prompts)
        args = self.arguments('sensitivity', '--prompts-json', str(path))
        raw = self.load(None, args, build_cases(args))
        self.assertEqual(raw['CUSTOM'], prompts)
        selected = select_inputs(raw['CUSTOM'], Tokenizer(), args.num_data, 4)
        self.assertEqual([sample['source_index'] for sample in selected], [3, 4, 5])

    @unittest.skipUnless(sys.version_info >= (3, 11), 'the runtime requires Python 3.11')
    def test_bad_pools_fail_before_heavy_imports_and_output_creation(self):
        code = """
import importlib.abc
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'initialization', 'speculative_inference_controller'
        }:
            raise AssertionError('Heavy import before data validation: ' + fullname)
        return None

sys.meta_path.insert(0, BlockHeavyImports())
from benchmarks.run_experiments import main
main()
"""
        malformed = self.root / 'malformed.json'
        malformed.write_text('{', encoding='utf-8')
        paths = [self.root / 'missing.json', malformed,
                 self.write_json('empty.json', []),
                 self.write_json('blank.json', [' ', '']),
                 self.write_json('insufficient.json', ['1 2 3 4', '', '5 6 7 8']),
                 self.write_json('invalid.json', ['1 2 3', 42])]
        config = self.root / 'config.toml'
        for experiment in ('sensitivity', 'routing'):
            for dataset in ('C4', 'CUSTOM'):
                for path in paths:
                    with self.subTest(experiment=experiment, dataset=dataset, path=path.name):
                        config.write_text(
                            '[paths]\ncache = "cache"\noutput = "results"\n'
                            f'[datasets]\nC4 = {json.dumps(path.as_posix())}\n',
                            encoding='utf-8')
                        options = (['--prompts-json', str(path)] if dataset == 'CUSTOM'
                                   else ['--datasets', dataset])
                        result = subprocess.run(
                            [sys.executable, '-B', '-c', code, '--experiment', experiment,
                             '--config', str(config), *options],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                            capture_output=True, text=True, timeout=30,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('FileNotFoundError' if path.name == 'missing.json'
                                      else 'ValueError', result.stderr)
                        self.assertIn(str(path), result.stderr)
                        self.assertIn(dataset, result.stderr)
                        self.assertNotIn('Heavy import before data validation', result.stderr)
                        self.assertFalse((self.root / 'results').exists())
                        self.assertFalse((self.root / 'cache').exists())


if __name__ == '__main__':
    unittest.main()
