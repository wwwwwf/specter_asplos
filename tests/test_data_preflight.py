"""Prepared data must fail before model/GPU initialization or result creation."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from data.loader import prepare_data, prepare_datasets


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='specter-data-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.prompts = [f' prompt {index} ' for index in range(7)]

    def read(self, path, name='TEST', count=5):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_data(path, count, dataset_name=name)

    def write_json(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding='utf-8')
        return path

    def test_json_strings_and_objects_preserve_first_nonempty_text_and_order(self):
        strings = self.write_json('strings.json', ['', *self.prompts])
        objects = self.write_json('objects.json', [
            {'text': '', 'question': 'Do not select this column'},
            *({'text': prompt, 'question': 'Do not select this column'}
              for prompt in self.prompts),
        ])
        self.assertEqual(self.read(strings), self.prompts[:5])
        self.assertEqual(self.read(objects), self.prompts[:5])

    def test_csv_preserves_gpqa_column_priority_and_first_nonempty_order(self):
        path = self.root / 'gpqa.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['Question', 'Pre-Revision Question'])
            writer.writeheader()
            for prompt in ['', *self.prompts]:
                writer.writerow({'Question': 'Do not select this column',
                                 'Pre-Revision Question': prompt})
        self.assertEqual(self.read(path, 'GP'), self.prompts[:5])

    def test_saved_dataset_preserves_split_and_dataset_column_priorities(self):
        path = self.root / 'gsm8k_prepared'
        path.mkdir()
        saved = {
            'train': [{'question': 'Wrong split'}],
            'validation': [{'question': 'Wrong split'}],
            'test': [{'question': prompt, 'text': 'Wrong column'}
                     for prompt in ['', *self.prompts]],
        }
        module = SimpleNamespace(load_from_disk=lambda _: saved)
        with patch.dict(sys.modules, {'datasets': module}):
            self.assertEqual(self.read(path, 'GK'), self.prompts[:5])
            del saved['test']
            saved['validation'] = [{'question': prompt} for prompt in self.prompts]
            self.assertEqual(self.read(path, 'GK'), self.prompts[:5])
            del saved['validation']
            saved['train'] = [{'question': prompt} for prompt in self.prompts]
            self.assertEqual(self.read(path, 'GK'), self.prompts[:5])

    @unittest.skipUnless(importlib.util.find_spec('datasets') is not None,
                         'requires Hugging Face datasets for the on-disk integration')
    def test_real_datasetdict_json_and_csv_select_identical_prompts(self):
        from datasets import Dataset, DatasetDict
        saved = self.root / 'gsm8k_saved'
        DatasetDict({
            'train': Dataset.from_dict({'question': ['Wrong split']}),
            'test': Dataset.from_dict({'question': ['', *self.prompts]}),
        }).save_to_disk(str(saved))
        json_path = self.write_json('reference.json', self.prompts)
        csv_path = self.root / 'reference.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['question'])
            writer.writerows([prompt] for prompt in ['', *self.prompts])
        expected = self.prompts[:5]
        for path in (saved, json_path, csv_path):
            with self.subTest(path=path):
                self.assertEqual(self.read(path), expected)

    def test_missing_and_insufficient_data_report_label_and_path(self):
        missing = self.root / 'missing_gsm8k'
        with self.assertRaises(FileNotFoundError) as caught:
            self.read(missing, 'GK')
        self.assertIn('GK', str(caught.exception))
        self.assertIn(str(missing), str(caught.exception))
        short = self.write_json('short.json', ['one', 'two', '   '])
        with self.assertRaises(ValueError) as caught:
            self.read(short, 'C4')
        self.assertIn('C4', str(caught.exception))
        self.assertIn(str(short), str(caught.exception))
        self.assertIn('Need 5 nonempty prompts; found 2', str(caught.exception))

    def test_candidate_mode_caps_count_without_weakening_default_or_validation(self):
        path = self.write_json('candidates.json', ['', *self.prompts, '   '])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_data(path, 1024, allow_fewer=True), self.prompts)
            self.assertEqual(prepare_data(path, 3, allow_fewer=True), self.prompts[:3])
        with self.assertRaisesRegex(ValueError, 'Need 1024 nonempty prompts; found 7'):
            self.read(path, count=1024)
        for name, payload in (('empty', []), ('blank', ['', '   ']),
                              ('invalid', ['one', 42])):
            invalid = self.write_json(f'{name}.json', payload)
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                prepare_data(invalid, 1024, dataset_name='C4', allow_fewer=True)
            self.assertIn('C4', str(caught.exception))
            self.assertIn(str(invalid), str(caught.exception))

    def test_malformed_json_and_nonstring_prompts_keep_resource_context(self):
        malformed = self.root / 'malformed.json'
        malformed.write_text('{', encoding='utf-8')
        invalid = self.write_json('invalid.json', ['first', 42, *self.prompts])
        for path in (malformed, invalid):
            with self.subTest(path=path), self.assertRaises(ValueError) as caught:
                self.read(path, 'C4')
            self.assertIn('C4', str(caught.exception))
            self.assertIn(str(path), str(caught.exception))

    def test_prepare_datasets_validates_in_requested_order(self):
        files = {name: self.write_json(f'{name}.json', [f'{name} {i}' for i in range(5)])
                 for name in ('GK', 'C4')}
        config = SimpleNamespace(path=lambda section, name: files[name])
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = prepare_datasets(config, ['C4', 'GK'], 5)
        self.assertEqual(list(prepared), ['C4', 'GK'])
        self.assertEqual(prepared['GK'], [f'GK {i}' for i in range(5)])

    @unittest.skipUnless(sys.version_info >= (3, 11), 'the runtime requires Python 3.11')
    def test_main_rejects_bad_data_before_heavy_imports_and_output_creation(self):
        # A subprocess catches even an attempted Torch/model import, regardless
        # of imports made by other tests in the current pytest process.
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
from benchmarks.run_tpot import main
main()
"""
        missing = self.root / 'missing_gsm8k'
        short = self.write_json('short.json', ['one'])
        for name, path, error_type in (
            ('missing', missing, 'FileNotFoundError'),
            ('short', short, 'ValueError'),
        ):
            config = self.root / f'{name}.toml'
            config.write_text(
                '[paths]\ncache = "cache"\noutput = "results"\n'
                f'[datasets]\nGK = {json.dumps(path.as_posix())}\n',
                encoding='utf-8')
            result = subprocess.run(
                [sys.executable, '-B', '-c', code, '--config', str(config),
                 '--datasets', 'GK', '--num-data', '5'],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error_type, result.stderr)
            self.assertIn(str(path), result.stderr)
            self.assertIn('GK', result.stderr)
            self.assertNotIn('Heavy import before data validation', result.stderr)
            self.assertFalse((self.root / 'results').exists())
            self.assertFalse((self.root / 'cache').exists())


if __name__ == '__main__':
    unittest.main()
