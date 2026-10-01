"""Checkpoint completeness checks run without importing the inference runtime."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from configuration import Configuration
from configuration.check import check_checkpoint


class ResourceCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkpoint = self.root / 'specter-assets' / 'models' / 'target'
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / 'config.json').write_text('{}', encoding='utf-8')
        self.config = Configuration(self.root / 'specter.local.toml', {
            'models': {'dsv2lite': {'target': './specter-assets/models/target'}}})

    def index(self, mapping):
        if mapping:
            mapping = {'model.embed_tokens.weight': next(iter(mapping.values())), **mapping}
        (self.checkpoint / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': mapping}), encoding='utf-8')

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            check_checkpoint(self.config, 'dsv2lite', 'target')
        return output.getvalue()

    def test_external_relative_layout_and_shared_shard_references(self):
        self.index({'layer.0.weight': 'part.safetensors',
                    'layer.1.weight': 'part.safetensors'})
        (self.checkpoint / 'part.safetensors').write_bytes(b'test fixture')
        self.assertIn('1 weight files found', self.check())

    def test_missing_shard_reports_model_role_and_resolved_path(self):
        self.index({'weight': 'missing.safetensors'})
        with self.assertRaises(FileNotFoundError) as error:
            self.check()
        self.assertIn('models.dsv2lite.target', str(error.exception))
        self.assertIn(str(self.checkpoint / 'missing.safetensors'), str(error.exception))

    def test_target_requires_index_even_when_unsharded_weights_exist(self):
        (self.checkpoint / 'model.safetensors').write_bytes(b'test fixture')
        with self.assertRaisesRegex(FileNotFoundError, 'missing required weight index'):
            self.check()

    def test_external_and_malformed_shard_references_are_rejected(self):
        (self.checkpoint.parent / 'outside.safetensors').write_bytes(b'test fixture')
        for name in ('../outside.safetensors', [1], None, ''):
            self.index({'weight': name})
            with self.subTest(name=name), self.assertRaises((ValueError, FileNotFoundError)):
                self.check()


if __name__ == '__main__':
    unittest.main()
