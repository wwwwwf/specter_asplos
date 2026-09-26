"""Resource configuration and checkpoint layout checks without model weights."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import configuration
from initialization import checkpoint_layout


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='specter-config-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.config_file = self.root / 'settings' / 'specter.toml'
        self.config_file.parent.mkdir()
        self.config_file.write_text(
            '[paths]\ncache = "../work/cache"\noutput = "../work/output"\n'
            '[models.qwen2moe]\ntarget = "../assets/Qwen1.5"\n'
            'draft = "$SPECTER_TEST_ASSETS/Qwen1.5-draft"\n',
            encoding='utf-8',
        )
        for module in (configuration, checkpoint_layout):
            root_patch = patch.object(module, 'ROOT', self.source)
            root_patch.start()
            self.addCleanup(root_patch.stop)
        environment_patch = patch.dict(os.environ, {
            'SPECTER_CONFIG': str(self.config_file),
            'SPECTER_TEST_ASSETS': str(self.root / 'assets'),
        })
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        configuration._read.cache_clear()
        self.addCleanup(configuration._read.cache_clear)
        self.config = configuration.get_config()

    def make_checkpoint(self, group, basename='Qwen1.5'):
        checkpoint = self.root / 'assets' / group / basename
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{}', encoding='utf-8')
        (checkpoint / 'model.safetensors').write_bytes(group.encode())
        (checkpoint / 'modeling_qwen2_moe_fused.py').write_text(
            'ORIGIN = "checkpoint"\n', encoding='utf-8')
        return checkpoint

    def make_local_code(self):
        code = self.source / 'models' / 'draft' / 'qwen2moe'
        code.mkdir(parents=True)
        for filename in ('configuration_qwen2_moe_fused.py', 'modeling_qwen2_moe_fused.py'):
            (code / filename).write_text('ORIGIN = "local"\n', encoding='utf-8')
        return code

    def test_paths_use_config_directory_and_expand_environment_then_home(self):
        self.assertEqual(self.config.cache_dir, self.root / 'work' / 'cache')
        self.assertEqual(self.config.path('models.qwen2moe', 'target'),
                         self.root / 'assets' / 'Qwen1.5')
        self.assertEqual(self.config.path('models.qwen2moe', 'draft'),
                         self.root / 'assets' / 'Qwen1.5-draft')
        with patch.dict(os.environ, {'SPECTER_TEST_ASSETS': '~/specter-test-assets'}):
            self.assertEqual(self.config.path('models.qwen2moe', 'draft'),
                             Path.home() / 'specter-test-assets' / 'Qwen1.5-draft')

    def test_outputs_are_external_and_relative_names_use_output_directory(self):
        self.assertEqual(self.config.output_path('case/run.json', 'unused.json'),
                         self.root / 'work' / 'output' / 'case' / 'run.json')
        with self.assertRaises(ValueError):
            self.config.output_path(str(self.source / 'result.json'), 'unused.json')
        self.config.output_dir.mkdir(parents=True)
        (self.config.output_dir / 'generation.json').symlink_to(self.source / 'result.json')
        with self.assertRaises(ValueError):
            self.config.output_path(None, 'generation.json')

    def test_configured_caches_replace_inherited_source_paths(self):
        variables = {
            'HF_HOME': 'huggingface',
            'HF_MODULES_CACHE': 'huggingface/modules',
            'TRITON_CACHE_DIR': 'triton',
            'TORCH_EXTENSIONS_DIR': 'torch_extensions',
        }
        with patch.dict(os.environ, {key: str(self.source / key) for key in variables}):
            configuration.configure(self.config_file)
            for key, child in variables.items():
                self.assertEqual(Path(os.environ[key]), self.config.cache_dir / child)

    def test_same_basename_checkpoints_have_distinct_local_code_overlays(self):
        code = self.make_local_code()
        first = self.make_checkpoint('first')
        second = self.make_checkpoint('second')
        overlays = [Path(checkpoint_layout.localize_checkpoint(path, 'qwen2moe'))
                    for path in (first, second)]
        self.assertNotEqual(*overlays)
        for overlay, checkpoint in zip(overlays, (first, second)):
            self.assertTrue(overlay.name.isidentifier())
            self.assertEqual((overlay / 'model.safetensors').resolve(), checkpoint / 'model.safetensors')
            self.assertEqual((overlay / 'modeling_qwen2_moe_fused.py').resolve(),
                             code / 'modeling_qwen2_moe_fused.py')
            self.assertIn('"local"', (overlay / 'modeling_qwen2_moe_fused.py').read_text())
            self.assertIn('"checkpoint"', (checkpoint / 'modeling_qwen2_moe_fused.py').read_text())
        self.assertEqual(Path(checkpoint_layout.localize_checkpoint(first, 'qwen2moe')), overlays[0])

    def test_checkpoint_alias_is_idempotent_and_safe_for_dotted_names(self):
        first = self.make_checkpoint('first', 'Phi-3.5')
        second = self.make_checkpoint('second', 'Phi-3.5')
        alias = Path(checkpoint_layout.checkpoint_alias(first))
        self.assertTrue(alias.is_symlink())
        self.assertTrue(alias.name.isidentifier())
        self.assertEqual(alias.resolve(), first)
        self.assertEqual(Path(checkpoint_layout.checkpoint_alias(first)), alias)
        self.assertEqual(Path(checkpoint_layout.checkpoint_alias(alias)), alias)
        self.assertNotEqual(Path(checkpoint_layout.checkpoint_alias(second)), alias)

    def test_missing_local_code_fails_before_overlay_creation(self):
        checkpoint = self.make_checkpoint('first')
        with self.assertRaises(FileNotFoundError):
            checkpoint_layout.localize_checkpoint(checkpoint, 'qwen2moe')
        self.assertFalse((self.config.cache_dir / 'checkpoints').exists())
        code = self.make_local_code()
        (code / 'modeling_qwen2_moe_fused.py').unlink()
        with self.assertRaises(FileNotFoundError):
            checkpoint_layout.localize_checkpoint(checkpoint, 'qwen2moe')
        self.assertFalse((self.config.cache_dir / 'checkpoints').exists())

    def test_cache_child_symlinks_cannot_write_into_source(self):
        checkpoint = self.make_checkpoint('first')
        self.make_local_code()
        self.config.cache_dir.mkdir(parents=True)
        for child, operation in (
            ('checkpoint_aliases', lambda: checkpoint_layout.checkpoint_alias(checkpoint)),
            ('checkpoints', lambda: checkpoint_layout.localize_checkpoint(checkpoint, 'qwen2moe')),
        ):
            destination = self.source / child
            destination.mkdir()
            (self.config.cache_dir / child).symlink_to(destination, target_is_directory=True)
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(list(destination.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
