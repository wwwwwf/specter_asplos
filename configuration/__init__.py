"""Shared resource paths, resolved relative to an external TOML configuration."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]

def external_path(path):
    """Keep generated files and large resources out of the source tree."""
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError(f'Choose a path outside the source directory: {path}')
    return path

@dataclass(frozen=True)
class Configuration:
    file: Path
    values: dict

    def path(self, section, key):
        value = self.values
        for part in section.split('.'):
            value = value[part]
        raw = value[key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f'Missing path: [{section}] {key}')
        path = Path(os.path.expandvars(raw)).expanduser()
        if not path.is_absolute():
            path = self.file.parent / path
        return external_path(path)

    @property
    def cache_dir(self):
        return self.path('paths', 'cache')

    @property
    def output_dir(self):
        return self.path('paths', 'output')

    def output_path(self, override, default):
        if override is None:
            return external_path(self.output_dir / default)
        path = Path(os.path.expandvars(str(override))).expanduser()
        return external_path(path if path.is_absolute() else self.output_dir / path)

@lru_cache(maxsize=8)
def _read(path):
    file = Path(path).expanduser().resolve()
    with file.open('rb') as stream:
        return Configuration(file, tomllib.load(stream))

def get_config():
    return _read(os.environ.get('SPECTER_CONFIG', str(ROOT / 'config.example.toml')))

def configure(path=None):
    if path is not None:
        os.environ['SPECTER_CONFIG'] = str(Path(path).expanduser().resolve())
    config = get_config()
    cache = config.cache_dir
    for key, value in {
        'HF_HOME': cache / 'huggingface',
        'HF_MODULES_CACHE': cache / 'huggingface' / 'modules',
        'TRITON_CACHE_DIR': cache / 'triton',
        'TORCH_EXTENSIONS_DIR': cache / 'torch_extensions',
    }.items():
        os.environ[key] = str(external_path(value))
    return config


def configure_from_cli():
    """Set cache locations before importing Transformers or Torch."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config')
    args, _ = parser.parse_known_args()
    return configure(args.config)
