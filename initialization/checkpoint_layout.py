"""Pair local model definitions with external checkpoint assets."""
import hashlib
from pathlib import Path
from configuration import get_config, external_path

ROOT = Path(__file__).resolve().parents[1]

def checkpoint_alias(path):
    """Give dynamic model imports an identifier-safe external directory name."""
    path = external_path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    alias = get_config().cache_dir / 'checkpoint_aliases' / ('checkpoint_' + digest)
    external_path(alias.parent)
    external_path(alias)
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        if alias.resolve() != path:
            raise ValueError(f'Checkpoint alias points to another directory: {alias}')
    elif alias.exists():
        raise FileExistsError(alias)
    else:
        alias.symlink_to(path, target_is_directory=True)
    return str(alias)

def localize_checkpoint(path, model_name):
    path = external_path(path)
    if model_name not in {'dsv2lite', 'qwen2moe', 'phimoe'}:
        raise ValueError(f'Unsupported model: {model_name}')
    code = ROOT / 'models' / 'draft' / model_name
    if not path.is_dir():
        raise FileNotFoundError(path)
    required = {
        'dsv2lite': ('configuration_deepseek.py', 'modeling_deepseek.py'),
        'qwen2moe': ('configuration_qwen2_moe_fused.py', 'modeling_qwen2_moe_fused.py'),
        'phimoe': ('configuration_phimoe.py', 'modeling_phimoe.py', 'modeling_fusedphimoe.py'),
    }[model_name]
    for filename in required:
        if not (code / filename).is_file():
            raise FileNotFoundError(code / filename)
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    overlay = get_config().cache_dir / 'checkpoints' / (model_name + '_' + digest)
    external_path(overlay)
    overlay.mkdir(parents=True, exist_ok=True)
    sources = {p.name: p for p in path.iterdir()
               if not p.name.startswith('.') and p.name != '__pycache__' and p.suffix != '.py'}
    sources.update({p.name: p for p in code.glob('*.py')})
    for name, source in sources.items():
        link = overlay / name
        if link.is_symlink():
            if link.resolve() == source.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(f'Refusing to overwrite checkpoint overlay: {link}')
        link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    for link in overlay.iterdir():
        if link.name not in sources and link.is_symlink():
            link.unlink()
    return str(overlay)
