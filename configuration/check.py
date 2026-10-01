"""Validate experiment resources before loading a model or allocating GPU memory."""
import argparse
import json

from configuration import configure


def read_metadata(path, label):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'{label}: cannot read {path}: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{label}: expected a JSON object in {path}')
    return value


def check_checkpoint(config, model, role):
    label = f'models.{model}.{role}'
    directory = config.path(f'models.{model}', role)
    if not directory.is_dir():
        raise FileNotFoundError(f'{label}: checkpoint directory not found: {directory}')
    metadata = directory / 'config.json'
    if not metadata.is_file():
        raise FileNotFoundError(f'{label}: missing model configuration: {metadata}')
    read_metadata(metadata, label)
    index = directory / 'model.safetensors.index.json'
    if role == 'target' and not index.is_file():
        raise FileNotFoundError(f'{label}: missing required weight index: {index}')
    if index.is_file():
        mapping = read_metadata(index, label).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f'{label}: missing weight_map in {index}')
        if role == 'target' and 'model.embed_tokens.weight' not in mapping:
            raise ValueError(f'{label}: missing model.embed_tokens.weight in {index}')
        if any(not isinstance(name, str) or not name for name in mapping.values()):
            raise ValueError(f'{label}: invalid shard name in {index}')
        shards = sorted(set(mapping.values()))
    else:
        shards = ['model.safetensors']
    for name in shards:
        shard = (directory / name).resolve()
        if not shard.is_relative_to(directory.resolve()) or not shard.is_file():
            raise FileNotFoundError(f'{label}: checkpoint shard unavailable: {shard}')
    print(f'[checkpoint] {label}: {len(shards)} weight files found in {directory}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='External TOML configuration (or SPECTER_CONFIG).')
    parser.add_argument('--model', choices=['dsv2lite', 'qwen2moe', 'phimoe'], default='dsv2lite')
    parser.add_argument('--datasets', nargs='+', choices=['GK', 'WT', 'HE', 'GP', 'C4'],
                        default=['GK', 'WT', 'HE', 'GP', 'C4'])
    parser.add_argument('--num-data', type=int, default=5)
    args = parser.parse_args(argv)
    if args.num_data < 1:
        parser.error('--num-data must be positive')
    try:
        config = configure(args.config)
        for role in ('target', 'draft'):
            check_checkpoint(config, args.model, role)
        from data.loader import prepare_datasets
        prompts = prepare_datasets(config, args.datasets, args.num_data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for name, texts in prompts.items():
        print(f'[dataset] {name}: {len(texts)} nonempty prompts', flush=True)
    print('Resource check passed; no model weights were loaded.', flush=True)


if __name__ == '__main__':
    main()
