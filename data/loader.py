"""Read prepared prompts and validate them before model initialization."""
import csv
import json
from pathlib import Path


def _read_huggingface(path):
    # JSON/CSV validation and missing-path errors do not need this dependency.
    from datasets import load_from_disk

    print(f"Loading Hugging Face dataset from directory: {path}")
    dataset = load_from_disk(str(path))
    available_splits = list(dataset.keys())
    split = next((name for name in ('test', 'validation', 'train')
                  if name in available_splits), None)
    if split is None:
        raise ValueError(f"Dataset has no supported split: {available_splits}")
    data_split = dataset[split]
    print(f"Using split: {split}, total samples: {len(data_split)}")

    # Keep the existing dataset-name and column priorities.
    directory_name = path.name.lower()
    fields = (('humaneval', 'prompt'), ('gsm8k', 'question'),
              ('gpqa', 'Question'), ('wikitext', 'text'))
    text_key = next((key for name, key in fields if name in directory_name), None)
    if text_key is None:
        sample = data_split[0] if len(data_split) else {}
        text_key = next((key for key in ('text', 'question', 'prompt', 'input', 'content', 'sentence')
                         if key in sample), None)
        if text_key is None:
            raise ValueError(f"Cannot infer the text field. Available fields: {list(sample)}")
    return [item[text_key] for item in data_split
            if isinstance(item[text_key], str) and item[text_key].strip()]


def _read_json(path):
    with path.open(encoding='utf-8') as stream:
        prompts = json.load(stream)
    if not isinstance(prompts, list):
        raise ValueError('JSON data must be a list')
    if not prompts:
        return []
    if isinstance(prompts[0], str):
        return prompts
    if isinstance(prompts[0], dict):
        candidates = ('text', 'question', 'prompt', 'Pre-Revision Question')
        text_key = next((key for key in candidates if key in prompts[0]), None)
        if text_key is None:
            raise ValueError(f'No recognized JSON field: {candidates}')
        if not all(isinstance(item, dict) for item in prompts):
            raise ValueError('JSON prompt records must all be objects')
        return [item[text_key] for item in prompts if item.get(text_key)]
    raise ValueError('JSON data must contain strings or prompt objects')


def _read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        candidates = ('Pre-Revision Question', 'text', 'question', 'prompt', 'Question', 'input')
        text_key = next((key for key in candidates if key in (reader.fieldnames or [])), None)
        if text_key is None:
            raise ValueError(f'No recognized CSV column: {candidates}')
        return [row[text_key] for row in reader
                if isinstance(row.get(text_key), str) and row[text_key].strip()]


def prepare_data(dataset_file, num_data, dataset_name=None, *, allow_fewer=False):
    """Read ordered, nonempty prompts, requiring the requested count by default.

    Candidate-pool callers may set ``allow_fewer`` to treat ``num_data`` as a
    maximum; an empty pool is still invalid. Final sample selection must then
    enforce the experiment's actual count and token-length requirements.
    """
    path = Path(dataset_file).expanduser().resolve()
    context = f"Dataset {dataset_name or path.name!r} at {path}"
    if not path.exists():
        raise FileNotFoundError(f'{context}: file or directory does not exist')
    if not isinstance(num_data, int) or num_data < 1:
        raise ValueError(f'{context}: requested prompt count must be positive')
    try:
        if path.is_dir():
            prompts = _read_huggingface(path)
        elif path.suffix.lower() == '.json':
            prompts = _read_json(path)
        elif path.suffix.lower() == '.csv':
            prompts = _read_csv(path)
        else:
            raise ValueError('Expected JSON, CSV, or a saved Hugging Face dataset directory; '
                             f'unsupported file extension {path.suffix!r}')
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError('Prompt values must be strings')
        selected = [prompt for prompt in prompts if prompt.strip()][:num_data]
        if allow_fewer and not selected:
            raise ValueError('Need at least 1 nonempty prompt; found 0')
        if not allow_fewer and len(selected) != num_data:
            raise ValueError(f'Need {num_data} nonempty prompts; found {len(selected)}')
    except FileNotFoundError as error:
        raise FileNotFoundError(f'{context}: {error}') from error
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError(f'{context}: {error}') from error
    print(f"[data] {dataset_name or path.name}: {len(selected)} prompts from {path}", flush=True)
    return selected


def prepare_datasets(config, dataset_names, num_data):
    """Validate and retain selected text for reuse after model loading."""
    return {name: prepare_data(config.path('datasets', name), num_data, dataset_name=name)
            for name in dataset_names}


def init_datasets():
    """Prepare the WikiText input in the configured external data directory."""
    from datasets import load_dataset
    from configuration import configure_from_cli
    config = configure_from_cli()
    destination = config.path('datasets', 'WT')
    if destination.exists():
        raise FileExistsError(destination)
    print('Loading WikiText...')
    dataset = load_dataset('wikitext', 'wikitext-2-v1',
                           cache_dir=str(config.cache_dir / 'datasets'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving dataset to: {destination}')
    dataset.save_to_disk(str(destination))
    print('Dataset saved.')


if __name__ == '__main__':
    init_datasets()
