from datasets import load_from_disk
import json
import csv
import os

def prepare_data(dataset_file, num_data):
    try:
        input_texts = []

        if os.path.isdir(dataset_file):
            # Load a saved Hugging Face dataset.
            print(f"Loading Hugging Face dataset from directory: {dataset_file}")
            dataset = load_from_disk(dataset_file)

            # Select a split, preferring test, validation, then train.
            available_splits = list(dataset.keys())
            split = None
            for candidate in ['test', 'validation', 'train']:
                if candidate in available_splits:
                    split = candidate
                    break
            if split is None:
                raise ValueError(f"Dataset has no supported split: {available_splits}")

            data_split = dataset[split]
            print(f"Using split: {split}, total samples: {len(data_split)}")

            # Infer the text field from the dataset directory name.
            dir_name = os.path.basename(dataset_file).lower()
            text_key = None

            # Known text fields by dataset name.
            if 'humaneval' in dir_name:
                text_key = 'prompt'
            elif 'gsm8k' in dir_name:
                text_key = 'question'
            elif 'gpqa' in dir_name:
                text_key = 'Question'  # The field name uses an uppercase Q.
            elif 'wikitext' in dir_name:
                text_key = 'text'
            else:
                # Detect a text field for other datasets.
                sample = data_split[0] if len(data_split) > 0 else {}
                key_candidates = ['text', 'question', 'prompt', 'input', 'content', 'sentence']
                for key in key_candidates:
                    if key in sample:
                        text_key = key
                        break

            if text_key is None:
                sample_keys = list(data_split[0].keys()) if len(data_split) > 0 else []
                raise ValueError(f"Cannot infer the text field. Available fields: {sample_keys}")

            # Select nonempty text values.
            input_texts = [
                item[text_key] for item in data_split
                if isinstance(item[text_key], str) and len(item[text_key].strip()) > 0
            ][:num_data]
            print(f"Selected field '{text_key}', nonempty samples: {len(input_texts)}")

        else:
            # Load JSON or CSV input.
            _, ext = os.path.splitext(dataset_file)
            if ext.lower() == '.json':
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    all_prompts = json.load(f)
                if isinstance(all_prompts, list):
                    if len(all_prompts) == 0:
                        raise ValueError("JSON file is empty")
                    if isinstance(all_prompts[0], str):
                        input_texts = all_prompts
                    elif isinstance(all_prompts[0], dict):
                        key_candidates = ['text', 'question', 'prompt', 'Pre-Revision Question']
                        found_key = None
                        for key in key_candidates:
                            if key in all_prompts[0]:
                                found_key = key
                                break
                        if found_key:
                            input_texts = [item[found_key] for item in all_prompts if item.get(found_key)]
                        else:
                            raise ValueError(f"No recognized JSON field: {key_candidates}")
                else:
                    raise ValueError("JSON data must be a list")

            elif ext.lower() == '.csv':
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if 'Pre-Revision Question' in reader.fieldnames:
                        input_texts = [row['Pre-Revision Question'] for row in reader if row['Pre-Revision Question'].strip()]
                    else:
                        key_candidates = ['text', 'question', 'prompt', 'Question', 'input']
                        found_key = None
                        for key in key_candidates:
                            if key in reader.fieldnames:
                                found_key = key
                                break
                        if found_key:
                            input_texts = [row[found_key] for row in reader if row[found_key].strip()]
                        else:
                            raise ValueError(f"No recognized CSV column: {key_candidates}")

            else:
                raise ValueError(f"Unsupported file format: {ext}. Expected JSON, CSV or a saved Hugging Face dataset directory")

        # Limit the number of inputs.
        input_texts = input_texts[:num_data]
        print(f"\nUsing {len(input_texts)} prefixes...")

    except FileNotFoundError:
        print(f"Dataset file or directory not found: {dataset_file}")
        exit(1)
    except Exception as e:
        print(f"Failed to read dataset: {e}")
        exit(1)

    return input_texts


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
