from datasets import load_dataset
from utils import get_tokens_and_spacing, save_dataset_to_json, initialize_clang_library

def process_example(example, index):
    try:
        tu = index.parse('example.c', unsaved_files=[('example.c', example["content"])])
        token_spellings, token_kinds, spaces = get_tokens_and_spacing(tu.cursor, example["content"])
        return {"token_spellings": token_spellings, "token_kinds": token_kinds, "labels": spaces}
    except Exception as e:
        print(f"Error processing example: {e}")
        return {"token_spellings": [], "token_kinds": [], "labels": []}

def drop_type_key(example):
    return {"content": example["content"]} 

def serialize_dataset():
    index = initialize_clang_library()

    for file in ['data/train.json, data/val.json, data/test.json']:
        ds = load_dataset('json', data_files=file)['train']
        content_only_ds = ds.map(drop_type_key)
        processed_dataset = content_only_ds.map(lambda example: process_example(example, index))
        save_dataset_to_json(processed_dataset, f"data/{file.split('/')[1].split('.')[0]}_serialized.json")

if __name__ == "__main__":
    serialize_dataset()