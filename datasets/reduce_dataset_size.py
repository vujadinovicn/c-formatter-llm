from utils import save_dataset_to_json, get_number_of_tokens, initialize_clang_library
from datasets import load_dataset

def filter_functions(example):
    return example['type'] == 'functions'

def filter_length(example, max_length):
    return len(example['content']) <= max_length

def reduce_dataset():
    ds = load_dataset("aircrypto/GitHub-C-Code-Segmented")['train']
    filtered_functions_ds = ds.filter(filter_functions)
    filtered_functions_ds = filtered_functions_ds.filter(lambda example: filter_length(example, 100))

    index = initialize_clang_library()
    filtered_length_ds = filtered_functions_ds.filter(lambda example: get_number_of_tokens(index.parse('example.c', unsaved_files=[('example.c', example['content'])]).cursor) <= 50)
    save_dataset_to_json(filtered_length_ds, 'data/reduced_dataset.json')

if __name__ == "__main__":
    reduce_dataset()
