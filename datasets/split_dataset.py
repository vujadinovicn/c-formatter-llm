from datasets import load_dataset

def split_and_save_dataset(train_size, val_size):
    """Split the dataset into training, validation, and test sets and save them."""
    ds = load_dataset('json', data_files='data/reduced_dataset.json')['train']

    train_test_split = ds.train_test_split(test_size=1 - train_size, seed=42)
    train_val_split = train_test_split['test'].train_test_split(test_size=val_size, seed=42)

    train_test_split['train'].to_json('data/train.json')
    train_val_split['train'].to_json('data/val.json')
    train_val_split['test'].to_json('data/test.json')
    
if __name__ == "__main__":
    split_and_save_dataset(train_size=0.7, val_size=0.5)
