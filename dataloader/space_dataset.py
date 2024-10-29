import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

class SpaceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.max_label_length = 50

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        no_of_tokens = len(self.dataset[idx]['token_spellings'])
        spellings = " ".join(self.dataset[idx]['token_spellings'])
        kinds = " ".join(self.dataset[idx]['token_kinds'])
        labels = self.dataset[idx]['labels']

        if len(labels) < self.max_label_length:
            labels = labels + [100] * (self.max_label_length - len(labels))  # padding with 100 for as ignoring index
        else:
            labels = labels[:self.max_label_length]

        labels = torch.tensor(labels, dtype=torch.float)
        return no_of_tokens, spellings, kinds, labels