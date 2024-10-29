import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.space_albert import SpaceALBERT
from dataloader.space_dataset import SpaceDataset

class Tester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpaceALBERT().to(self.device)
        self.load_checkpoint("checkpoints/space_7.pkl")
        
        hf_test_set = load_dataset('json', data_files='data/train_serialized.json')['train']
        self.dataset = SpaceDataset(hf_test_set)
        self.dataloader = DataLoader(self.dataset, batch_size = 16, shuffle = True)

    def test(self, threshold = 0.5):
        self.model.eval()
        correct_predictions = 0
        total_valid_predictions = 0

        with torch.no_grad():
            for batch in self.dataloader:
                no_of_tokens, spellings, kinds, labels = batch

                labels = torch.tensor(labels, dtype=torch.float).to(self.device)

                outputs = self.model(spellings, kinds)
                outputs = outputs[:, :50]

                mask = (labels != 100).float()
                masked_outputs = outputs * mask
                masked_labels = labels * mask

                predictions = (masked_outputs > threshold).float()
                masked_predictions = predictions * mask

                correct_predictions += ((masked_predictions == masked_labels) * mask).sum().item()
                total_valid_predictions += mask.sum().item() 

        accuracy = correct_predictions / total_valid_predictions * 100
        print(f'Test Accuracy: {accuracy:.2f}%')


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(msg)

if __name__ == "__main__":
    tester = Tester()
    tester.test()