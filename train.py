import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.space_albert import SpaceALBERT
from dataloader.space_dataset import SpaceDataset

class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpaceALBERT().to(self.device)
        # self.load_checkpoint(self.config.get("checkpoint_path"))

        self.load_datasets()
        self.configure_optimizer()
        self.criterion = nn.BCELoss()

        self.max_epochs = 1
        self.best_val_loss = 1e6

    def load_datasets(self):
        hf_train_set = load_dataset('json', data_files='data/train_serialized.json')['train']
        self.train_dataset = SpaceDataset(hf_train_set)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = 16, shuffle = True)

        hf_val_set = load_dataset('json', data_files='data/val_serialized.json')['train']
        self.val_dataset = SpaceDataset(hf_val_set)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size = 16, shuffle = False)

    def fit(self):
        for epoch in range(0, self.max_epochs):
            print(f"Train epoch no {epoch}")
            self.train()
            print(f"Valid epoch no {epoch}")
            self.evaluate(epoch)

    def train(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            no_of_tokens, spellings, kinds, labels = batch

            self.optimizer.zero_grad()
            labels = torch.tensor(labels, dtype=torch.float).to(self.device)

            outputs = self.model(spellings, kinds)
            outputs = outputs[:, :50]

            mask = (labels != 100).float()
            masked_outputs = outputs * mask
            masked_labels = labels * mask

            loss = self.criterion(masked_outputs, masked_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        print(f'Training loss: {total_loss/len(self.train_dataloader)}')


    def evaluate(self, epoch, threshold = 0.5):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_valid_predictions = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                no_of_tokens, spellings, kinds, labels = batch

                labels = torch.tensor(labels, dtype=torch.float).to(self.device)

                outputs = self.model(spellings, kinds)
                outputs = outputs[:, :50]

                mask = (labels != 100).float()
                masked_outputs = outputs * mask
                masked_labels = labels * mask

                loss = self.criterion(masked_outputs, masked_labels)
                total_loss += loss.item()

                predictions = (masked_outputs > threshold).float()
                masked_predictions = predictions * mask

                correct_predictions += ((masked_predictions == masked_labels) * mask).sum().item()
                total_valid_predictions += mask.sum().item() 

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct_predictions / total_valid_predictions * 100
        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch)

    def configure_optimizer(self):
        learning_rate = 1e-5
        self.optimizer = Adam(params=self.model.parameters(), lr=learning_rate)

    def save_checkpoint(self, epoch: int, checkpoint_name='checkpoint'):
        model_state_dict = self.model.state_dict()
        torch.save(dict(model_state_dict=model_state_dict, epoch=epoch), f"space_{epoch}.pkl")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(msg)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.fit()