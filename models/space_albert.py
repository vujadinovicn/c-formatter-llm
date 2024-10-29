from torch import nn
import torch
from transformers import AlbertTokenizer, AlbertModel
# import torch_directml
# dml = torch_directml.device()

class SpaceALBERT(nn.Module):
    def __init__(self, pretrained_model_name='albert-base-v2'):
        super(SpaceALBERT, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AlbertModel.from_pretrained(pretrained_model_name).to(self.device)
        self.tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(768, 1)

    def forward(self, spellings, kinds):
        inputs = self.tokenizer(spellings, kinds, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        token_logits = self.fc(last_hidden_state).squeeze(-1)
        space_preds = torch.sigmoid(token_logits)

        return space_preds