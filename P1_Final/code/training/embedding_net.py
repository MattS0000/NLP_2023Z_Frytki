import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, base: nn.Module, tokenizer: nn.Module):
        super(EmbeddingNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.base = base.to(self.device)
        self.tokenizer = tokenizer

    def forward(self, x):
        output = self.tokenizer(x, return_tensors='pt').to(self.device)
        output = self.base(**output)
        return output['last_hidden_state'][0,-1,:]

    def get_embedding(self, x):
        return self.forward(x)