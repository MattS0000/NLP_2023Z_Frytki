from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
# Class loading preprocessed data needed to train the model
class TokenizedPletsDataset(Dataset):
    def __init__(self, base: Dataset, tokenizer: nn.Module):
        self.base = base
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        result = [self.tokenizer(e, return_tensors='pt') for e in self.base[index]]
        max_len = 256
        for e in result:
            for k in e:
                e[k] = nn.functional.pad(e[k], (0, max_len - e[k].shape[1]), value=0)
        return result, []
