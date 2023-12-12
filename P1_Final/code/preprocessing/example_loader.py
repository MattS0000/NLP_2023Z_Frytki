from torch.utils.data import DataLoader
from textdataset import TripletsDataset
import numpy as np
dataset = TripletsDataset(".")
dataloader = DataLoader(dataset,batch_size=5)

for i in dataloader:
    print(np.array(i).transpose())
    print(np.array(i).transpose().shape)
    break