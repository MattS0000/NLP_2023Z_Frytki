from torch.utils.data import Dataset
import numpy as np
# Class loading preprocessed data needed to train the model
class PletsDataset(Dataset):
    def __init__(self, file_paths:str, plets:str="triplets"):
        self.file_paths = file_paths
        self.triplets=np.load(file_paths+f"/{plets}.npy")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index].tolist()
