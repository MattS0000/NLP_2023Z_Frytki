from torch.utils.data import Dataset
import numpy as np
# Class loading preprocessed data needed to train the model
class PletsDataset(Dataset):
    def __init__(self, file_paths: str, plets: str, direct: bool = False):
        self.file_paths = file_paths
        self.triplets=np.load(file_paths+f"/{plets}.npy")
        self.direct = direct

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        result = self.triplets[index].tolist()
        if self.direct:
            return result
        return result, []
