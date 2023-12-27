from .base import LanguageModelBase
from preprocessing.dataset_loader import PletsDataset
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from .embedding_net import EmbeddingNet
from .training_func import fit
from .triplet_net import TripletNet
from evaluation.triplet_loss import TripletLoss


if torch.cuda.is_available():   
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('Using GPU', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class Trainer(LanguageModelBase):
    def __init__(self, base: LanguageModelBase):
        super().__init__()
        self.base = base.train()

    def train(self, train_data: PletsDataset, epochs: int = 20) -> 'Trainer':
        self._tokenizer = self.base.tokenizer
        self._model = self.base.model

        cuda = torch.cuda.is_available()
        embedding = EmbeddingNet(self.base.model, self.base.tokenizer)
        model = TripletNet(embedding)
        if cuda:
            model.cuda()
        loss_fn = TripletLoss(1)
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        log_interval = 100
        fit(train_data, train_data, model, loss_fn, optimizer, scheduler, epochs, cuda, log_interval)

        return self