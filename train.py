import torch
from torch.nn import functional as F

from opt import get_opts

#datasets
from torch.utils.data import DataLoader
from dataset import ImageDataset

#models
from models import MLP

#optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(42, workers = True)


class CoordinateMLPSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if hparams == 'identity':
            self.net = MLP()
    
    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        hparams = self.hparams
        self.train_dataset = ImageDataset(hparams.image_path, 'train')
        self.val_dataset = ImageDataset(hparams.image_path, 'val')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)

        return self.optimizer
    
    def training_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])

    
        
        
