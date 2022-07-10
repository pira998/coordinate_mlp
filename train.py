from einops import rearrange
import torch
from torch.nn import functional as F
from zmq import device
from losses import MSELoss
from metrics import PSNR

from opt import get_opts

#datasets
from torch.utils.data import DataLoader
from dataset import ImageDataset

#models
from models import MLP, PE

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
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams

        if hparams.arch == 'identity':
            self.net = MLP()
        
        elif hparams.arch == 'pe':
            P = torch.cat([torch.eye(2) * 2 ** i for i in range(10)], dim= 1) # 10x2x2
            self.pe = PE(P)
            self.net = MLP(n_input=self.pe.out_dim)
        
        elif hparams.arch == 'ff': # fourier features
            ## hparams.sc should be 2 * np.pi * hparams.sc
            P = hparams.sc * torch.normal(torch.zeros(2,256), torch.ones(2,256))
            self.pe = PE(P)
            self.net = MLP(n_input=self.pe.out_dim)

        self.loss = MSELoss()
    
    def forward(self, x):
        if hparams.arch == 'identity':
            return self.net(x)
        else:
            return self.net(self.pe(x))

    def setup(self, stage=None):
        hparams = self.hparams_
        self.train_dataset = ImageDataset(hparams.image_path, 'train')
        self.val_dataset = ImageDataset(hparams.image_path, 'val')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams_.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams_.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams_.lr)

        return self.optimizer
    
    def training_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])
        loss = self.loss(rgb_pred, batch['rgb'])
        psnr_ = PSNR(rgb_pred, batch['rgb'])

        self.log('train_loss', loss)
        self.log('train_psnr', psnr_, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])
        loss = self.loss(rgb_pred, batch['rgb'])
        psnr_ = PSNR(rgb_pred, batch['rgb'])

        log = {'val_loss': loss, 'val_psnr': psnr_, 'rgb_pred': rgb_pred, 'rgb': batch['rgb']}
        return log

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        rgb_pred = torch.cat([x['rgb_pred'] for x in outputs]) 
        rgb_pred = rearrange(rgb_pred, '(h w) c -> c h w',
                             h =  self.train_dataset.r,
                                w =  self.train_dataset.r) # (3, H, W)
        self.logger.experiment.add_image('val/rgb_pred', rgb_pred, self.global_step) # (3, H, W)


        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_psnr', avg_psnr, prog_bar=True)
        return {'val_loss': avg_loss, 'val_psnr': avg_psnr}


if __name__ == '__main__':
    hparams = get_opts()
    coordMLPsystem = CoordinateMLPSystem(hparams)

    ckpt_cb = ModelCheckpoint(
        save_top_k=-1,
        dirpath=f'ckpts/{hparams.exp_name}',
        filename='{epoch}-{val_psnr:.4f}',
    )

    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(
        save_dir=f'logs/{hparams.exp_name}',
        name=hparams.exp_name,
        default_hp_metric=False
    )

    trainer = Trainer(
        max_epochs=hparams.epochs,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=True,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=0,
        benchmark=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=20,
    )

    trainer.fit(coordMLPsystem)





    
        
        
