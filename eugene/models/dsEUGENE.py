import argparse
import numpy as np

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
import torchmetrics

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.cli import LightningCLI

# CLAIM
from claim.modules import BasicConv1D, BasicRecurrent, BasicFullyConnectedModule
from claim.utils import init_weights

# EUGENE
from eugene.dataloading.SeqDataModule import SeqDataModule
from eugene.utils.seq_transforms import ReverseComplement, Augment, OneHotEncode, ToTensor


class dsEUGENE(LightningModule):
    def __init__(self, conv_kwargs={}, rnn_kwargs={}, fc_kwargs={}, learning_rate=1e-3):
        super().__init__()
        self.convnet = BasicConv1D(**conv_kwargs)
        self.recurrentnet = BasicRecurrent(input_dim=self.convnet.out_channels*2, **rnn_kwargs)
        self.fcnet = BasicFullyConnectedModule(input_dim=self.recurrentnet.out_dim, **fc_kwargs)

        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC()
        
        self.save_hyperparameters()
        init_weights(self)
        
    def forward(self, x, x_rev_comp):
        batch_size, channels, seq_len = x.size()
        x = self.convnet(x)
        x_rev_comp = self.convnet(x_rev_comp)
        x, x_rev_comp = x.transpose(1, 2), x_rev_comp.transpose(1, 2)
        x_cat = torch.cat([x, x_rev_comp], dim=2)
        out, _ = self.recurrentnet(x_cat)
        out = self.fcnet(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")
        
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        ID, x, x_rev_comp, y = batch
        outs = self(x, x_rev_comp).squeeze(dim=1)
        return np.stack((ID.squeeze(dim=1).detach().cpu().numpy(), outs.detach().cpu().numpy(), y.squeeze(dim=1).detach().cpu().numpy()), axis=-1)
        
    def _common_step(self, batch, batch_idx, stage: str):
        # Get and log loss
        _, x, x_rev_comp, y = batch
        outs = self(x, x_rev_comp).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(outs, y)
        self.log(f"{stage}_loss", loss, on_epoch=True)
        
        # Get and log the accuracy
        preds = torch.round(torch.sigmoid(outs))
        acc = self.accuracy(preds, y.long())
        self.log(f"{stage}_acc", acc, on_epoch=True)
        
        # Get and log the auroc
        probs = torch.sigmoid(outs)
        auroc = self.auroc(probs, y.long())
        self.log(f"{stage}_auroc", auroc, on_epoch=True)
        
        if stage == "val":
            self.log("hp_metric", auroc, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    
if __name__ == "__main__":
    cli = LightningCLI(dsEUGENE, SeqDataModule, save_config_overwrite=True)