import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from claim.modules import BasicConv1D, BasicRecurrent, BasicFullyConnectedModule


class ssEUGENE(LightningModule):
    def __init__(self, conv_kwargs={}, rnn_kwargs={}, fc_kwargs={}):
        super().__init__()
        self.convnet = BasicConv1D(**conv_kwargs)
        self.recurrentnet = BasicRecurrent(input_dim=self.convnet.out_channels, **rnn_kwargs)
        self.fcnet = BasicFullyConnectedModule(input_dim=self.recurrentnet.out_dim, **fc_kwargs)

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        x = self.convnet(x)
        x = x.transpose(1, 2)
        x, _ = self.recurrentnet(x)
        x = self.fcnet(x[:, -1, :])
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x).squeeze(dim=1)
        train_loss = F.binary_cross_entropy_with_logits(outs, y)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x).squeeze(dim=1)
        val_loss = F.binary_cross_entropy_with_logits(outs, y)
        self.log("val_loss", val_loss)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x).squeeze(dim=1)
        test_loss = F.binary_cross_entropy_with_logits(outs, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)