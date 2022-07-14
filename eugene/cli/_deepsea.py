# Classics
import numpy as np

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import torchmetrics

# PyTorch Lightning
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI

# EUGENE
from .base import BasicFullyConnectedModule, BasicConv1D
from ..dataloading.dataloaders import SeqDataModule
from ..preprocessing._seq_preprocess import ascii_decode

# omit_
from .base._utils import GetFlattenDim

class DeepSEA(LightningModule):
    # Need to modify this to have as many blocks as I want
    def __init__(self, task="regression", input_len=1000, channels=[320, 480, 960], conv_kernels=8, pool_kernels=4, dropout_rates=[0.2, 0.2, 0.5]):
        """
        Generates a PyTorch module with architecture matching the convnet part of DeepSea. Default parameters are those specified in the DeepSea paper
        Parameters
        ----------
        input_len : int, input sequence length
        channels : list-like or int, channel width for each conv layer. If int each of the three layers will be the same channel width
        conv_kernels : list-like or int, conv kernel size for each conv layer. If int will be the same for all conv layers
        pool_kernels : list-like or int, maxpooling kernel size for the first two conv layers. If int will be the same for all conv layers
        dropout_rates : list-like or float, dropout rates for each conv layer. If int will be the same for all conv layers
        """
        super(DeepSEA, self).__init__()

        # If only one conv channel size provided, apply it to all convolutional layers
        if type(channels) == int:
            channels = list(np.full(3, channels))
        else:
            assert len(channels) == 3

        # If only one conv kernel size provided, apply it to all convolutional layers
        if type(conv_kernels) == int:
            conv_kernels = list(np.full_like(channels, conv_kernels))
        else:
            assert len(conv_kernels) == len(channels)

        # If only one dropout rate provided, apply it to all convolutional layers
        if type(dropout_rates) == float:
            dropout_rates = list(np.full_like(channels, dropout_rates))
        else:
            assert len(dropout_rates) == len(channels)

        # If only one maxpool kernel size provided, apply it to the first two conv layers
        if type(pool_kernels) == int:
            pool_kernels = list(np.full(3, pool_kernels))
        else:
            assert len(pool_kernels) == len(channels)-1

        # Build the architecture as a sequential model
        self.module = nn.Sequential(
            nn.Conv1d(4, channels[0], kernel_size=conv_kernels[0]),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=pool_kernels[0], stride=pool_kernels[0]),
            nn.Dropout(p=dropout_rates[0]),
            nn.Conv1d(channels[0], channels[1], kernel_size=conv_kernels[1]),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=pool_kernels[1], stride=pool_kernels[1]),
            nn.Dropout(p=dropout_rates[0]),
            nn.Conv1d(channels[1], channels[2], kernel_size=conv_kernels[2]),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rates[0]))

        # Build the fully connected layer
        self.flatten_dim = GetFlattenDim(self.module, seq_len=input_len)*channels[-1]
        self.fcnet = BasicFullyConnectedModule(input_dim=self.flatten_dim, output_dim=1)

        self.task = task

        # Add task specific metrics
        if self.task == "regression":
            self.r_squared = torchmetrics.R2Score()
        elif self.task == "binary_classification":
            self.accuracy = torchmetrics.Accuracy()
            self.auroc = torchmetrics.AUROC()

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        x = self.module(x)
        x = x.view(x.size(0), self.flatten_dim)
        return self.fcnet(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        ID, x, x_rev_comp, y = batch
        ID = np.array([ascii_decode(item) for item in ID.squeeze(dim=1).detach().cpu().numpy()])
        y = y.detach().cpu().numpy()
        outs = self(x, x_rev_comp).squeeze(dim=1).detach().cpu().numpy()
        return np.stack([ID, outs, y], axis=-1)

    def _common_step(self, batch, batch_idx, stage: str):
        # Get and log loss
        _, x, x_rev_comp, y = batch
        outs = self(x, x_rev_comp).squeeze(dim=1)

        if self.task == "binary_classification":
            # Get the loss
            loss = F.binary_cross_entropy_with_logits(outs, y)

            # Get and log the accuracy
            preds = torch.round(torch.sigmoid(outs))
            acc = self.accuracy(preds, y.long())
            self.log(f"{stage}_acc", acc, on_epoch=True)

            # Get and log the auroc
            probs = torch.sigmoid(outs)
            auroc = self.auroc(probs, y.long())
            self.log(f"{stage}_auroc", auroc, on_epoch=True)

        elif self.task == "regression":
            # Get the loss
            loss = F.mse_loss(outs, y)

            # Get and log R2Score
            r_squared = self.r_squared(outs, y)
            self.log(f"{stage}_R2", r_squared, on_epoch=True)

        self.log(f"{stage}_loss", loss, on_epoch=True)

        if stage == "val":
            if self.task == "binary_classification":
                self.log("hp_metric", auroc, on_epoch=True)
            elif self.task == "regression":
                self.log("hp_metric", r_squared, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
