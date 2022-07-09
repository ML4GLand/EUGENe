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
from ..preprocessing._encoding import ascii_decode

# omit_final_pool should be set to True in conv_kwargs
class DeepBind(LightningModule):
    def __init__(self, input_len, strand="ss", task="regression", aggr=None, mp_kwargs = {}, conv_kwargs = {}, fc_kwargs = {}):
        super().__init__()
        self.flattened_input_dims = 8*input_len
        self.strand = strand
        self.task = task
        self.aggr = aggr

        self.mp_kwargs, self.conv_kwargs, self.fc_kwargs = self.kwarg_handler(mp_kwargs, conv_kwargs, fc_kwargs)

        self.max_pool = nn.MaxPool1d(**self.mp_kwargs)
        self.avg_pool = nn.AvgPool1d(**self.mp_kwargs)

        # Add strand specific modules
        if self.strand == "ss":
            self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.fcn = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim//(mp_kwargs.get("kernel_size")//2), **self.fc_kwargs)
        elif self.strand == "ds":
            self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.fcn = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim//(mp_kwargs.get("kernel_size")//4), **self.fc_kwargs)
        elif self.strand == "ts":
            self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.fcn = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim//(mp_kwargs.get("kernel_size")//2), **self.fc_kwargs)
            self.reverse_convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.reverse_fcn = BasicFullyConnectedModule(input_dim=self.reverse_convnet.flatten_dim//(mp_kwargs.get("kernel_size")//2), **self.fc_kwargs)

        # Add task specific metrics
        if self.task == "regression":
            self.r_squared = torchmetrics.R2Score()
        elif self.task == "binary_classification":
            self.accuracy = torchmetrics.Accuracy()
            self.auroc = torchmetrics.AUROC()

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, x_rev_comp = None):
        x = self.convnet(x)
        x = x.view(x.size(0), self.convnet.flatten_dim)
        x = torch.cat((self.max_pool(x), self.avg_pool(x)), dim=1)

        if self.strand == "ss":
            x = self.fcn(x)
        elif self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
            x = torch.cat((x, x_rev_comp), dim=1)

            x = self.fcn(x)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)

            x = self.fcn(x)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = self.reverse_fcn(x_rev_comp)
            x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x

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

    # Sets default kwargs if not specified
    def kwarg_handler(self, mp_kwargs, conv_kwargs, fc_kwargs):
        mp_kwargs.setdefault("kernel_size", 4)
        # Add mp_kwargs for stride

        conv_kwargs.setdefault("channels", [4, 16])
        conv_kwargs.setdefault("conv_kernels", [4])
        conv_kwargs.setdefault("pool_kernels", [4])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("dropout_rates", 0.2)
        conv_kwargs.setdefault("batchnorm", False)
        # Add conv_kwargs for stride

        fc_kwargs.setdefault("output_dim", 1)
        fc_kwargs.setdefault("hidden_dims", [256, 64, 16, 4])
        fc_kwargs.setdefault("dropout_rate", 0.2)
        fc_kwargs.setdefault("batchnorm", False)

        return mp_kwargs, conv_kwargs, fc_kwargs

if __name__ == "__main__":
    cli = LightningCLI(DeepBind, SeqDataModule, save_config_overwrite=True)
