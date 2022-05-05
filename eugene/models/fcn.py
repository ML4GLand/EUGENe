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

# CLAIM
from claim.modules import BasicFullyConnectedModule

# EUGENE
from eugene.dataloading.SeqDataModule import SeqDataModule
from eugene.utils.seq_utils import ascii_decode
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from eugene.utils.custom_callbacks import PredictionWriter

class FCN(LightningModule):
    def __init__(self, input_len, strand="ss", task="regression", aggr=None, fc_kwargs={}):
        super().__init__()
        self.flattened_input_dims = 4*input_len
        self.strand = strand
        self.task = task
        self.aggr = aggr
        
        # Add strand specific modules
        if self.strand == "ss":
            self.fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims, **fc_kwargs)
        elif self.strand == "ds":
            self.fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims*2, **fc_kwargs)
        elif self.strand == "ts":
            self.fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims, **fc_kwargs)
            self.reverse_fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims, **fc_kwargs)   
            
        # Add task specific metrics
        if self.task == "regression":
            self.r_squared = torchmetrics.R2Score()
        elif self.task == "binary_classification":
            self.accuracy = torchmetrics.Accuracy()
            self.auroc = torchmetrics.AUROC()
       
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, x_rev_comp):
        x = x.flatten(start_dim=1)
        if self.strand == "ss":
            x = self.fcn(x)
        elif self.strand == "ds":
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x = torch.cat((x, x_rev_comp), dim=1)
            x = self.fcn(x)
        elif self.strand == "ts":
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

if __name__ == "__main__":
    cli = LightningCLI(FCN, SeqDataModule, save_config_overwrite=True)
