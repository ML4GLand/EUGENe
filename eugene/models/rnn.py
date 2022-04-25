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
from claim.modules import BasicFullyConnectedModule, BasicRecurrent

# EUGENE
from eugene.dataloading.SeqDataModule import SeqDataModule

class RNN(LightningModule):
    def __init__(self, input_len, strand="ss", task="regression", aggr=None, rnn_kwargs={}, fc_kwargs={}):
        super().__init__()
        self.strand = strand
        self.task = task
        self.aggr = aggr
        
        # Add strand specific modules
        if self.strand == "ss":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.rnn.out_dim, **fc_kwargs)
        elif self.strand == "ds":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.rnn.out_dim*2, **fc_kwargs)
        elif self.strand == "ts":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.reverse_rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.rnn.out_dim*2, **fc_kwargs)
            
        # Add task specific metrics
        if self.task == "regression":
            self.r_squared = torchmetrics.R2Score()
        elif self.task == "binary_classification":
            self.accuracy = torchmetrics.Accuracy()
            self.auroc = torchmetrics.AUROC()
       
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, x_rev_comp):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        if self.strand == "ds":
            x_rev_comp, _ = self.rnn(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            x = torch.cat((x, x_rev_comp), dim=1)
        elif self.strand == "ts":
            x_rev_comp, _ = self.reverse_rnn(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            x = torch.cat((x, x_rev_comp), dim=1)
        x = self.fcnet(x)
        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")
        
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        pass
        
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
    cli = LightningCLI(RNN, SeqDataModule, save_config_overwrite=True)
