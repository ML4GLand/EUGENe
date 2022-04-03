import argparse

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.cli import LightningCLI
from claim.modules import BasicConv1D, BasicRecurrent, BasicFullyConnectedModule
import torchmetrics
import optuna

# MPRADataModule
from MPRADataModule import MPRADataModule
from torchvision import transforms
from transforms import ReverseComplement, Augment, OneHotEncode, ToTensor


class dsEUGENE(LightningModule):
    def __init__(self, conv_kwargs={}, rnn_kwargs={}, fc_kwargs={}, learning_rate=1e-3):
        super().__init__()
        self.convnet = BasicConv1D(**conv_kwargs)
        self.recurrentnet = BasicRecurrent(input_dim=self.convnet.out_channels*2, **rnn_kwargs)
        self.fcnet = BasicFullyConnectedModule(input_dim=self.recurrentnet.out_dim, **fc_kwargs)

        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC()
        
        self.save_hyperparameters()
        
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

    def _common_step(self, batch, batch_idx, stage: str):
        # Get and log loss
        x, x_rev_comp, y = batch["sequence"], batch["reverse_complement"], batch["target"]
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
        self.log(f"{stage}_auroc", acc, on_epoch=True)
        
        if stage == "val":
            self.log("hp_metric", auroc, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
        

# Hyperoptimization objective function
def objective(trial: optuna.trial.Trial, datamodule, tb_name="test", max_epochs=5):
    fcn_layers = trial.suggest_int("fcn_n_layers", 1, 3)
    fcn_dropout = trial.suggest_float("fcn_dropout", 0.2, 0.5)
    fcn_output_dims = [
        trial.suggest_int("fcn_n_units_l{}".format(i), 4, 128, log=True) for i in range(fcn_layers)
    ]
    cnn=dict(input_len=66, channels=[4, 16], conv_kernels=[15, 5], pool_kernels=[1, 1])
    rnn=dict(output_dim=32, batch_first=True)
    fc=dict(output_dim=1, hidden_dims=fcn_output_dims, dropout_rate=fcn_dropout)
    eugene = dsEUGENE(conv_kwargs=cnn, rnn_kwargs=rnn, fc_kwargs=fc)
    init_weights(eugene)
    logger = TensorBoardLogger(tb_name, name="dsEUGENE", version="trial_{}".format(trial.number))
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
    )
    hyperparameters = dict(fcn_layers=fcn_layers, fcn_dropout=fcn_dropout, fcn_output_dims=fcn_output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(eugene, datamodule=datamodule)
    return trainer.callback_metrics["val_auroc_epoch"].item()
   
    
if __name__ == "__main__":
    cli = LightningCLI(dsEUGENE, MPRADataModule)