import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from .base import BaseModel, BasicFullyConnectedModule, BasicConv1D, BasicRecurrent
from ..datasets import random_ohe_seqs
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning import seed_everything


class GAN(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        generator: nn.Module,
        discriminator: nn.Module,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "cross_entropy",
        optimizer: str = "adam",
        gen_lr: float = 1e-3,
        disc_lr: float = 1e-3,
        scheduler: str = "lr_scheduler",
        scheduler_patience: int = 2,
        optimizer_kwargs: dict = {},
        seed: int = None,
        **kwargs
    ):

        super().__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.strand = strand
        self.task = task
        self.aggr = aggr
        self.loss_fxn = loss_fxn_dict[loss_fxn]
        self.optimizer = optimizer
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.scheduler = scheduler
        self.scheduler_patience = scheduler_patience
        self.optimizer_kwargs = optimizer_kwargs
        seed_everything(seed) if seed is not None else None
        self.kwargs = kwargs
        self.save_hyperparameters()

    def forward(self, x, x_rev_comp=None):
        x = x.flatten(start_dim=1)
        if self.strand == "ss":
            x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._common_step(batch, batch_idx, optimizer_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, None, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, None, "test")

    def _common_step(self, batch, batch_idx, optimizer_idx, stage : str):
        ID, x, x_rev_comp, y = batch

        rand = random_ohe_seqs(seq_len=x.size(2), batch_size=x.size(0), return_tensor=True, device=x.device, dtype=x.dtype)

        rand = rand.flatten(start_dim=1)
        x = x.flatten(start_dim=1)

        # Generator
        if optimizer_idx == 0 or optimizer_idx is None:
            val = torch.ones(x.size(0), 1, dtype=torch.long)
            val = val.type_as(x)

            loss = F.binary_cross_entropy(self.discriminator(rand), val)

            self.log(f"{stage}_generator_loss", loss, on_epoch=False, rank_zero_only=True)    

        # Discriminator
        elif optimizer_idx == 1:
            val = torch.ones(x.size(0), 1, dtype=torch.long)
            val = val.type_as(x)
            inv = torch.zeros(x.size(0), 1, dtype=torch.long)
            inv = inv.type_as(x)

            val_loss = F.binary_cross_entropy(self.discriminator(x), val)
            inv_loss = F.binary_cross_entropy(self.discriminator(rand), inv)
            loss = (val_loss + inv_loss) / 2

            self.log(f"{stage}_discriminator_loss", loss, on_epoch=False, rank_zero_only=True)

        return loss

    def configure_optimizers(self):
        gen_optimizer = optimizer_dict[self.optimizer](
            self.parameters(), lr=self.gen_lr, **self.optimizer_kwargs
        )
        disc_optimizer = optimizer_dict[self.optimizer](
            self.parameters(), lr=self.disc_lr, **self.optimizer_kwargs
        )

        gen_scheduler = (
            ReduceLROnPlateau(gen_optimizer, patience=self.scheduler_patience)
            if self.scheduler_patience is not None
            else None
        )
        disc_scheduler = (
            ReduceLROnPlateau(disc_optimizer, patience=self.scheduler_patience)
            # LambdaLR(disc_optimizer)
            if self.scheduler_patience is not None
            else None
        )

        return (
            {
                "optimizer": gen_optimizer,
                "lr_scheduler": {
                    "scheduler": gen_scheduler,
                    "monitor": "val_loss",
                },
            },
            {
                "optimizer": disc_optimizer,
                "lr_scheduler": {
                    "scheduler": disc_scheduler,
                    "monitor": "val_loss",
                },
            },
        )

    def summary(self):
        """Print a summary of the model"""
        print(f"Model: {self.__class__.__name__}")
        print(f"Input length: {self.input_len}")
        print(f"Strand: {self.strand}")
        print(f"Task: {self.task}")
        print(f"Aggregation: {self.aggr}")
        print(f"Loss function: {self.loss_fxn.__name__}")
        print(f"Optimizer: {self.optimizer}")
        print(f"\tOptimizer parameters: {self.optimizer_kwargs}")
        print(f"Learning rate: {self.lr}")
        print(f"Scheduler: {self.scheduler}")
        print(f"Scheduler patience: {self.scheduler_patience}")
        return ModelSummary(self)

loss_fxn_dict = {
    "mse": F.mse_loss,
    "poisson": F.poisson_nll_loss,
    "bce": F.binary_cross_entropy_with_logits,
    "cross_entropy": F.cross_entropy,
}

optimizer_dict = {"adam": Adam, "sgd": SGD}
