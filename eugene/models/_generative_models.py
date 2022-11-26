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


class GAN(BaseModel):
    def __init__(
        self,
        latent_dim: int,
        generator: nn.Module,
        discriminator: nn.Module,
        mode: str = "wgan",
        grad_clip: float = None,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str ="mse",
        gen_lr: float = 1e-3,
        disc_lr: float = 1e-3,
        **kwargs
    ):
        super().__init__(
            input_len=None,
            output_dim=None,
            strand=strand, 
            task=task, 
            aggr=aggr, 
            loss_fxn=loss_fxn,
            lr=gen_lr,
            save_hp=False,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.mode = mode
        self.grad_clip = grad_clip
        self.strand = strand
        self.task = task
        self.aggr = aggr
        self.loss_fxn = self.loss_fxn_dict[loss_fxn]
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.kwargs = kwargs

    def forward(self, x, x_rev_comp=None):
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

        rand = torch.randn(x.size(0), 4, self.latent_dim)
        rand = rand.type_as(x)

        # Generator
        if optimizer_idx == 0 or optimizer_idx is None:
            if self.mode == "gan":
                val = torch.ones(x.size(0), 1, dtype=torch.long)
                val = val.type_as(x)

                gen_seqs = self(rand).view(x.size(0), 4, -1)
                gen_loss = F.binary_cross_entropy(self.discriminator(gen_seqs), val)

            elif self.mode == "wgan":
                gen_loss = -torch.mean(self.discriminator(self(rand)))

            self.log(f"{stage}_generator_loss", gen_loss, on_step=True, rank_zero_only=True)    
            return gen_loss

        # Discriminator
        elif optimizer_idx == 1:
            if self.mode == "gan":
                val = torch.ones(x.size(0), 1, dtype=torch.long)
                val = val.type_as(x)
                inv = torch.zeros(x.size(0), 1, dtype=torch.long)
                inv = inv.type_as(x)

                val_loss = F.binary_cross_entropy(self.discriminator(x), val)
                gen_seqs = self(rand).detach().view(x.size(0), 4, -1)
                inv_loss = F.binary_cross_entropy(self.discriminator(gen_seqs), inv)
                disc_loss = (val_loss + inv_loss) / 2

            elif self.mode == "wgan":
                disc_loss = -torch.mean(self.discriminator(x)) + torch.mean(self.discriminator(self(rand)))

                if self.grad_clip is not None:
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.grad_clip, self.grad_clip)

            self.log(f"{stage}_discriminator_loss", disc_loss, on_step=True, rank_zero_only=True)
            return disc_loss

    def configure_optimizers(self):
        gen_optimizer = self.optimizer_dict[self.optimizer](
            self.parameters(), lr=self.gen_lr, **self.optimizer_kwargs
        )
        disc_optimizer = self.optimizer_dict[self.optimizer](
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
