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
from collections import OrderedDict


class GAN(BaseModel):
    def __init__(
        self,
        seq_len: int,
        latent_dim: int,
        generator: nn.Module,
        discriminator: nn.Module,
        mode: str = "wgan",
        grad_clip: float = None,
        gen_lr: float = 1e-3,
        disc_lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        n_critic: int = 5,
        **kwargs
    ):
        super().__init__(
            input_len=None,
            output_dim=None,
            lr=gen_lr,
            save_hp=False,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.mode = mode
        self.grad_clip = grad_clip
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.b1 = b1
        self.b2 = b2
        self.n_critic = n_critic
        self.kwargs = kwargs

    def forward(self, x, x_rev_comp=None):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._common_step(batch, batch_idx, optimizer_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, None, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, None, "test")

    def _common_step(self, batch, batch_idx, optimizer_idx, stage : str):
        
        # Get a batch
        _, x, _, _ = batch

        # Sample noise as generator input
        z = torch.normal(0, 1, (x.size(0), self.latent_dim))
        z.requires_grad_(True)
        z = z.type_as(x)

        # Generator step
        if optimizer_idx == 0:

            # Fake seqs and labels (all 1s)
            valid = torch.ones(x.size(0), 1, dtype=torch.long)
            valid = valid.type_as(x)

            # Loss measures generator's ability to fool the discriminator
            if self.mode == "gan":
                gen_loss = F.binary_cross_entropy(self.discriminator(self(z)), valid)
            elif self.mode == "wgan":
                gen_loss = -torch.mean(self.discriminator(self(z)))
            self.log(f"{stage}_generator_loss", gen_loss, on_step=True, rank_zero_only=True)    
            
            # To return
            tqdm_dict = {'g_loss': gen_loss.detach()}
            output = OrderedDict({
                'loss': gen_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            return output

        # Discriminator step
        elif optimizer_idx == 1:
            if self.mode == "gan":

                # Real seqs and labels (all 1s)
                valid = torch.ones(x.size(0), 1, dtype=torch.long)
                valid = valid.type_as(x)
                real_loss = F.binary_cross_entropy(self.discriminator(x), valid)

                # Fake seqs and labels (all 0s)
                fake = torch.zeros(x.size(0), 1, dtype=torch.long)
                fake = fake.type_as(x)
                fake_loss = F.binary_cross_entropy(self.discriminator(self(z)), fake)
                
                # Total discriminator loss is average
                disc_loss = (real_loss + fake_loss) / 2

            elif self.mode == "wgan":
                disc_loss = -torch.mean(self.discriminator(x)) + torch.mean(self.discriminator(self(z)))

            if self.grad_clip is not None:
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.grad_clip, self.grad_clip)

            self.log(f"{stage}_discriminator_loss", disc_loss, on_step=True, rank_zero_only=True)

            tqdm_dict = {'d_loss': disc_loss}
            output = OrderedDict({
                'loss': disc_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        gen_optimizer = self.optimizer_dict[self.optimizer](
            self.generator.parameters(), 
            lr=self.gen_lr, 
            betas=(self.b1, self.b2),
            **self.optimizer_kwargs
        )
        disc_optimizer = self.optimizer_dict[self.optimizer](
            self.discriminator.parameters(), 
            lr=self.disc_lr, 
            betas=(self.b1, self.b2),
            **self.optimizer_kwargs
        )

        return (
            {'optimizer': gen_optimizer, 'frequency': 1},
            {'optimizer': disc_optimizer, 'frequency': self.n_critic}
        )
