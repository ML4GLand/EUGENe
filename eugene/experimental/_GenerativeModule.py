import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from ..models.base import (
    BaseModel,
    BasicFullyConnectedModule,
    BasicConv1D,
    BasicRecurrent,
)
from ..datasets import random_ohe_seqs
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning import seed_everything
from collections import OrderedDict
from torchmetrics import Accuracy


class GAN(LightningModule):
    def __init__(
        self,
        # seq_len: int,
        latent_dim: int,
        generator: nn.Module,
        discriminator: nn.Module,
        mode: str = "wgan",
        grad_clip: float = 0.01,
        lambda_gp: int = 10,
        gen_lr: float = 1e-3,
        disc_lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.9,
        n_critic: int = 5,
        **kwargs,
    ):
        super().__init__(
            input_len=None, output_dim=None, lr=gen_lr, save_hp=False, **kwargs
        )
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.mode = mode
        self.grad_clip = grad_clip
        self.lambda_gp = lambda_gp
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.b1 = b1
        self.b2 = b2
        self.n_critic = n_critic
        self.kwargs = kwargs

    def forward(self, x, x_rev_comp=None):
        x = self.generator(x)
        return x

    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     return self._common_step(batch, batch_idx, optimizer_idx, "train")

    # def validation_step(self, batch, batch_idx):
    #     return self._common_step(batch, batch_idx, None, "val")

    # def test_step(self, batch, batch_idx):
    #     return self._common_step(batch, batch_idx, None, "test")

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: str):
        # Get a batch
        _, x, _, _ = batch

        # Sample noise as generator input
        z = torch.normal(0, 1, (x.size(0), self.latent_dim))
        z.requires_grad_(True)
        z = z.type_as(x)

        # Generator step
        if optimizer_idx == 0:
            # real = self.discriminator(x)
            fake = self.discriminator(self(z))

            # Fake seqs and labels (all 1s)
            valid = torch.ones(x.size(0), 1, dtype=torch.long)
            valid = valid.type_as(x)

            # Loss measures generator's ability to fool the discriminator
            if self.mode == "gan":
                gen_loss = F.binary_cross_entropy(fake, valid)
            elif self.mode == "wgan" or self.mode == "wgangp":
                gen_loss = -torch.mean(fake)
            self.log(
                f"{stage}_generator_loss", gen_loss, on_step=True, rank_zero_only=True
            )

            # To return
            # tqdm_dict = {'g_loss': gen_loss.detach()}
            # output = OrderedDict({
            #     'loss': gen_loss,
            #     'progress_bar': tqdm_dict,
            #     'log': tqdm_dict
            # })

            return gen_loss

        # Discriminator step
        elif optimizer_idx == 1 or None:
            gen_seqs = self(z)
            real = self.discriminator(x)
            fake = self.discriminator(gen_seqs)

            real_labels = torch.ones(x.size(0), 1, dtype=torch.long).type_as(x)
            fake_labels = torch.zeros(x.size(0), 1, dtype=torch.long).type_as(x)

            if self.mode == "gan":
                # Real seqs and labels (all 1s)
                real_loss = F.binary_cross_entropy(real, real_labels)

                # Fake seqs and labels (all 0s)
                fake_loss = F.binary_cross_entropy(fake, fake_labels)

                # Total discriminator loss is average
                disc_loss = (real_loss + fake_loss) / 2

            elif self.mode == "wgan":
                disc_loss = -torch.mean(real) + torch.mean(fake)

                if self.grad_clip is not None:
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.grad_clip, self.grad_clip)

            elif self.mode == "wgangp":
                gradient_penalty = self.compute_gradient_penalty(x, gen_seqs.data)

                disc_loss = (
                    -torch.mean(real)
                    + torch.mean(fake)
                    + self.lambda_gp * gradient_penalty
                )

            self.log(
                f"{stage}_discriminator_loss",
                disc_loss,
                on_step=True,
                rank_zero_only=True,
            )

            # preds = torch.round(torch.sigmoid(torch.cat((real, fake))))
            # labels = torch.cat((real_labels, fake_labels))
            # accuracy = Accuracy(task="binary")
            # metric = accuracy(preds, labels)
            # self.log(f"{stage}_metric", metric, on_step=True, rank_zero_only=True)

            tqdm_dict = {"d_loss": disc_loss}
            output = OrderedDict(
                {"loss": disc_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def configure_optimizers(self):
        gen_optimizer = self.optimizer_dict[self.optimizer](
            self.generator.parameters(),
            lr=self.gen_lr,
            betas=(self.b1, self.b2),
            **self.optimizer_kwargs,
        )
        disc_optimizer = self.optimizer_dict[self.optimizer](
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=(self.b1, self.b2),
            **self.optimizer_kwargs,
        )

        return (
            {"optimizer": gen_optimizer, "frequency": 1},
            {"optimizer": disc_optimizer, "frequency": self.n_critic},
        )

    def compute_gradient_penalty(self, x, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        real_samples = x.data
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.shape))).type_as(x)
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        interpolates = interpolates.type_as(x)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).type_as(x)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).type_as(x)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
