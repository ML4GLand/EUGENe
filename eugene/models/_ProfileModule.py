from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from tqdm.auto import tqdm

from .base._losses import LOSS_REGISTRY
from ..evaluate.metrics._profile_prediction import calculate_performance_measures
from .base._optimizers import OPTIMIZER_REGISTRY
from .base._schedulers import SCHEDULER_REGISTRY


class ProfileModule(LightningModule):
    """
    A model that is a profile of a sequence.
    """

    def __init__(
        self,
        arch: torch.nn.Module,
        task: str = "profile",
        profile_loss_fxn: Optional[str] = "MNLLLoss",
        count_loss_fxn: Optional[str] = "log1pMSELoss",
        optimizer: Optional[str] = "adam",
        optimizer_lr: Optional[float] = 1e-3,
        optimizer_kwargs: Optional[dict] = {},
        scheduler: Optional[str] = None,
        scheduler_monitor: str = "count_loss",
        scheduler_kwargs: dict = {},
        metric: str = None,
        metric_kwargs: dict = None,
        seed: int = None,
        save_hyperparams: bool = True,
        arch_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        super().__init__()

        # Set the base attributes of the Profile model
        self.arch = arch
        self.input_len = arch.input_len
        self.output_dim = arch.output_dim

        # Set the task
        self.task = task
        self.arch_name = (
            arch_name if arch_name is not None else self.arch.__class__.__name__
        )

        # Set the loss functions
        self.profile_loss_fxn = LOSS_REGISTRY[profile_loss_fxn] if isinstance(profile_loss_fxn, str) else profile_loss_fxn
        self.count_loss_fxn = LOSS_REGISTRY[count_loss_fxn] if isinstance(count_loss_fxn, str) else count_loss_fxn

        # Set the optimizer
        self.optimizer = OPTIMIZER_REGISTRY[optimizer]
        self.optimizer_lr = optimizer_lr if optimizer_lr is not None else 1e-3
        self.optimizer_kwargs = optimizer_kwargs

        # Set the scheduler
        self.scheduler = (
            SCHEDULER_REGISTRY[scheduler] if scheduler is not None else None
        )
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_kwargs = scheduler_kwargs

        # Ignore the metric for now

        # Set the seed
        if seed is not None:
            self.seed = seed
            seed_everything(self.seed)
        else:
            self.seed = None

        # Set the hyperparameters if passed in
        if save_hyperparams:
            self.save_hyperparameters(ignore=["arch"])

        # Set the model name, if none given make up a fun one
        self.model_name = model_name if model_name is not None else "model"

    def forward(self, x, x_ctl) -> torch.Tensor:
        """
        Forward pass of the arch.

        Parameters:
        ----------
        x (torch.Tensor):
            input sequence
        """
        return self.arch(x, x_ctl)
    
    def predict(
        self,
        X: torch.Tensor,
        X_ctl: Optional[torch.Tensor] = None,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        with torch.no_grad():
            starts = np.arange(0, X.shape[0], batch_size)
            ends = starts + batch_size

            y_profiles, y_counts = [], []
            for start, end in zip(starts, ends):
                X_batch = X[start:end].to(device)
                X_ctl_batch = None if X_ctl is None else X_ctl[start:end].to(device)

                y_profiles_, y_counts_ = self(X_batch, X_ctl_batch)
                y_profiles_ = y_profiles_.cpu()
                y_counts_ = y_counts_.cpu()

                y_profiles.append(y_profiles_)
                y_counts.append(y_counts_)

            y_profiles = torch.cat(y_profiles)
            y_counts = torch.cat(y_counts)
            return y_profiles, y_counts


    def _common_step(self, batch, batch_idx, stage: str):
        X, X_ctl, y = batch
        y_profile, y_counts = self(X, X_ctl)
        y_profile = y_profile.reshape(y_profile.shape[0], -1)
        y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
        y = y.reshape(y.shape[0], -1)

        # Calculate the profile and count losses
        profile_loss = self.profile_loss_fxn(y_profile, y).mean()
        count_loss = self.count_loss_fxn(y_counts, y.sum(dim=-1).reshape(-1, 1)).mean()

        # Extract the profile loss for logging
        profile_loss_ = profile_loss.item()
        count_loss_ = count_loss.item()

        # Mix losses together and update the model
        loss = profile_loss + self.arch.alpha * count_loss

        return {
            "loss": loss,
            "profile_loss": profile_loss_,
            "count_loss": count_loss_,
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        step_dict = self._common_step(batch, batch_idx, "train")
        self.log("train_loss", step_dict["loss"], on_step=True, on_epoch=False)
        self.log("train_profile_loss", step_dict["profile_loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_count_loss", step_dict["count_loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss_epoch", step_dict["loss"], on_step=False, on_epoch=True)
        return step_dict

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        #step_dict = self._common_step(batch, batch_idx, "val")
        #self.log("val_loss_epoch", step_dict["loss"], on_step=False, on_epoch=True)
        #self.log("val_profile_loss_epoch", step_dict["profile_loss"], on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val_count_loss_epoch", step_dict["count_loss"], on_step=False, on_epoch=True, prog_bar=True)
        X, X_ctl, y = batch
        y_profile, y_counts = self(X, X_ctl)
        z = y_profile.shape
        y_profile = y_profile.reshape(y_profile.shape[0], -1)
        y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
        y_profile = y_profile.reshape(*z)
        measures = calculate_performance_measures(y_profile, 
            y, y_counts, kernel_sigma=7, 
            kernel_width=81, measures=['profile_mnll', 
            'profile_pearson', 'count_pearson', 'count_mse'])
        profile_mnll = measures['profile_mnll'].mean().item()
        count_mse  = measures['count_mse'].mean().item()
        profile_corr = measures['profile_pearson'].cpu()
        count_corr = measures['count_pearson'].cpu()
        valid_loss = measures['profile_mnll'].mean().item()
        valid_loss += self.arch.alpha * measures['count_mse'].mean().item()
        self.log("val_loss_epoch", valid_loss, on_step=False, on_epoch=True)
        self.log("val_profile_loss_epoch", profile_mnll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_count_loss_epoch", count_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_profile_corr_epoch", np.nan_to_num(profile_corr).mean().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_count_corr_epoch", np.nan_to_num(count_corr).mean().item(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step"""
        step_dict = self._common_step(batch, batch_idx, "test")
        self.log("test_loss", step_dict["loss"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizers

        Returns:
        ----------
        torch.optim.Optimizer:
            optimizer
        torch.optim.lr_scheduler._LRScheduler:
            learning rate scheduler
        """
        optimizer = self.optimizer(
            self.parameters(), lr=self.optimizer_lr, **self.optimizer_kwargs
        )
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.scheduler_monitor,
            }
