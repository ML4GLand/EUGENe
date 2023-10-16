import torch
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Callable
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning import seed_everything
from ._losses import MNLLLoss, log1pMSELoss

class ProfileModel(LightningModule, ABC):
    """
    A model that is a profile of a sequence.
    """
    @abstractmethod
    def __init__(
        self, 
        input_len: int,
        output_dim: int,
        task: str = "profile",
        **kwargs
    ):
        super().__init__()
        self.input_len = input_len
        self.output_dim = output_dim
        self.task = task
        self.profile_loss = "MNLLLoss"
        self.count_loss = "log1pMSELoss"

    @abstractmethod
    def forward(self, X, X_ctl=None) -> torch.Tensor:
        """
        Forward pass of the model. This method must be implemented by the child class.

        Parameters:
        ----------
        x (torch.Tensor):
            input sequence
        x_rev_comp (torch.Tensor):
            reverse complement of the input sequence
        """
    
    def predict(self, x, X, X_ctl=None, batch_size=32):
        """
        Predict the output of the model in batches
        """
        with torch.no_grad():
            device = self.device
            self.eval()
            outs = []
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device)
                X_ctl_batch = X_ctl[i:i+batch_size].to(device) if X_ctl is not None else None
                outs.append(self(X_batch, X_ctl_batch))
            return torch.cat(outs, dim=0)

    def _common_step(self, batch, batch_idx, stage: str):
        X, X_ctl, y = batch
        y_profile, y_counts = self(X, X_ctl)
        y_profile = y_profile.reshape(y_profile.shape[0], -1)
        y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
        
        y = y.reshape(y.shape[0], -1)

        # Calculate the profile and count losses
        profile_loss = MNLLLoss(y_profile, y).mean()
        count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 1)).mean()

        # Extract the profile loss for logging
        profile_loss_ = profile_loss.item()
        count_loss_ = count_loss.item()

        # Mix losses together and update the model
        loss = profile_loss + self.alpha * count_loss

        return {
            "loss": loss,
            "profile_loss": profile_loss_,
            "count_loss": count_loss_,
        }

    def training_step(self, batch, batch_idx):
        """Training step"""
        step_dict = self._common_step(batch, batch_idx, "train")
        self.log("train_loss", step_dict["loss"])
        return step_dict

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        step_dict = self._common_step(batch, batch_idx, "val")
        self.log("val_loss", step_dict["loss"], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step"""
        step_dict = self._common_step(batch, batch_idx, "test")
        self.log("test_loss", step_dict["loss"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure the optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer