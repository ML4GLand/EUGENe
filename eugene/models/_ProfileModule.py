import torch
import numpy as np
from typing import Optional, Tuple
from pytorch_lightning.core import LightningModule
from base._losses import MNLLLoss, log1pMSELoss

class ProfileModel(LightningModule):
    """
    A model that is a profile of a sequence.
    """
    def __init__(
        self, 
        task: str = "profile",
        arch: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.task = task
        self.profile_loss = "MNLLLoss"
        self.count_loss = "log1pMSELoss"

        # Set the architecture
        self.arch = arch if arch is not None else self.__class__.__name__

        # Set the model name
        self.name = name if name is not None else "model"

    def training_step(self, batch, batch_idx):
        """Training step"""
        step_dict = self._common_step(batch, batch_idx, "train")
        self.log("train_loss", step_dict["loss"])
        return step_dict

    def predict(
        self, 
        X: torch.Tensor, 
        X_ctl: Optional[torch.Tensor] = None, 
        batch_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            starts = np.arange(0, X.shape[0], batch_size)
            ends = starts + batch_size

            y_profiles, y_counts = [], []
            for start, end in zip(starts, ends):
                X_batch = X[start:end].cuda()
                X_ctl_batch = None if X_ctl is None else X_ctl[start:end].cuda()

                y_profiles_, y_counts_ = self(X_batch, X_ctl_batch)
                y_profiles_ = y_profiles_.cpu()
                y_counts_ = y_counts_.cpu()
                
                y_profiles.append(y_profiles_)
                y_counts.append(y_counts_)

            y_profiles = torch.cat(y_profiles)
            y_counts = torch.cat(y_counts)
            return y_profiles, y_counts
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        step_dict = self._common_step(batch, batch_idx, "val")
        self.log("val_loss", step_dict["loss"], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step"""
        step_dict = self._common_step(batch, batch_idx, "test")
        self.log("test_loss", step_dict["loss"], on_step=False, on_epoch=True)

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
    
    def configure_optimizers(self):
        """Configure the optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer