"""
Model (implemented in Pytorch Lightning) demonstrating how to use augmentations
during training. Modified evoaug
"""

import torch
from pytorch_lightning import LightningModule
import numpy as np


class RobustModel(LightningModule):
    """PyTorch Lightning module to specify how augmentation should be applied to a model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    criterion : callable
        PyTorch loss function
    optimizer : torch.optim.Optimizer or dict
        PyTorch optimizer as a class or dictionary
    augment_list : list
        List of data augmentations, each a callable class from augment.py.
        Default is empty list -- no augmentations.
    max_augs_per_seq : int
        Maximum number of augmentations to apply to each sequence. Value is superceded by the number of augmentations in augment_list.
    hard_aug : bool
        Flag to set a hard number of augmentations, otherwise the number of augmentations is set randomly up to max_augs_per_seq, default is True.
    finetune : bool
        Flag to turn off augmentations during training, default is False.
    inference_aug : bool
        Flag to turn on augmentations during inference, default is False.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        augment_list=[],
        max_augs_per_seq=0,
        hard_aug=True,
        finetune=False,
        inference_aug=False,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.augment_list = augment_list
        self.max_augs_per_seq = np.minimum(max_augs_per_seq, len(augment_list))
        self.hard_aug = hard_aug
        self.inference_aug = inference_aug
        self.optimizer = optimizer
        self.max_num_aug = len(augment_list)
        self.insert_max = augment_max_len(augment_list)
        self.finetune = finetune

    def forward(self, x):
        """Standard forward pass."""
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        """Standard optimizer configuration."""
        return self.optimizer

    def training_step(self, batch, batch_idx):
        """Training step with augmentations."""
        x, y = batch
        if self.finetune:  # if finetune, no augmentations
            if (
                self.insert_max
            ):  # if insert_max is larger than 0, then pad each sequence with random DNA
                x = self._pad_end(x)
        else:
            x = self._apply_augment(x)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step without (or with) augmentations."""
        x, y = batch
        if (
            self.inference_aug
        ):  # if inference_aug, then apply augmentations during inference
            x = self._apply_augment(x)
        else:
            if (
                self.insert_max
            ):  # if insert_max is larger than 0, then pad each sequence with random DNA
                x = self._pad_end(x)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step without (or with) augmentations."""
        x, y = batch
        if (
            self.inference_aug
        ):  # if inference_aug, then apply augmentations during inference
            x = self._apply_augment(x)
        else:
            if (
                self.insert_max
            ):  # if insert_max is larger than 0, then pad each sequence with random DNA
                x = self._pad_end(x)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Prediction step without (or with) augmentations."""
        x = batch
        if (
            self.inference_aug
        ):  # if inference_aug, then apply augmentations during inference
            x = self._apply_augment(x)
        else:
            if (
                self.insert_max
            ):  # if insert_max is larger than 0, then pad each sequence with random DNA
                x = self._pad_end(x)
        return self(x)

    def _sample_aug_combos(self, batch_size):
        """Set the number of augmentations and randomly select augmentations to apply
        to each sequence.
        """
        # determine the number of augmentations per sequence
        if self.hard_aug:
            batch_num_aug = self.max_augs_per_seq * np.ones((batch_size,), dtype=int)
        else:
            batch_num_aug = np.random.randint(
                1, self.max_augs_per_seq + 1, (batch_size,)
            )

        # randomly choose which subset of augmentations from augment_list
        aug_combos = [
            list(sorted(np.random.choice(self.max_num_aug, sample, replace=False)))
            for sample in batch_num_aug
        ]
        return aug_combos

    def _apply_augment(self, x):
        """Apply augmentations to each sequence in batch, x."""
        # number of augmentations per sequence
        aug_combos = self._sample_aug_combos(x.shape[0])

        # apply augmentation combination to sequences
        x_new = []
        for aug_indices, seq in zip(aug_combos, x):
            seq = torch.unsqueeze(seq, dim=0)
            insert_status = True  # status to see if random DNA padding is needed
            for aug_index in aug_indices:
                seq = self.augment_list[aug_index](seq)
                if hasattr(self.augment_list[aug_index], "insert_max"):
                    insert_status = False
            if insert_status:
                if self.insert_max:
                    seq = self._pad_end(seq)
            x_new.append(seq)
        return torch.cat(x_new)

    def _pad_end(self, x):
        """Add random DNA padding of length insert_max to the end of each sequence in batch."""
        N, A, L = x.shape
        a = torch.eye(A)
        p = torch.tensor([1 / A for _ in range(A)])
        padding = torch.stack(
            [
                a[p.multinomial(self.insert_max, replacement=True)].transpose(0, 1)
                for _ in range(N)
            ]
        ).to(x.device)
        x_padded = torch.cat([x, padding.to(x.device)], dim=2)
        return x_padded

    def finetune_mode(self, optimizer=None):
        """Turn on finetune flag -- no augmentations during training."""
        self.finetune = True
        if optimizer != None:
            self.optimizer = optimizer


def load_model_from_checkpoint(model, checkpoint_path):
    """Load PyTorch lightning model from checkpoint.

    Parameters
    ----------
    model : RobustModel
        RobustModel instance.
    checkpoint_path : str
        path to checkpoint of model weights

    Returns
    -------
    RobustModel
        Object with weights and config loaded from checkpoint.
    """
    return model.load_from_checkpoint(
        checkpoint_path,
        model=model.model,
        criterion=model.criterion,
        optimizer=model.optimizer,
        augment_list=model.augment_list,
        max_augs_per_seq=model.max_augs_per_seq,
        hard_aug=model.hard_aug,
        finetune=model.finetune,
        inference_aug=model.inference_aug,
    )


# ------------------------------------------------------------------------
# Helper function
# ------------------------------------------------------------------------


def augment_max_len(augment_list):
    """Determine whether insertions are applied to determine the insert_max,
    which will be applied to pad other sequences with random DNA.

    Parameters
    ----------
    augment_list : list
        List of augmentations.

    Returns
    -------
    int
        Value for insert max.
    """
    insert_max = 0
    for augment in augment_list:
        if hasattr(augment, "insert_max"):
            insert_max = augment.insert_max
    return insert_max
