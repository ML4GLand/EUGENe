from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from tqdm.auto import tqdm

from .base._losses import LOSS_REGISTRY
from .base._metrics import (
    DEFAULT_METRIC_KWARGS,
    DEFAULT_TASK_METRICS,
    METRIC_REGISTRY,
    calculate_metric,
)
from .base._optimizers import OPTIMIZER_REGISTRY
from .base._schedulers import SCHEDULER_REGISTRY


class SequenceModule(LightningModule):
    """
    Base sequence model class to be inherited by all models in EUGENe

    Parameters:
    ----------
    input_len (int):
        length of input sequence
    output_dim (int):
        number of output dimensions
    task (str):
        task of the model
    loss_fxn (str):
        loss function to use
    hp_metric (str):
        metric to use for hyperparameter tuning
    kwargs (dict):
        additional arguments to pass to the model
    """

    def __init__(
        self,
        arch: torch.nn.Module,
        task: str = "regression",
        loss_fxn: Union[str, Callable] = "mse",
        optimizer: str = "adam",
        optimizer_lr: float = 1e-3,
        optimizer_kwargs: dict = {},
        scheduler: str = None,
        scheduler_monitor: str = "val_loss_epoch",
        scheduler_kwargs: dict = {},
        metric: str = None,
        metric_kwargs: dict = None,
        seed: int = None,
        save_hyperparams: bool = True,
        arch_name: str = None,  # TODO: this should be a class attribute
        model_name: str = None
    ):
        super().__init__()

        # Set the base attributes of a Sequence model
        self.arch = arch
        self.input_len = arch.input_len
        self.output_dim = arch.output_dim

        # Set the task
        self.task = task
        self.arch_name = (
            arch_name if arch_name is not None else self.arch.__class__.__name__
        )

        # Set the loss function
        self.loss_fxn = (
            LOSS_REGISTRY[loss_fxn] if isinstance(loss_fxn, str) else loss_fxn
        )

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

        # Set the metric
        (
            self.train_metric,
            self.metric_kwargs,
            self.metric_name,
        ) = self.configure_metrics(metric=metric, metric_kwargs=metric_kwargs)
        self.val_metric = self.train_metric.clone()
        self.test_metric = self.train_metric.clone()

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

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the arch.

        Parameters:
        ----------
        x (torch.Tensor):
            input sequence
        """
        return self.arch(x)

    def predict(self, x, batch_size=128, verbose=True):
        """
        Predict the output of the model in batches.
        """
        with torch.no_grad():
            device = self.device
            self.eval()
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            outs = []
            for _, i in tqdm(
                enumerate(range(0, len(x), batch_size)),
                desc="Predicting on batches",
                total=len(x) // batch_size,
                disable=not verbose,
            ):
                batch = x[i : i + batch_size].to(device)
                out = self(batch).detach().cpu()
                outs.append(out)
            return torch.cat(outs)

    def _common_step(self, batch, batch_idx, stage: str):
        """Common step for training, validation and test

        Parameters:
        ----------
        batch (tuple):
            batch of data
        batch_idx (int):
            index of the batch
        stage (str):
            stage of the training

        Returns:
        ----------
        dict:
            dictionary of metrics
        """
        # Get and log loss
        X, y = batch["ohe_seq"], batch["target"]
        outs = self(X).squeeze()
        loss = self.loss_fxn(outs, y)  # train
        return {"loss": loss, "outs": outs.detach(), "y": y.detach()}

    def training_step(self, batch, batch_idx):
        """Training step"""
        step_dict = self._common_step(batch, batch_idx, "train")
        self.log(
            "train_loss", step_dict["loss"], on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train_loss_epoch", step_dict["loss"], on_step=False, on_epoch=True)
        calculate_metric(
            self.train_metric, self.metric_name, step_dict["outs"], step_dict["y"]
        )
        self.log(
            f"train_{self.metric_name}_epoch",
            self.train_metric,
            on_step=False,
            on_epoch=True,
        )
        return step_dict

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        step_dict = self._common_step(batch, batch_idx, "val")
        self.log("val_loss_epoch", step_dict["loss"], on_step=False, on_epoch=True)
        calculate_metric(
            self.val_metric, self.metric_name, step_dict["outs"], step_dict["y"]
        )
        self.log(
            f"val_{self.metric_name}_epoch",
            self.val_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        """Test step"""
        step_dict = self._common_step(batch, batch_idx, "test")
        self.log("test_loss", step_dict["loss"], on_step=False, on_epoch=True)
        calculate_metric(
            self.test_metric, self.metric_name, step_dict["outs"], step_dict["y"]
        )
        self.log(
            f"test_{self.metric_name}", self.test_metric, on_step=False, on_epoch=True
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Predict step"""
        X, y = batch["ohe_seq"], batch["target"]
        y = y.detach().cpu().numpy()
        outs = self(X).squeeze().detach().cpu().numpy()
        return np.column_stack([outs, y])

    def configure_metrics(self, metric, metric_kwargs):
        """Configure metrics
        Keeping this a function allows for the metric to be reconfigured
        in inherited classes
        TODO: add support for multiple metrics
        Returns:
        ----------
        torchmetrics.Metric:
            metric
        """
        metric_name = DEFAULT_TASK_METRICS[self.task] if metric is None else metric
        metric_kwargs = (
            metric_kwargs
            if metric_kwargs is not None
            else DEFAULT_METRIC_KWARGS[self.task]
        )
        metric = METRIC_REGISTRY[metric_name](
            num_outputs=self.output_dim, **metric_kwargs
        )
        return metric, metric_kwargs, metric_name

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

    def summary(self):
        """Print a summary of the model"""
        print(f"Model: {self.arch.__class__.__name__}")
        print(f"Sequence length: {self.input_len}")
        print(f"Output dimension: {self.output_dim}")
        print(f"Task: {self.task}")
        print(f"Loss function: {self.loss_fxn.__name__}")
        print(f"Optimizer: {self.optimizer.__name__}")
        print(f"\tOptimizer parameters: {self.optimizer_kwargs}")
        print(f"\tOptimizer starting learning rate: {self.optimizer_lr}")
        print(
            f"Scheduler: {self.scheduler.__name__}"
        ) if self.scheduler is not None else print("Scheduler: None")
        print(f"\tScheduler parameters: {self.scheduler_kwargs}")
        print(f"Metric: {self.metric_name}")
        print(f"\tMetric parameters: {self.metric_kwargs}")
        print(f"Seed: {self.seed}")
        print(f"Parameters summary:")
        return ModelSummary(self)

    @property
    def arch(self) -> nn.Module:
        """Model"""
        return self._arch

    @arch.setter
    def arch(self, value: nn.Module):
        """Set model"""
        self._arch = value

    @property
    def task(self) -> str:
        """Task"""
        return self._task

    @task.setter
    def task(self, value: str):
        """Set task"""
        self._task = value

    @property
    def loss_fxn(self) -> Callable:
        """Loss function"""
        return self._loss_fxn

    @loss_fxn.setter
    def loss_fxn(self, value: Callable):
        """Set loss function"""
        self._loss_fxn = value

    @property
    def optimizer(self) -> Callable:
        """Optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Callable):
        """Set optimizer"""
        self._optimizer = value

    @property
    def optimizer_lr(self) -> float:
        """Optimizer starting learning rate"""
        return self._optimizer_lr

    @optimizer_lr.setter
    def optimizer_lr(self, value: float):
        """Set optimizer starting learning rate"""
        self._optimizer_lr = value

    @property
    def scheduler(self) -> Callable:
        """Scheduler"""
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value: Callable):
        """Set scheduler"""
        self._scheduler = value

    @property
    def train_metric(self) -> str:
        """Train metric"""
        return self._train_metric

    @train_metric.setter
    def train_metric(self, value: str):
        """Set train metric"""
        self._train_metric = value

    @property
    def seed(self) -> int:
        """Seed"""
        return self._seed

    @seed.setter
    def seed(self, value: int):
        """Set seed"""
        self._seed = value
