import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Callable, Optional
from pytorch_lightning import seed_everything
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from ._losses import LOSS_REGISTRY, MNLLLoss, log1pMSELoss
from ._optimizers import OPTIMIZER_REGISTRY
from ._schedulers import SCHEDULER_REGISTRY
from ._metrics import METRIC_REGISTRY, DEFAULT_TASK_METRICS, DEFAULT_METRIC_KWARGS


class SequenceModel(LightningModule, ABC):
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
    @abstractmethod
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        task: str = "regression",
        loss_fxn: Union[str, Callable] = "mse",
        optimizer: str = "adam",
        optimizer_lr: float = 1e-3,
        optimizer_kwargs: dict = {},
        scheduler: str = None,
        scheduler_monitor: str = "val_loss",
        scheduler_kwargs: dict = {},
        metric: str = None,
        metric_kwargs: dict = None,
        seed: int = None,
        save_hyperparams: bool = True,
        arch: str = None,
        name: str = None
    ):
        super().__init__()

        # Set the base attributes of a Sequence model
        self.input_len = input_len
        self.output_dim = output_dim
        self.task = task
        
        # Set the loss function
        self.loss_fxn = LOSS_REGISTRY[loss_fxn] if isinstance(loss_fxn, str) else loss_fxn
    
        # Set the optimizer
        self.optimizer = OPTIMIZER_REGISTRY[optimizer]
        self.optimizer_lr = optimizer_lr if optimizer_lr is not None else 1e-3
        self.optimizer_kwargs = optimizer_kwargs

        # Set the scheduler
        self.scheduler = SCHEDULER_REGISTRY[scheduler] if scheduler is not None else None
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_kwargs = scheduler_kwargs

        # Set the metric
        self.train_metric, self.metric_kwargs, self.metric_name = self.configure_metrics(metric=metric, metric_kwargs=metric_kwargs)
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
            self.save_hyperparameters()

        # Set the architecture
        self.arch = arch if arch is not None else self.__class__.__name__

        # Set the model name
        self.name = name if name is not None else "model"

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. This method must be implemented by the child class.

        Parameters:
        ----------
        x (torch.Tensor):
            input sequence
        """

    def predict(self, x, batch_size=32):
        """
        Predict the output of the model in batches
        """
        with torch.no_grad():
            device = self.device
            self.eval()
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            outs = []
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size].to(device)
                outs.append(self(batch))
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
        X, y = batch["seq"], batch["target"]
        outs = self(X).squeeze(dim=1)
        loss = self.loss_fxn(outs, y) # train
        return {
            "loss": loss, 
            "outs": outs.detach(), 
            "y": y.detach()
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        step_dict = self._common_step(batch, batch_idx, "train")
        self.log("train_loss", step_dict["loss"])
        self.train_metric(step_dict["outs"], step_dict["y"].long())
        self.log(f"train_{self.metric_name}", self.train_metric, on_step=True, on_epoch=True)
        return step_dict

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        step_dict = self._common_step(batch, batch_idx, "val")
        self.log("val_loss", step_dict["loss"], on_step=False, on_epoch=True)
        self.val_metric(step_dict["outs"], step_dict["y"].long())
        self.log(f"val_{self.metric_name}", self.val_metric, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step"""
        step_dict = self._common_step(batch, batch_idx, "test")
        self.log("test_loss", step_dict["loss"], on_step=False, on_epoch=True)
        self.test_metric(step_dict["outs"], step_dict["y"].long())
        self.log(f"test_{self.metric_name}", self.test_metric, on_step=False, on_epoch=True)

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
        metric_kwargs = metric_kwargs if metric_kwargs is not None else DEFAULT_METRIC_KWARGS[self.task]
        metric = METRIC_REGISTRY[metric_name](num_outputs=self.output_dim, **metric_kwargs)
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
            self.parameters(), 
            lr=self.optimizer_lr, 
            **self.optimizer_kwargs
        )
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = self.scheduler(
                optimizer,
                **self.scheduler_kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.scheduler_monitor,
            }

    def summary(self):
        """Print a summary of the model"""
        print(f"Model: {self.__class__.__name__}")
        print(f"Sequence length: {self.input_len}")
        print(f"Output dimension: {self.output_dim}")
        print(f"Task: {self.task}")
        print(f"Loss function: {self.loss_fxn.__name__}")
        print(f"Optimizer: {self.optimizer.__name__}")
        print(f"\tOptimizer parameters: {self.optimizer_kwargs}")
        print(f"\tOptimizer starting learning rate: {self.optimizer_lr}")
        print(f"Scheduler: {self.scheduler.__name__}") if self.scheduler is not None else print("Scheduler: None")
        print(f"\tScheduler parameters: {self.scheduler_kwargs}")
        print(f"Metric: {self.metric_name}")
        print(f"\tMetric parameters: {self.metric_kwargs}")
        print(f"Seed: {self.seed}")
        print(f"Parameters summary:")
        return ModelSummary(self)

    @property
    def input_len(self) -> int:
        """Input length"""
        return self._input_len
    
    @input_len.setter
    def input_len(self, value: int):
        """Set input length"""
        self._input_len = value

    @property
    def output_dim(self) -> int:
        """Output dimension"""
        return self._output_dim
    
    @output_dim.setter
    def output_dim(self, value: int):
        """Set output dimension"""
        self._output_dim = value
     
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
        arch: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.input_len = input_len
        self.output_dim = output_dim
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
    