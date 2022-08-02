import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchmetrics import (
    R2Score,
    ExplainedVariance,
    Accuracy,
    AUROC,
    F1Score,
    Precision,
    Recall,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from ...preprocessing._utils import ascii_decode


class BaseModel(LightningModule):
    """Base model for all models

    Attributes:
    ----------

    input_len (int):
        length of input sequence
    output_dim (int):
        number of output dimensions
    strand (str):
        strand of the input sequence
    task (str):
        task of the model
    aggr (str):
        aggregation method for the input sequence
    loss_fxn (str):
        loss function to use
    hp_metric (str):
        metric to use for hyperparameter tuning
    kwargs (dict):
        additional arguments to pass to the model
    """

    def __init__(
        self,
        input_len,
        output_dim,
        strand="ss",
        task="regression",
        aggr=None,
        loss_fxn="mse",
        optimizer="adam",
        lr=1e-3,
        scheduler=None,
        scheduler_patience=None,
        hp_metric=None,
        **kwargs,
    ):
        super().__init__()

        # Instance variables
        self.input_len = input_len
        self.output_dim = output_dim
        self.strand = strand
        self.task = task
        self.aggr = aggr
        self.loss_fxn = loss_fxn_dict[loss_fxn]
        self.hp_metric_name = (
            hp_metric if hp_metric is not None else default_hp_metric_dict[self.task]
        )
        self.hp_metric = _metric_handler(self.hp_metric_name, output_dim)
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_patience = scheduler_patience
        self.kwargs = kwargs

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, x_rev_comp=None) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
        ----------
        x (torch.Tensor):
            input sequence
        x_rev_comp (torch.Tensor):
            reverse complement of the input sequence
        """
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        """Training step"""
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Test step"""
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        """Predict step

        Parameters:
        ----------
        batch (tuple):
            batch of data
        batch_idx (int):
            index of the batch

        Returns:
        ----------
        dict:
            dictionary of predictions
        """
        ID, x, x_rev_comp, y = batch
        ID = np.array(
            [ascii_decode(item) for item in ID.squeeze(dim=1).detach().cpu().numpy()]
        )
        y = y.detach().cpu().numpy()
        outs = self(x, x_rev_comp).squeeze(dim=1).detach().cpu().numpy()
        return np.column_stack([ID, outs, y])

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
        ID, x, x_rev_comp, y = batch
        outs = self(x, x_rev_comp).squeeze(dim=1)
        loss = self.loss_fxn(outs, y)
        hp_metric = self._calculate_metric(outs, y)

        # Get and log metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, rank_zero_only=True)
        self.log(
            f"{stage}_{self.hp_metric_name}",
            hp_metric,
            on_epoch=True,
            rank_zero_only=True,
        )

        # Get and log "hp_metric" useful for hyperopt
        if stage == "val":
            self.log("hp_metric", hp_metric, on_epoch=True, rank_zero_only=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = (
            ReduceLROnPlateau(optimizer, patience=self.scheduler_patience)
            if self.scheduler_patience is not None
            else None
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _calculate_metric(self, outs, y):
        """Calculate metric

        Parameters:
        ----------
        outs (torch.Tensor):
            output of the model
        y (torch.Tensor):
            ground truth

        Returns:
        ----------
        torch.Tensor:
            metric
        """
        if self.hp_metric_name == "r2":
            return self.hp_metric(outs, y)
        elif self.hp_metric_name == "auroc":
            if self.task == "binary_classification":
                probs = torch.sigmoid(outs)
                return self.hp_metric(probs, y.long())
            return self.hp_metric(outs, y.long())
        elif self.hp_metric_name == "accuracy":
            preds = torch.round(torch.sigmoid(outs))
            return self.hp_metric(preds, y.long())

    def summary(self):
        """Summary of the model"""
        print(f"Model: {self.__class__.__name__}")
        print(f"Input length: {self.input_len}")
        print(f"Output dimension: {self.output_dim}")
        print(f"Strand: {self.strand}")
        print(f"Task: {self.task}")
        print(f"Aggregation: {self.aggr}")
        print(f"Loss function: {self.loss_fxn.__name__}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Learning rate: {self.lr}")
        print(f"Scheduler: {self.scheduler}")
        print(f"Scheduler patience: {self.scheduler_patience}")
        return ModelSummary(self)


def _metric_handler(metric_name, num_outputs):
    """Handler for metrics

    Parameters:
    ----------
    metric_name (str):
        name of the metric
    num_outputs (int):
        number of outputs of the model

    Returns:
    ----------
    torch.Tensor:
        metric
    """
    if metric_name == "r2":
        return R2Score(num_outputs=num_outputs)
    elif metric_name == "explained_variance":
        return ExplainedVariance()
    elif metric_name == "auroc":
        return AUROC(num_classes=num_outputs)
    elif metric_name == "accuracy":
        return Accuracy(num_classes=num_outputs)
    elif metric_name == "f1":
        return F1Score(num_classes=num_outputs)
    elif metric_name == "precision":
        return Precision(num_classes=num_outputs)
    elif metric_name == "recall":
        return Recall(num_classes=num_outputs)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


loss_fxn_dict = {
    "mse": F.mse_loss,
    "poisson": F.poisson_nll_loss,
    "bce": F.binary_cross_entropy_with_logits,
    "cross_entropy": F.cross_entropy,
}


default_hp_metric_dict = {
    "regression": "r2",
    "multitask_regression": "r2",
    "binary_classification": "auroc",
    "multiclass_classification": "auroc",
}


optimizer_dict = {"adam": Adam, "sgd": SGD}


lr_scheduler_dict = {"reduce_lr_on_plateau": ReduceLROnPlateau}
