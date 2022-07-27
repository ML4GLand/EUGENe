# Classics
import numpy as np

# PyTorch
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

# PyTorch Lightning
from pytorch_lightning.core.lightning import LightningModule

# EUGENE
from ...preprocessing._utils import ascii_decode

# omit_final_pool should be set to True in conv_kwargs


class BaseModel(LightningModule):
    def __init__(
        self,
        input_len,
        output_dim,
        strand="ss",
        task="regression",
        aggr=None,
        loss_fxn="mse",
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
        self.kwargs = kwargs

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, x_rev_comp=None) -> torch.Tensor:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        ID, x, x_rev_comp, y = batch
        ID = np.array(
            [ascii_decode(item) for item in ID.squeeze(dim=1).detach().cpu().numpy()]
        )
        y = y.detach().cpu().numpy()
        outs = self(x, x_rev_comp).squeeze(dim=1).detach().cpu().numpy()
        return np.column_stack([ID, outs, y])

    def _common_step(self, batch, batch_idx, stage: str):
        # Get and log loss
        ID, x, x_rev_comp, y = batch
        outs = self(x, x_rev_comp).squeeze(dim=1)
        loss = self.loss_fxn(outs, y)
        hp_metric = self._calculate_metric(outs, y)

        # Get and log metrics
        self.log(f"{stage}_loss", loss, on_epoch=True)
        self.log(f"{stage}_{self.hp_metric_name}", hp_metric, on_epoch=True)

        # Get and log "hp_metric" useful for hyperopt
        if stage == "val":
            self.log("hp_metric", hp_metric, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def _calculate_metric(self, outs, y):
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


def _metric_handler(metric_name, num_outputs):
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
