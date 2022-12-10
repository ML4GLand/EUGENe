import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
