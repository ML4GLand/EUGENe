import torch
import torchmetrics

METRIC_REGISTRY = {
    "r2score": torchmetrics.R2Score,
    "pearson": torchmetrics.PearsonCorrCoef,
    "spearman": torchmetrics.SpearmanCorrCoef,
    "explainedvariance": torchmetrics.ExplainedVariance,
    "auroc": torchmetrics.AUROC,  # can be binary, multiclass, or multilabel, can handle logits or probabilities
    "accuracy": torchmetrics.Accuracy,  # can be binary, multiclass, or multilabel, can handle logits or probabilities
    "f1score": torchmetrics.F1Score,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
}

DEFAULT_TASK_METRICS = {
    "regression": "r2score",
    "multitask_regression": "r2score",
    "binary_classification": "auroc",
    "multiclass_classification": "auroc",
    "multilabel_classification": "auroc",
}

DEFAULT_METRIC_KWARGS = {
    "regression": {},
    "binary_classification": {"task": "binary"},
    "multiclass_classification": {"task": "multiclass"},
    "multilabel_classification": {"task": "multilabel"},
}


def calculate_metric(metric, metric_name, outs, y):
    """Calculate a metric from a metric name and the model outputs and targets.

    Args:
        metric (torchmetrics.Metric): The metric to calculate.
        metric_name (str): The name of the metric.
        y (torch.Tensor): The targets.
        outs (torch.Tensor): The model outputs.

    Returns:
        float: The calculated metric.
    """
    if metric_name in ["accuracy", "auroc", "f1score", "precision", "recall"]:
        if len(y.shape) > 1:
            y = torch.argmax(y.squeeze(), dim=1)
    else:
        outs = outs.squeeze()
    metric(outs, y)
