import torchmetrics

METRIC_REGISTRY = {
    "r2score": torchmetrics.R2Score,
    "explainedvariance": torchmetrics.ExplainedVariance,
    "auroc": torchmetrics.AUROC, # can be binary, multiclass, or multilabel, can handle logits or probabilities
    "accuracy": torchmetrics.Accuracy, # can be binary, multiclass, or multilabel, can handle logits or probabilities
    "f1score": torchmetrics.F1Score,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
}

DEFAULT_TASK_METRICS = {
    "regression": "r2score",
    "multitask_regression": "r2score",
    "binary_classification": "auroc",
    "multiclass_classification": "auroc",
    "multilabel_classification": "auroc"
}

DEFAULT_METRIC_KWARGS = {
    "regression": {},
    "binary_classification": {"task": "binary"},
    "multiclass_classification": {"task": "multiclass"},
    "multilabel_classification": {"task": "multilabel"},
}
