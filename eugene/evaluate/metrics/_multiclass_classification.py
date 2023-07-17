import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_auroc(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(roc_auc_score(y_true[:, class_index], y_score[:, class_index]))
    return np.array(vals)


def calculate_aupr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(
            average_precision_score(y_true[:, class_index], y_score[:, class_index])
        )
    return np.array(vals)
