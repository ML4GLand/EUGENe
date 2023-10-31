import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Literal, Optional, Union

def calculate_auroc(
    y_true: NDArray,
    y_score: NDArray,
) -> NDArray:
    """Computes the area under the receiver operating characteristic (AUROC) curve for each class.

    Copied from https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/utils.py.
    Expects y_true and y_score to be numpy arrays with shape (N, C) where N is the number of samples and 
    C is the number of classes.

    This function name will be changed to include the word "multiclass" in the future.

    Parameters
    ----------
    y_true : NDArray
        NumPy array of shape (N, C) where N is the number of samples and C is the number of classes.
        Represents the true labels.
    y_score : NDArray
        NumPy array of shape (N, C) where N is the number of samples and C is the number of classes.
        Represents the predicted scores.

    Returns
    -------
    NDArray
        NumPy array of shape (C,) where C is the number of classes.
        Represents the AUROC for each class.
    """
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(roc_auc_score(y_true[:, class_index], y_score[:, class_index]))
    return np.array(vals)


def calculate_aupr(y_true, y_score):
    """Computes the area under the precision-recall (AUPR) curve for each class.

    Copied from https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/utils.py.
    Expects y_true and y_score to be numpy arrays with shape (N, C) where N is the number of samples and
    C is the number of classes.
    Parameters
    ----------
    y_true : NDArray
        NumPy array of shape (N, C) where N is the number of samples and C is the number of classes.
        Represents the true labels.
    y_score : NDArray
        NumPy array of shape (N, C) where N is the number of samples and C is the number of classes.
        Represents the predicted scores.

    This function name will be changed to include the word "multiclass" in the future.
    
    Returns
    -------
    NDArray
        NumPy array of shape (C,) where C is the number of classes.
        Represents the AUPR for each class.
    """
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(
            average_precision_score(y_true[:, class_index], y_score[:, class_index])
        )
    return np.array(vals)
