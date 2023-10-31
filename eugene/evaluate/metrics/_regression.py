import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def calculate_mse(y_true, y_score):
    """Computes the mean squared error (MSE) for each prediction task.

    Copied from https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/utils.py.
    Expects y_true and y_score to be numpy arrays with shape (N, C) where N is the number of samples and 
    C is the number of classes.

    This function name will be changed in the future.

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
        Represents the MSE for each task.
    """
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(mean_squared_error(y_true[:, class_index], y_score[:, class_index]))
    return np.array(vals)


def calculate_pearsonr(y_true, y_score):
    """Computes the Pearson correlation coefficient (r) for each prediction task.

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
        Represents the Pearson correlation coefficient for each task.
    """
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.pearsonr(y_true[:, class_index], y_score[:, class_index])[0])
    return np.array(vals)


def calculate_spearmanr(y_true, y_score):
    """Computes the Spearmans rank correlation coefficient (r) for each prediction task.

    Copied from https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/utils.py.
    Expects y_true and y_score to be numpy arrays with shape (N, C) where N is the number of samples and 
    C is the number of classes.

    This function name will modified in future releases.

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
        Represents the Spearman rank correlation coefficient for each task.
    """
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.spearmanr(y_true[:, class_index], y_score[:, class_index])[0])
    return np.array(vals)
