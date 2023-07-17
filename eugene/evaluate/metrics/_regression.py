import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def calculate_mse(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(mean_squared_error(y_true[:, class_index], y_score[:, class_index]))
    return np.array(vals)


def calculate_pearsonr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.pearsonr(y_true[:, class_index], y_score[:, class_index])[0])
    return np.array(vals)


def calculate_spearmanr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.spearmanr(y_true[:, class_index], y_score[:, class_index])[0])
    return np.array(vals)
