import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def binarize_values(values, upper_threshold=0.5, lower_threshold=None):
    """
    Function to binarize values based on thresholds

    Parameters
    ----------
    values: numpy.ndarray
        The values to binarize
    upper_threshold: float, optional
        The upper threshold to use
    lower_threshold: float, optional
        The lower threshold to use
    """
    bin_values = np.where(values > upper_threshold, 1, np.nan)
    if lower_threshold is not None:
        bin_values = np.where(values < lower_threshold, 0, bin_values)
    else:
        bin_values = np.where(values <= upper_threshold, 0, bin_values)
    return bin_values


def ohe_features(values):
    """
    Function to one-hot encode features

    Parameters
    ----------
    values: numpy.ndarray
        The values to one-hot encode
    """
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(values)
    values_ohe = ohe.fit_transform(values)
    return values_ohe
