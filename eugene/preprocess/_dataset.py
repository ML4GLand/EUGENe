import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def split_train_test(X_data, y_data, split=0.8, subset=None, rand_state=13, shuf=True):
    """
    Function to perform train test splitting

    Wraps sklearn train_test_split function with allowance for a subset of the data to be used

    Parameters
    ----------
    X_data: numpy.ndarray
        The data to split
    y_data: numpy.ndarray
        The labels to split
    split: float, optional
        The percentage of the data to use for training
    subset: numpy.ndarray, optional
        The subset of the data to use for testing
    rand_state: int, optional
        The random state to use for splitting
    """
    train_X, test_X, train_y, test_y = train_test_split(
        X_data, y_data, train_size=split, random_state=rand_state, shuffle=shuf
    )
    if subset is not None:
        num_train = int(len(train_X) * subset)
        num_test = int(len(test_X) * subset)
        train_X, test_X, train_y, test_y = (
            train_X[:num_train, :],
            test_X[:num_test, :],
            train_y[:num_train],
            test_y[:num_test],
        )
    return train_X, test_X, train_y, test_y

def standardize_features(train_X, test_X, indeces=None, stats_file=None):
    """
    Function to standardize features based on passed in indeces and optionally save stats

    Parameters
    ----------
    train_X: numpy.ndarray
        The training data
    test_X: numpy.ndarray
        The testing data
    indeces: list of int, optional
        The indeces of the features to standardize
    stats_file: str, optional
        The file to save the stats to
    """
    if indeces is None:
        indeces = np.array(range(train_X.shape[1]))
    elif len(indeces) == 0:
        return train_X, test_X

    means = train_X[:, indeces].mean(axis=0)
    train_X_scaled = train_X[:, indeces] - means
    test_X_scaled = test_X[:, indeces] - means

    stds = train_X[:, indeces].std(axis=0)
    valid_std_idx = np.where(stds != 0)[0]
    indeces = indeces[valid_std_idx]
    stds = stds[valid_std_idx]
    train_X_scaled[:, indeces] = train_X_scaled[:, indeces] / stds
    test_X_scaled[:, indeces] = test_X_scaled[:, indeces] / stds

    if stats_file is not None:
        stats_dict = {"indeces": indeces, "means": means, "stds": stds}
        with open(stats_file, "wb") as handle:
            pickle.dump(stats_dict, handle)

    return train_X_scaled, test_X_scaled

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

def ohe_features(dataframe, feature_cols):
    ohe = OneHotEncoder(sparse=False)
    X = dataframe[feature_cols]
    ohe.fit(X)
    X_ohe = ohe.fit_transform(X)
    return X_ohe