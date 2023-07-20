import numpy as np
from numpy.typing import NDArray
from typing import List, Literal, Optional, Union, cast


def binarize_values(
    values: NDArray[Union[float, int]],
    upper_threshold: float,
    lower_threshold: Optional[float] = None
) -> NDArray[Union[float, int]]:
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
