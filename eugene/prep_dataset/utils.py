import yaml
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.stats import kruskal, f_oneway
from statsmodels.stats.multitest import fdrcorrection


def merge_parameters(parameters, default_parameters):
	"""Merge the provided parameters with the default parameters.

	
	Parameters
	----------
	parameters: str
		Name of the JSON folder with the provided parameters

	default_parameters: dict
		The default parameters for the operation.


	Returns
	-------
	params: dict
		The merged set of parameters.
	"""

	with open(parameters, "r") as infile:
		parameters = yaml.safe_load(infile)

	unset_parameters = ()
	for param_class in default_parameters.keys():
		for parameter, value in default_parameters[param_class].items():
			if parameter not in parameters:
				if value is None and parameter not in unset_parameters:
					raise ValueError("Must provide value for '{}'".format(parameter))

				parameters[param_class][parameter] = value

	return parameters


def infer_covariate_types(df: pd.DataFrame, categorical_threshold=0.05):
    """
    Infers the type of covariate for each column in a DataFrame.

    Args:
    - df: Input pandas DataFrame.
    - categorical_threshold: The threshold (as a proportion of the total rows) 
      below which a numeric column is considered categorical. Default is 5%.

    Returns:
    - A dictionary where the keys are column names and the values are the inferred types:
      'binary', 'categorical', or 'continuous'.
    """
    covariate_types = {}

    for col in df.columns:
        # Drop NaN values for proper evaluation
        unique_values = df[col].dropna().unique()
        num_unique = len(unique_values)
        total_rows = df[col].dropna().shape[0]

        # Skip columns with all NaNs
        if total_rows == 0:
            continue

        # Check if column is binary (exactly 2 unique values)
        if num_unique == 2:
            covariate_types[col] = 'binary'

        # Check if column is categorical (non-numeric or few unique values)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, decide based on the number of unique values
            unique_ratio = num_unique / total_rows

            # If unique_ratio is below a threshold, consider it categorical
            if unique_ratio < categorical_threshold:
                covariate_types[col] = 'categorical'
            else:
                covariate_types[col] = 'continuous'
        else:
            # For non-numeric types, it's considered categorical if there are multiple unique values
            covariate_types[col] = 'categorical' if num_unique > 1 else 'binary'

    return covariate_types


def correlate_cnt_with_continuous(
    cnt,
    covariate,
    normalize=False,
    method="pearson",
):
    """Calculate the correlation between two variables.

    Parameters
    ----------
    cnt: np.ndarray
        The count variable as a 1D numpy array.
    covariate: np.ndarray
        The continuous variable as a 1D numpy array.
    normalize: bool
        Whether to normalize the count so that it sums to 1.
    method: str
        The method to use for correlation. Options are "pearson", "spearman", and "kendall".

    Returns
    -------
    (float, float)
        The correlation coefficient and p-value as a tuple.
    """
    if sum(cnt) == 0:
        return (np.nan, np.nan)
    
    if normalize:
        cnt = cnt / cnt.sum()

    if method == "pearson":
        return pearsonr(cnt, covariate)
    elif method == "spearman":
        return spearmanr(cnt, covariate)
    elif method == "kendall":
        return kendalltau(cnt, covariate)
    else:
        raise ValueError(f"Unknown method: {method}")
    

def run_continuous_correlations(
    cnts: np.ndarray,
    covariate: np.ndarray,
    method: str = "pearson",
):
    """Calculate the correlation between each column of counts and a continuous covariate.

    Parameters
    ----------
    counts: np.ndarray
        The count variables as a 2D numpy array.
    covariate: np.ndarray
        The continuous variable as a 1D numpy array.
    method: str
        The method to use for correlation. Options are "pearson", "spearman", and "kendall".

    Returns
    -------
    (np.ndarray, np.ndarray)
        The correlation coefficients and p-values as numpy arrays.
    """
    n = cnts.shape[1]
    corrs = np.zeros(n)
    pvals = np.zeros(n)

    for i in range(n):
        corrs[i], pvals[i] = correlate_cnt_with_continuous(cnts[:, i], covariate, method=method)

    return corrs, pvals


def correlate_cnt_with_binary(
    cnt,
    binary,
    normalize=False,
    method="mannwhitneyu",
):
    """Calculate the correlation between a count variable and a binary variable.

    Parameters
    ----------
    cnt: np.ndarray
        The count variable as a 1D numpy array.
    binary: np.ndarray
        The binary variable as a 1D numpy array.
    normalize: bool
        Whether to normalize the count so that it sums to 1.
    method: str
        The method to use for correlation. Options are "mannwhitneyu", "ttest_ind", "pearson", "spearman", and "kendall".

    Returns
    -------
    (float, float)
        The correlation coefficient and p-value as a tuple.
    """
    if sum(cnt) == 0:
        return (np.nan, np.nan)
    
    if normalize:
        cnt = cnt / cnt.sum()

    if method == "mannwhitneyu":
        return mannwhitneyu(cnt, binary)
    elif method == "ttest_ind":
        return ttest_ind(cnt, binary)
    elif method in ["pearson", "spearman", "kendall"]:
        return correlate_cnt_with_continuous(cnt, binary, normalize=normalize, method=method)
    

def run_binary_correlations(
    cnts: np.ndarray,
    binary: np.ndarray,
    method: str = "mannwhitneyu",
):
    """Calculate the correlation between each column of counts and a binary covariate.

    Parameters
    ----------
    counts: np.ndarray
        The count variables as a 2D numpy array.
    binary: np.ndarray
        The binary variable as a 1D numpy array.
    method: str
        The method to use for correlation. Options are "mannwhitneyu" and "ttest_ind".

    Returns
    -------
    (np.ndarray, np.ndarray)
        The correlation coefficients and p-values as numpy arrays.
    """
    n = cnts.shape[1]
    corrs = np.zeros(n)
    pvals = np.zeros(n)

    for i in range(n):
        corrs[i], pvals[i] = correlate_cnt_with_binary(cnts[:, i], binary, method=method)

    return corrs, pvals


def correlate_cnt_with_categorical(
    cnt,
    categorical,
    normalize=False,
    method="kruskal",
):
    """Calculate the correlation between a count variable and a categorical variable.

    Parameters
    ----------
    cnt: np.ndarray
        The count variable as a 1D numpy array.
    categorical: np.ndarray
        The categorical variable as a 1D numpy array.
    normalize: bool
        Whether to normalize the count so that it sums to 1.
    method: str
        The method to use for correlation. Options are "kruskal" and "f_oneway".

    Returns
    -------
    (float, float)
        The correlation coefficient and p-value as a tuple.
    """
    if sum(cnt) == 0:
        return (np.nan, np.nan)
    
    if normalize:
        cnt = cnt / cnt.sum()

    if method == "kruskal":
        return kruskal(*[cnt[categorical == i] for i in np.unique(categorical)])
    elif method == "f_oneway":
        return f_oneway(*[cnt[categorical == i] for i in np.unique(categorical)])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    
def run_categorical_correlations(
    cnts: np.ndarray,
    categorical: np.ndarray,
    method: str = "kruskal",
):
    """Calculate the correlation between each column of counts and a categorical covariate.

    Parameters
    ----------
    counts: np.ndarray
        The count variables as a 2D numpy array.
    categorical: np.ndarray
        The categorical variable as a 1D numpy array.
    method: str
        The method to use for correlation. Options are "kruskal" and "f_oneway".

    Returns
    -------
    (np.ndarray, np.ndarray)
        The correlation coefficients and p-values as numpy arrays.
    """
    n = cnts.shape[1]
    corrs = np.zeros(n)
    pvals = np.zeros(n)

    for i in range(n):
        corrs[i], pvals[i] = correlate_cnt_with_categorical(cnts[:, i], categorical, method=method)

    return corrs, pvals
