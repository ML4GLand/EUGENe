# performance.py
# Authors: Alex Tseng <amtseng@stanford.edu>
#          Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains performance measures that are used to evaluate the
model but are not explicitly used as losses for optimization.

IMPORTANT: MANY OF THESE FUNCTIONS ASSUME THE INPUTS TO BE PREDICTED LOG
PROBABILITIES AND TRUE COUNTS. THE FIRST ARGUMENT MUST BE IN LOG SPACE
AND THE SECOND ARGUMENT MUST BE IN COUNT SPACE FOR THESE FUNCTIONS.
"""

import torch

from ...models.base._losses import MNLLLoss
from ...models.base._losses import log1pMSELoss


def smooth_gaussian1d(x, kernel_sigma, kernel_width):
    """Smooth a signal along the sequence length axis.

    This function is a replacement for the scipy.ndimage.gaussian1d
    function that works on PyTorch tensors. It applies a Gaussian kernel
    to each position which is equivalent to applying a convolution across
    the sequence with weights equal to that of a Gaussian distribution.
    Each sequence, and each channel within the sequence, is smoothed
    independently.

    Parameters
    ----------
    x: torch.tensor, shape=(n_sequences, n_channels, seq_len)
        A tensor to smooth along the last axis. n_channels must be at
        least 1.

    kernel_sigma: float
        The standard deviation of the Gaussian to be applied.

    kernel_width: int
        The width of the kernel to be applied.

    Returns
    -------
    x_smooth: torch.tensor, shape=(n_sequences, n_channels, seq_len)
        The smoothed tensor.
    """

    meshgrid = torch.arange(kernel_width, dtype=torch.float32,
        device=x.device)

    mean = (kernel_width - 1.) / 2.
    kernel = torch.exp(-0.5 * ((meshgrid - mean) / kernel_sigma) ** 2.0)
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.reshape(1, 1, kernel_width).repeat(x.shape[1], 1, 1)
    return torch.nn.functional.conv1d(x, weight=kernel, groups=x.shape[1], 
        padding='same')

def batched_smoothed_function(logps, true_counts, f, smooth_predictions=False, 
    smooth_true=False, kernel_sigma=7, kernel_width=81, 
    exponentiate_logps=False, batch_size=200):
    """Batch a calculation with optional smoothing.

    Given a set of predicted and true values, apply some function to them in
    a batched manner and store the results. Optionally, either the true values
    or the predicted ones can be smoothed.

    Parameters
    ----------
    logps: torch.tensor
        A tensor of the predicted log probability values.

    true_counts: torch.tensor
        A tensor of the true values, usually integer counts.

    f: function
        A function to be applied to the predicted and true values.

    smooth_predictions: bool, optional
        Whether to apply a Gaussian filter to the predictions. Default is 
        False.

    smooth_true: bool, optional
        Whether to apply a Gaussian filter to the true values. Default is
        False.

    kernel_sigma: float, optional
        The standard deviation of the Gaussian to be applied. Default is 7.

    kernel_width: int, optional
        The width of the kernel to be applied. Default is 81.

    exponentiate_logps: bool, optional
        Whether to exponentiate each batch of log probabilities. Default is
        False.

    batch_size: int, optional
        The number of examples in each batch to evaluate at a time. Default
        is 200.


    Returns
    -------
    results: torch.tensor
        The results of applying the function to the tensor.
    """

    n = logps.shape[0]
    results = torch.empty(*logps.shape[:2])

    for start in range(0, n, batch_size):
        end = start + batch_size

        logps_ = logps[start:end]
        true_counts_ = true_counts[start:end]

        if smooth_predictions:
            logps_ = torch.exp(logps_)
            logps_ = smooth_gaussian1d(logps_, kernel_sigma, kernel_width)

            if exponentiate_logps == False:
                logps_ = torch.log(logps_)
        else:
            if exponentiate_logps:
                logps_ = torch.exp(logps_)

        if smooth_true:
            true_counts_ = smooth_gaussian1d(true_counts_, kernel_sigma, kernel_width)

        results[start:end] = f(logps_, true_counts_) 

    return results

def _kl_divergence(probs1, probs2):
    """
    Computes the KL divergence in the last dimension of `probs1` and `probs2`
    as KL(P1 || P2). `probs1` and `probs2` must be the same shape. For example,
    if they are both A x B x L arrays, then the KL divergence of corresponding
    L-arrays will be computed and returned in an A x B array. Does not
    renormalize the arrays. If probs2[i] is 0, that value contributes 0.
    """

    idxs = ((probs1 != 0) & (probs2 != 0))
    quot_ = torch.divide(probs1, probs2)

    quot = torch.ones_like(probs1)
    quot[idxs] = quot_[idxs]
    return torch.sum(probs1 * torch.log(quot), dim=-1)

def jensen_shannon_distance(logps, true_counts):
    """
    Computes the Jesnsen-Shannon distance in the last dimension of `probs1` and
    `probs2`. `probs1` and `probs2` must be the same shape. For example, if they
    are both A x B x L arrays, then the KL divergence of corresponding L-arrays
    will be computed and returned in an A x B array. This will renormalize the
    arrays so that each subarray sums to 1. If the sum of a subarray is 0, then
    the resulting JSD will be NaN.
    """
    # Renormalize both distributions, and if the sum is NaN, put NaNs all around

    probs1 = torch.exp(logps)
    probs1_sum = torch.sum(probs1, dim=-1, keepdims=True)
    probs1 = torch.divide(probs1, probs1_sum, out=torch.zeros_like(probs1))

    probs2_sum = torch.sum(true_counts, dim=-1, keepdims=True)
    probs2 = torch.divide(true_counts, probs2_sum, out=torch.zeros_like(true_counts))

    mid = 0.5 * (probs1 + probs2)
    return 0.5 * (_kl_divergence(probs1, mid) + _kl_divergence(probs2, mid))

def pearson_corr(arr1, arr2):
    """The Pearson correlation between two tensors across the last axis.

    Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array.

    Parameters
    ----------
    arr1: torch.tensor
        One of the tensor to correlate.

    arr2: torch.tensor
        The other tensor to correlation.

    Returns
    -------
    correlation: torch.tensor
        The correlation for each element, calculated along the last axis.
    """

    mean1 = torch.mean(arr1, axis=-1).unsqueeze(-1)
    mean2 = torch.mean(arr2, axis=-1).unsqueeze(-1)
    dev1, dev2 = arr1 - mean1, arr2 - mean2

    sqdev1, sqdev2 = torch.square(dev1), torch.square(dev2)
    numer = torch.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = torch.sum(sqdev1, axis=-1), torch.sum(sqdev2, axis=-1)  # Variances
    denom = torch.sqrt(var1 * var2)
   
    # Divide numerator by denominator, but use 0 where the denominator is 0
    correlation = torch.zeros_like(numer)
    correlation[denom != 0] = numer[denom != 0] / denom[denom != 0]
    return correlation


def spearman_corr(arr1, arr2):
    """The Spearman correlation between two tensors across the last axis.

    Computes the Spearman correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array.

    A dense ordering is used and ties are broken based on position in the
    tensor.

    Parameters
    ----------
    arr1: torch.tensor
        One of the tensor to correlate.

    arr2: torch.tensor
        The other tensor to correlation.

    Returns
    -------
    correlation: torch.tensor
        The correlation for each element, calculated along the last axis.
    """

    ranks1 = arr1.argsort().argsort().type(torch.float32)
    ranks2 = arr2.argsort().argsort().type(torch.float32)
    return pearson_corr(ranks1, ranks2)


def mean_squared_error(arr1, arr2):
    """The mean squared error between two tensors averaged along the last axis.

    Computes the element-wise squared error between two tensors and averages
    these across the last dimension. `arr1` and `arr2` must be the same shape. 
    For example, if they are both A x B x L arrays, then the correlation of 
    corresponding L-arrays will be computed and returned in an A x B array.

    Parameters
    ----------
    arr1: torch.tensor
        A tensor of values.

    arr2: torch.tensor
        Another tensor of values.

    Returns
    -------
    mse: torch.tensor
        The L2 distance between two tensors.
    """

    return torch.mean(torch.square(arr1 - arr2), axis=-1)

def calculate_performance_measures(logps, true_counts, pred_log_counts,
    kernel_sigma=7, kernel_width=81, smooth_true=False, 
    smooth_predictions=False, measures=None):
    """
    Computes some evaluation metrics on a set of positive examples, given the
    predicted profiles/counts, and the true profiles/counts.
    Arguments:
        `true_profs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `log_pred_profs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities 
        `true_counts`: a N x T x 2 array, containing the true total counts
            for each task and strand
        `log_pred_counts`: a N x T x 2 array, containing the predicted LOG total
            counts for each task and strand
        `smooth_true_profs`: if True, smooth the true profiles before computing
            JSD and correlations; true profiles will not be smoothed for any
            other metric
        `smooth_pred_profs`: if True, smooth the predicted profiles before
            computing NLL, cross entropy, JSD, and correlations; predicted
            profiles will not be smoothed for any other metric
        `print_updates`: if True, print out updates and runtimes
    Returns a dictionary with the following:
        A N x T-array of the average negative log likelihoods for the profiles
            (given predicted probabilities, the likelihood for the true counts),
            for each sample/task (strands averaged)
        A N x T-array of the average cross entropy for the profiles (given
            predicted probabilities, the likelihood for the true counts), for
            each sample/task (strands averaged)
        A N x T array of average Jensen-Shannon divergence between the predicted
            and true profiles (strands averaged)
        A N x T array of the Pearson correlation of the predicted and true (log)
            counts, for each sample/task (strands pooled)
        A N x T array of the Spearman correlation of the predicted and true
            (log) counts, for each sample/task (strands pooled)
        A N x T array of the mean squared error of the predicted and true (log)
            counts, for each sample/task (strands pooled)
        A T-array of the Pearson correlation of the (log) total counts, over all
            strands and samples
        A T-array of the Spearman correlation of the (log) total counts, over
            all strands and samples
        A T-array of the mean squared error of the (log) total counts, over all
            strands and samples
    """

    measures_ = {}

    if measures is None or 'profile_mnll' in measures: 
        measures_['profile_mnll'] = batched_smoothed_function(logps=logps, 
            true_counts=true_counts, f=MNLLLoss, 
            smooth_predictions=smooth_predictions, smooth_true=False, 
            kernel_sigma=kernel_sigma, kernel_width=kernel_width)

    if measures is None or 'profile_jsd' in measures: 
        measures_['profile_jsd'] = batched_smoothed_function(logps=logps, 
            true_counts=true_counts, f=jensen_shannon_distance, 
            smooth_predictions=smooth_predictions, smooth_true=smooth_true, 
            kernel_sigma=kernel_sigma, kernel_width=kernel_width)

    if measures is None or 'profile_pearson' in measures:
        measures_['profile_pearson'] = batched_smoothed_function(logps=logps, 
            true_counts=true_counts, f=pearson_corr, 
            smooth_predictions=smooth_predictions, smooth_true=smooth_true, 
            exponentiate_logps=True, kernel_sigma=kernel_sigma, 
            kernel_width=kernel_width)

    if measures is None or 'profile_spearman' in measures:
        measures_['profile_spearman'] = batched_smoothed_function(logps=logps, 
            true_counts=true_counts, f=spearman_corr, 
            smooth_predictions=smooth_predictions, smooth_true=smooth_true, 
            exponentiate_logps=True, kernel_sigma=kernel_sigma, 
            kernel_width=kernel_width)


    # Total count correlations/MSE
    true_log_counts = torch.log(true_counts.sum(dim=-1)+1)

    if measures is None or 'count_pearson' in measures:
        measures_['count_pearson'] = pearson_corr(pred_log_counts.T, 
            true_log_counts.T)

    if measures is None or 'count_spearman' in measures:
        measures_['count_spearman'] = spearman_corr(pred_log_counts.T, 
            true_log_counts.T)

    if measures is None or 'count_mse' in measures:
        measures_['count_mse'] = mean_squared_error(pred_log_counts.T, 
            true_log_counts.T)

    return measures_