import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
tqdm.pandas()
from sklearn.metrics import auc
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


# Useful helpers for generating and checking for kmers
def generate_all_possible_kmers(n=7, alphabet="AGCU"):
    """
    Generate all possible kmers of length and alphabet provided
    """
    return ["".join(c) for c in product(alphabet, repeat=n)]


def kmer_in_seqs(seqs, kmer):
    """
    Return a 0/1 array of whether a kmer is in each of the passed in sequences
    """
    seqs_s = pd.Series(seqs)
    kmer_binary = seqs_s.str.contains(kmer).astype(int).values
    return kmer_binary


def calc_auc(y, z, want_curve=False):
    # https://github.com/jisraeli/DeepBind/blob/master/code/deepfind.py
    """Given predictions z and 0/1 targets y, computes AUC with optional ROC curve"""
    if isinstance(y, pd.Series):
        y = y.values

    z = z.ravel()
    y = y.ravel()
    assert len(z) == len(y)

    # Remove any pair with NaN in y
    m = ~np.isnan(y)
    y = y[m]
    z = z[m]
    assert np.all(
        np.logical_or(y == 0, y == 1)
    ), "Cannot calculate AUC for non-binary targets"

    order = np.argsort(z, axis=0)[
        ::-1
    ].ravel()  # Sort by decreasing order of prediction strength
    z = z[order]
    y = y[order]
    npos = np.count_nonzero(y)  # Total number of positives.
    nneg = len(y) - npos  # Total number of negatives.
    if nneg == 0 or npos == 0:
        return (np.nan, None) if want_curve else 1

    n = len(y)
    fprate = np.zeros((n + 1, 1))
    tprate = np.zeros((n + 1, 1))
    ntpos, nfpos = 0.0, 0.0
    for i, yi in enumerate(y):
        if yi:
            ntpos += 1
        else:
            nfpos += 1
        tprate[i + 1] = ntpos / npos
        fprate[i + 1] = nfpos / nneg
    auc = float(np.trapz(tprate, fprate, axis=0))
    if want_curve:
        curve = np.hstack([fprate, tprate])
        return auc, curve
    return auc


def median_calc(preds, y):
    """Calculate the median of the predictions for the top half of the y values"""
    nan_mask = ~np.isnan(y)
    y = y[nan_mask]
    preds = preds[nan_mask]
    indeces_1 = np.where(preds == 1)[0]
    return np.median(y[indeces_1])


def auc_calc(preds, y):
    if isinstance(preds, pd.Series):
        preds = preds.values
    nan_mask = ~np.isnan(y)
    y = y[nan_mask]
    preds = preds[nan_mask]
    order = np.argsort(y)
    y_sorted = y[order]
    preds_sorted = preds[order]
    return auc(y_sorted, preds_sorted)


def escore(preds, y, use_calc_auc=True):
    if isinstance(preds, pd.Series):
        preds = preds.values
    nan_mask = ~np.isnan(y)
    y = y[nan_mask]
    if isinstance(preds, pd.Series):
        preds = preds.values
    preds = preds[nan_mask]
    l_0 = np.where(preds == 0)[0]
    l_1 = np.where(preds == 1)[0]
    y_0 = y[l_0]
    y_1 = y[l_1]
    indeces_y_0, indeces_y_1 = np.argsort(y_0)[::-1], np.argsort(y_1)[::-1]
    sorted_y_0, sorted_y_1 = np.sort(y_0)[::-1], np.sort(y_1)[::-1]
    indeces_y_0_top = indeces_y_0[: int(len(sorted_y_0) / 2)]
    indeces_y_1_top = indeces_y_1[: int(len(sorted_y_1) / 2)]
    sorted_y_0_top = sorted_y_0[: int(len(sorted_y_0) / 2)]
    sorted_y_1_top = sorted_y_1[: int(len(sorted_y_1) / 2)]
    l_0_top = l_0[indeces_y_0_top]
    l_1_top = l_1[indeces_y_1_top]
    l_top = np.concatenate([l_0_top, l_1_top])
    if use_calc_auc:
        return calc_auc(y[l_top], preds[l_top])
    else:
        return auc_calc(preds=preds[l_top], y=y[l_top])

metric_to_func_map = {
    "median": median_calc,
    "auc": calc_auc,
    "escore": escore
}


def word_scores(
    presence_mtx, 
    y, 
    num_kmers=None, 
    metric="zscore", 
    verbose=True
):
    df = pd.DataFrame(presence_mtx)
    df_sub = df[:num_kmers]
    func = metric_to_func_map[metric]
    if verbose:
        w_scores = df_sub.progress_apply(lambda preds: func(preds, y), axis=1)
    else:
        w_scores = df_sub.apply(lambda preds: func(preds, y), axis=1)
    if metric == "zscore":
        w_scores = (w_scores - np.mean(w_scores)/np.std(w_scores, ddof=1))
    return w_scores


def rnacomplete_scores(
    presence_mtx, 
    y, 
    num_kmers=None, 
    verbose=True
):
    df = pd.DataFrame()
    df["Median"] = word_scores(presence_mtx, y, num_kmers, metric="median", verbose=verbose)
    df["AUC"] = word_scores(presence_mtx, y, num_kmers, metric="auc", verbose=verbose)
    df["E-score"] = word_scores(presence_mtx, y, num_kmers, metric="escore", verbose=verbose)
    df["Z-score"] = (df["Median"] - np.mean(df["Median"]))/np.std(df["Median"], ddof=1)
    return df


def rna_complete_scores_sdata(
    sdata, 
    presence_mtx,
    prediction_keys,
    num_kmers=None, 
    verbose=True
):
    if isinstance(predictions, str):
        predictions = [predictions]
