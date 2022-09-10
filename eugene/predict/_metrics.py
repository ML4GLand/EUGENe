#import swifter
import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm

tqdm.pandas()
from sklearn.metrics import auc, roc_auc_score
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


def median_calc(y_true, y_score):
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    nan_mask = ~np.isnan(y_score)
    y_score = y_score[nan_mask]
    y_true = y_true[nan_mask]
    indeces_1 = np.where(y_true == 1)[0]
    return np.median(y_score[indeces_1])


def auc_calc(y_true, y_score):
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    nan_mask = ~np.isnan(y_score)
    y_true = y_true[nan_mask]
    y_score = y_score[nan_mask]
    return roc_auc_score(y_true, y_score)


def escore(y_true, y_score):
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    nan_mask = ~np.isnan(y_score)
    y_score = y_score[nan_mask]
    y_true = y_true[nan_mask]
    l_0 = np.where(y_true == 0)[0]
    l_1 = np.where(y_true == 1)[0]
    y_0 = y_score[l_0]
    y_1 = y_score[l_1]
    indeces_y_0, indeces_y_1 = np.argsort(y_0)[::-1], np.argsort(y_1)[::-1]
    sorted_y_0, sorted_y_1 = np.sort(y_0)[::-1], np.sort(y_1)[::-1]
    indeces_y_0_top = indeces_y_0[: int(len(sorted_y_0) / 2)]
    indeces_y_1_top = indeces_y_1[: int(len(sorted_y_1) / 2)]
    sorted_y_0_top = sorted_y_0[: int(len(sorted_y_0) / 2)]
    sorted_y_1_top = sorted_y_1[: int(len(sorted_y_1) / 2)]
    l_0_top = l_0[indeces_y_0_top]
    l_1_top = l_1[indeces_y_1_top]
    l_top = np.concatenate([l_0_top, l_1_top])
    return auc_calc(y_true[l_top], y_score[l_top])


def rnacomplete_metrics(kmer_presence_mtx, intensities, verbose=True, swifter=False):
    y_score = intensities
    df = pd.DataFrame(kmer_presence_mtx).astype(np.int8)
    if verbose:
        if not swifter:
            rbp_eval = df.progress_apply(
                lambda y_true: pd.Series(
                    {
                        "Median": median_calc(y_true, y_score),
                        "AUC": auc_calc(y_true, y_score),
                        "E-score": escore(y_true, y_score),
                    }
                ),
                axis=1,
            )
        else:
            rbp_eval = df.swifter.apply(
                lambda y_true: pd.Series(
                    {
                        "Median": median_calc(y_true, y_score),
                        "AUC": auc_calc(y_true, y_score),
                        "E-score": escore(y_true, y_score),
                    }
                ),
                axis=1,
            )

    else:
        rbp_eval = df.apply(
            lambda y_true: pd.Series(
                {
                    "Median": median_calc(y_true, y_score),
                    "AUC": auc_calc(y_true, y_score),
                    "E-score": escore(y_true, y_score),
                }
            ),
            axis=1,
        )
    rbp_eval["Z-score"] = (rbp_eval["Median"] - np.mean(rbp_eval["Median"])) / np.std(
        rbp_eval["Median"], ddof=1
    )
    return (
        rbp_eval["Z-score"].values,
        rbp_eval["AUC"].values,
        rbp_eval["E-score"].values,
    )


def rnacomplete_metrics_sdata_plot(
    sdata,
    kmer_presence_mtx,
    target_key,
    return_cors=False,
    verbose=True,
    preds_suffix="_predictions",
    **kwargs,
):

    observed = sdata[target_key].values
    preds = sdata[f"{target_key}{preds_suffix}"].values

    # Get zscores, aucs and escores from observed intensities
    observed_zscores, observed_aucs, observed_escores = rnacomplete_metrics(
        kmer_presence_mtx, observed, verbose=verbose, **kwargs
    )

    # Get zscores, aucs, and escores from predicted intensities
    preds_zscores, preds_aucs, preds_escores = rnacomplete_metrics(
        kmer_presence_mtx, preds, verbose=verbose, **kwargs
    )
    # Z-scores
    zscore_nan_mask = np.isnan(observed_zscores) | np.isnan(preds_zscores)
    preds_zscores = preds_zscores[~zscore_nan_mask]
    observed_zscores = observed_zscores[~zscore_nan_mask]
    if len(observed_zscores) > 0 and len(preds_zscores) > 0:
        zscore_pearson = pearsonr(preds_zscores, observed_zscores)[0]
        zscore_spearman = spearmanr(preds_zscores, observed_zscores).correlation
    else:
        zscore_pearson = np.nan
        zscore_spearman = np.nan

    # AUCs
    auc_nan_mask = np.isnan(observed_aucs) | np.isnan(preds_aucs)
    preds_aucs = preds_aucs[~auc_nan_mask]
    observed_aucs = observed_aucs[~auc_nan_mask]
    auc_pearson = pearsonr(preds_aucs, observed_aucs)[0]
    auc_spearman = spearmanr(preds_aucs, observed_aucs).correlation

    # E-scores
    escore_nan_mask = np.isnan(observed_escores) | np.isnan(preds_escores)
    preds_escores = preds_escores[~escore_nan_mask]
    observed_escores = observed_escores[~escore_nan_mask]
    escore_pearson = pearsonr(preds_escores, observed_escores)[0]
    escore_spearman = spearmanr(preds_escores, observed_escores).correlation

    # Intensities
    intensity_nan_mask = np.isnan(observed) | np.isnan(preds)
    preds = preds[~intensity_nan_mask]
    observed = observed[~intensity_nan_mask]
    intensity_pearson = pearsonr(observed, preds)[0]
    intensity_spearman = spearmanr(observed, preds).correlation

    if return_cors:
        pearson = {
            "Z-score": zscore_pearson,
            "AUC": auc_pearson,
            "E-score": escore_pearson,
            "Intensity": intensity_pearson,
        }
        spearman = {
            "Z-score": zscore_spearman,
            "AUC": auc_spearman,
            "E-score": escore_spearman,
            "Intensity": intensity_spearman,
        }
        return pearson, spearman

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].scatter(observed_zscores, preds_zscores)
    ax[0].set_title("Z-scores")
    ax[0].set_xlabel("Observed")
    ax[0].set_ylabel("Predicted")
    ax[0].text(
        0.75,
        0.05,
        "r="
        + str(round(zscore_pearson, 2))
        + "\nrho="
        + str(round(zscore_spearman, 2)),
        transform=ax[0].transAxes,
    )

    ax[1].scatter(observed_aucs, preds_aucs)
    ax[1].set_title("AUCs")
    ax[1].set_xlabel("Observed")
    ax[1].set_ylabel("Predicted")
    ax[1].text(
        0.75,
        0.05,
        "r=" + str(round(auc_pearson, 2)) + "\nrho=" + str(round(auc_spearman, 2)),
        transform=ax[1].transAxes,
    )

    ax[2].scatter(observed_escores, preds_escores)
    ax[2].set_title("E-scores")
    ax[2].set_xlabel("Observed")
    ax[2].set_ylabel("Predicted")
    ax[2].text(
        0.75,
        0.05,
        "r="
        + str(round(escore_pearson, 2))
        + "\nrho="
        + str(round(escore_spearman, 2)),
        transform=ax[2].transAxes,
    )

    ax[3].scatter(observed, preds)
    ax[3].set_title("Intensities")
    ax[3].set_xlabel("Observed")
    ax[3].set_ylabel("Predicted")
    ax[3].text(
        0.75,
        0.05,
        "r="
        + str(round(intensity_pearson, 2))
        + "\nrho="
        + str(round(intensity_spearman, 2)),
        transform=ax[3].transAxes,
    )

    plt.tight_layout()


def rnacomplete_metrics_sdata_table(
    sdata,
    kmer_presence_mtx,
    target_keys,
    num_kmers=100,
    verbose=False,
    preds_suffix="_predictions",
    **kwargs,
):
    if isinstance(target_keys, str):
        target_keys = [target_keys]
    spearman_summary = pd.DataFrame()
    pearson_summary = pd.DataFrame()
    if num_kmers is not None:
        random_kmers = np.random.choice(
            np.arange(kmer_presence_mtx.shape[0]), size=num_kmers
        )
        kmer_presence_mtx = kmer_presence_mtx[random_kmers, :]
    valid_kmers = np.where(np.sum(kmer_presence_mtx, axis=1) > 155)[0]
    kmer_presence_mtx = kmer_presence_mtx[valid_kmers, :]
    for i, target_key in tqdm(
        enumerate(target_keys), desc="Evaluating probes", total=len(target_keys)
    ):
        rs, rhos = rnacomplete_metrics_sdata_plot(
            sdata,
            kmer_presence_mtx,
            target_key=target_key,
            return_cors=True,
            verbose=verbose,
            preds_suffix=preds_suffix,
            **kwargs,
        )
        pearson_summary = pd.concat(
            [pearson_summary, pd.DataFrame(rs, index=[target_key])], axis=0
        )
        spearman_summary = pd.concat(
            [spearman_summary, pd.DataFrame(rhos, index=[target_key])], axis=0
        )
    return pearson_summary, spearman_summary
