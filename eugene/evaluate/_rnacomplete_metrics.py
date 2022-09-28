import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List
tqdm.pandas()
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from ._metrics import median_calc, auc_calc, escore


def rnacomplete_metrics(
    kmer_presence_mtx: np.ndarray, 
    intensities: np.ndarray, 
    verbose: bool = True, 
    swifter: bool = False
):
    """
    Calculate the RNAcomplete metrics for a set of k-mers and scores for a set of sequences.

    Parameters
    ----------
    kmer_presence_mtx : np.ndarray
        A binary matrix of k-mers x samples. A 1 in entry (i, j) indicates that sequence j contains k-mer i. 
    intensities : np.ndarray
        A vector of scores for each sequence.
    """
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
            try:
                import swifter
            except ImportError:
                raise ImportError(
                    "swifter is not installed. Please install swifter to use this feature."
                )
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
    kmer_presence_mtx: np.ndarray,
    target_key: str,
    return_cors: bool = False,
    verbose: bool = True,
    preds_suffix: str = "_predictions",
    **kwargs
):
    """
    Calculate the RNAcomplete metrics for a set of k-mers and intensities in a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        A SeqData object containing the intensities for each sequence in seqs_annot.
    kmer_presence_mtx : np.ndarray
        A binary matrix of k-mers x samples. A 1 in entry (i, j) indicates that sequence j contains k-mer i.
    target_key : str
        The key in sdata.seqs_annot to use for the intensities.
    return_cors : bool, optional
        Whether to return the Pearson and Spearman correlations, by default False (plots only) 
    verbose : bool, optional
        Whether to show a progress bar for all the k-mers, by default True
    preds_suffix : str, optional
        The suffix to use for the predictions, by default "_predictions"
    """
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
    kmer_presence_mtx: np.ndarray,
    target_keys: List[str],
    num_kmers: int = 100,
    verbose: bool = False,
    preds_suffix: str = "_predictions",
    **kwargs
):
    """
    Generate a table of RNAcomplete metrics for a list of target keys.

    Parameters
    ----------
    sdata : SeqData
        SeqData object with observed and predicted scores in columns of seqs_annot
    kmer_presence_mtx : np.ndarray
       A binary matrix of k-mers x samples. A 1 in entry (i, j) indicates that sequence j contains k-mer i.
    target_keys : List[str]
        List of target keys to compute metrics for
    num_kmers : int, optional
        Number of k-mers to sample to compute metrics, by default 100. For large sets of k-mers this can take long
    verbose : bool, optional
        Whether to print progress, by default False
    preds_suffix : str, optional
        Suffix of predicted scores in seqs_annot, by default "_predictions"
    
    Returns
    -------
    pd.DataFrame
        A table of RNAcomplete metrics for each target key
    """
    if isinstance(target_keys, str):
        target_keys = [target_keys]
    spearman_summary = pd.DataFrame()
    pearson_summary = pd.DataFrame()
    if num_kmers is not None:
        random_kmers = np.random.choice(np.arange(kmer_presence_mtx.shape[0]), size=num_kmers)
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
