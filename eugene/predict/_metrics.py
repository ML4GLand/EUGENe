import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from sklearn.metrics import auc


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


def calc_auc(z, y, want_curve=False):
    # https://github.com/jisraeli/DeepBind/blob/master/code/deepfind.py
    """Given predictions z and 0/1 targets y, computes AUC with optional ROC curve"""
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
    nan_mask = ~np.isnan(y)
    y = y[nan_mask]
    preds = preds[nan_mask]
    indeces_1 = np.where(preds == 1)[0]
    return np.median(y[indeces_1])


def auc_calc(preds, y):
    nan_mask = ~np.isnan(y)
    y = y[nan_mask]
    preds = preds[nan_mask]
    order = np.argsort(y)
    y_sorted = y[order]
    preds_sorted = preds[order]
    return auc(y_sorted, preds_sorted)


def escore(preds, y, use_calc_auc=False):
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


def rna_complete_metrics(
    probes,
    intensities,
    n=7,
    alphabet="AGCU",
    num_kmers=None,
    use_calc_auc=False,
    verbose=True,
):
    possible_kmers = generate_all_possible_kmers(n=n, alphabet=alphabet)
    num_kmers = len(possible_kmers) if num_kmers is None else num_kmers
    kmers = possible_kmers[:num_kmers]
    kmer_medians, kmer_aucs, kmer_escores = [], [], []
    y = intensities
    for i, kmer in tqdm(
        enumerate(kmers), desc="Scoring k-mers", total=len(kmers), disable=not verbose
    ):
        preds = kmer_in_seqs(seqs=probes, kmer=kmer)
        kmer_medians.append(median_calc(preds=preds, y=y))
        if use_calc_auc:
            kmer_aucs.append(calc_auc(y, preds))
        else:
            kmer_aucs.append(auc_calc(preds=preds, y=y))
        kmer_escores.append(escore(preds=preds, y=y, use_calc_auc=use_calc_auc))
    kmer_medians, kmer_aucs, kmer_escore = (
        np.array(kmer_medians),
        np.array(kmer_aucs),
        np.array(kmer_escores),
    )
    kmer_zscores = (kmer_medians - np.mean(kmer_medians)) / np.std(kmer_medians, ddof=1)
    return kmer_zscores, kmer_aucs, kmer_escores


def evaluate_rbp(
    sdata, probe_id, n_kmers=100, return_cors=False, verbose=True, use_calc_auc=False
):
    seqs = sdata.seqs
    observed = sdata[probe_id].values
    preds = sdata[f"{probe_id}_predictions"].values

    # Get zscores, aucs and escores from observed intensities
    observed_zscores, observed_aucs, observed_escores = eu.predict.rna_complete_metrics(
        seqs, observed, num_kmers=n_kmers, verbose=verbose, use_calc_auc=use_calc_auc
    )

    # Get zscores, aucs, and escores from predicted intensities
    preds_zscores, preds_aucs, preds_escores = eu.predict.rna_complete_metrics(
        seqs, preds, num_kmers=n_kmers, verbose=verbose, use_calc_auc=use_calc_auc
    )

    # Z-scores
    zscore_pearson = pearsonr(preds_zscores, observed_zscores)[0]
    zscore_spearman = spearmanr(preds_zscores, observed_zscores).correlation

    # AUCs
    auc_pearson = pearsonr(preds_aucs, observed_aucs)[0]
    auc_spearman = spearmanr(preds_aucs, observed_aucs).correlation

    # E-scores
    escore_pearson = pearsonr(preds_escores, observed_escores)[0]
    escore_spearman = spearmanr(preds_escores, observed_escores).correlation

    # Intensities
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


def rna_complete_metrics_apply(
    kmer_presence_mtx, intensities, num_kmers=None, use_calc_auc=False, verbose=True
):
    df = pd.DataFrame(kmer_presence_mtx)
    y = intensities
    df_sub = df[:num_kmers]
    if verbose:
        if use_calc_auc:
            rbp_eval = df_sub.progress_apply(
                lambda preds: pd.Series(
                    {
                        "Median": median_calc(preds, y),
                        "AUC": calc_auc(y, preds),
                        "E-score": escore(preds, y, use_calc_auc=True),
                    }
                ),
                axis=1,
            )
        else:
            rbp_eval = df_sub.progress_apply(
                lambda preds: pd.Series(
                    {
                        "Median": median_calc(preds, y),
                        "AUC": auc_calc(preds, y),
                        "E-score": escore(preds, y),
                    }
                ),
                axis=1,
            )
    else:
        if use_calc_auc:
            rbp_eval = df_sub.apply(
                lambda preds: pd.Series(
                    {
                        "Median": median_calc(preds, y),
                        "AUC": calc_auc(y, preds),
                        "E-score": escore(preds, y, use_calc_auc=True),
                    }
                ),
                axis=1,
            )
        else:
            rbp_eval = df_sub.apply(
                lambda preds: pd.Series(
                    {
                        "Median": median_calc(preds, y),
                        "AUC": auc_calc(preds, y),
                        "E-score": escore(preds, y),
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


def column_rnac_metrics_apply(
    sdata,
    kmer_presence_mtx,
    probe_id,
    n_kmers=None,
    return_cors=False,
    use_calc_auc=False,
    verbose=True,
):
    observed = sdata[probe_id].values
    preds = sdata[f"{probe_id}_predictions"].values

    # Get zscores, aucs and escores from observed intensities
    observed_zscores, observed_aucs, observed_escores = rna_complete_metrics_apply(
        kmer_presence_mtx,
        observed,
        num_kmers=n_kmers,
        verbose=verbose,
        use_calc_auc=use_calc_auc,
    )

    # Get zscores, aucs, and escores from predicted intensities
    preds_zscores, preds_aucs, preds_escores = rna_complete_metrics_apply(
        kmer_presence_mtx,
        preds,
        num_kmers=n_kmers,
        verbose=verbose,
        use_calc_auc=use_calc_auc,
    )

    # Z-scores
    zscore_pearson = pearsonr(preds_zscores, observed_zscores)[0]
    zscore_spearman = spearmanr(preds_zscores, observed_zscores).correlation

    # AUCs
    auc_pearson = pearsonr(preds_aucs, observed_aucs)[0]
    auc_spearman = spearmanr(preds_aucs, observed_aucs).correlation

    # E-scores
    escore_pearson = pearsonr(preds_escores, observed_escores)[0]
    escore_spearman = spearmanr(preds_escores, observed_escores).correlation

    # Intensities
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


def summarize_rbps_apply(
    sdata, kmer_presence_mtx, probe_ids, n_kmers=100, verbose=False, use_calc_auc=False
):
    spearman_summary = pd.DataFrame()
    pearson_summary = pd.DataFrame()
    for i, probe_id in tqdm(
        enumerate(probe_ids), desc="Evaluating probes", total=len(probe_ids)
    ):
        rs, rhos = column_rnac_metrics_apply(
            sdata,
            kmer_presence_mtx,
            probe_id=probe_id,
            n_kmers=n_kmers,
            return_cors=True,
            verbose=verbose,
            use_calc_auc=use_calc_auc,
        )
        pearson_summary = pd.concat(
            [pearson_summary, pd.DataFrame(rs, index=[probe_id])], axis=0
        )
        spearman_summary = pd.concat(
            [spearman_summary, pd.DataFrame(rhos, index=[probe_id])], axis=0
        )
    return pearson_summary, spearman_summary
