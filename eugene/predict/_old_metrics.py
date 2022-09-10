def looped_rna_complete_metrics(
    probes,
    intensities,
    n=7,
    alphabet="AGCU",
    num_kmers=None,
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
        kmer_aucs.append(auc_calc(preds=preds, y=y))
        kmer_escores.append(escore(preds=preds, y=y))
    kmer_medians, kmer_aucs, kmer_escore = (
        np.array(kmer_medians),
        np.array(kmer_aucs),
        np.array(kmer_escores),
    )
    kmer_zscores = (kmer_medians - np.mean(kmer_medians)) / np.std(kmer_medians, ddof=1)
    return kmer_zscores, kmer_aucs, kmer_escores


def looped_evaluate_rbp(
    sdata,
    target_key,
    n_kmers=100,
    return_cors=False,
    verbose=True,
):
    seqs = sdata.seqs
    observed = sdata[target_key].values
    preds = sdata[f"{target_key}_predictions"].values

    # Get zscores, aucs and escores from observed intensities
    observed_zscores, observed_aucs, observed_escores = looped_rna_complete_metrics(
        seqs, observed, num_kmers=n_kmers, verbose=verbose
    )

    # Get zscores, aucs, and escores from predicted intensities
    preds_zscores, preds_aucs, preds_escores = looped_rna_complete_metrics(
        seqs, preds, num_kmers=n_kmers, verbose=verbose
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
