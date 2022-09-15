import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from typing import Union, Sequence, Tuple, Dict, List
from sklearn.preprocessing import binarize
from ._utils import _check_input


default_rc_context = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}


def _plot_binary_confusion_mtx(
    sdata, target, prediction, threshold, title=None, **kwargs
) -> None:
    """
    Plot a confusion matrix for binary classification.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target : str
        Name of the target variable.
    prediction : str
        Name of the prediction variable.
    threshold : float
        Threshold for prediction.
    title : str
        Title of the plot.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    None
    """
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    cf_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    ts = sdata.seqs_annot[target].values.reshape(-1, 1)
    ps = binarize(
        sdata.seqs_annot[prediction].values.reshape(-1, 1), threshold=threshold
    )
    cf_mtx = confusion_matrix(ts, ps)
    cf_pcts = [
        "{0:.2%}".format(value)
        for value in (cf_mtx / cf_mtx.sum(axis=1)[:, None]).flatten()
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cf_mtx.flatten(), cf_pcts, cf_names)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(
        cf_mtx, annot=labels, fmt="s", cmap="Blues", cbar=False, ax=ax, **kwargs
    )
    ax.set_xlabel("Predicted Label", fontsize=20)
    ax.set_ylabel("True Label", fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.set_yticklabels(["Inactive", "Active"], fontsize=16)
    ax.set_xticklabels(
        [
            "Inactive (Score<{})".format(str(threshold)),
            "Active (Score>{})".format(str(threshold)),
        ],
        fontsize=16,
    )
    plt.tight_layout()


def confusion_mtx(
    sdata,
    target,
    prediction,
    kind="binary",
    threshold=0.5,
    rc_context=default_rc_context,
    **kwargs,
) -> None:
    """
    Plots a confusion matrix using seaborn.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target : str
        Key to use as target.
    prediction : str
        Key to use as prediction.
    kind : str
        Kind of confusion matrix.
    threshold : float
        Threshold for binary confusion matrix.
    rc_context : Mapping[str, str]
        Matplotlib rc context.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
        None
    """
    with plt.rc_context(rc_context):
        if kind == "binary":
            _plot_binary_confusion_mtx(sdata, target, prediction, threshold, **kwargs)


def _plot_curve(
    ax,
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    label: str = "",
    xlab: str = "",
    ylab: str = "",
):
    ax.plot(x, y, label=label)
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    ax.legend(loc="lower right")
    plt.tight_layout()


def auroc(
    sdata,
    targets: Union[Sequence[str], str],
    predictions: Union[Sequence[str], str],
    labels: Union[Sequence[str], str] = "",
    colors: Union[Sequence[str], str] = "bgrcmyk",
    **kwargs,
) -> None:
    targets, predictions, labels = _check_input(sdata, targets, predictions, labels)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for label, target, prediction in zip(labels, targets, predictions):
        ts = sdata.seqs_annot[target].values.reshape(-1, 1)
        ps = sdata.seqs_annot[prediction].values.reshape(-1, 1)
        fpr, tpr, _ = roc_curve(ts, ps)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.legend(loc="lower right")
        plt.tight_layout()


def auprc(
    sdata,
    targets: Union[Sequence[str], str],
    predictions: Union[Sequence[str], str],
    labels: Union[Sequence[str], str] = "",
    colors: Union[Sequence[str], str] = "bgrcmyk",
    save: bool = False,
    **kwargs,
) -> None:
    targets, predictions, labels = _check_input(sdata, targets, predictions, labels)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for label, target, prediction in zip(labels, targets, predictions):
        ts = sdata.seqs_annot[target].values.reshape(-1, 1)
        ps = sdata.seqs_annot[prediction].values.reshape(-1, 1)
        precision, recall, _ = precision_recall_curve(ts, ps)
        average_precision = average_precision_score(ts, ps)
        ax.plot(recall, precision, label=f"{label} (AP = {average_precision:.3f})")
        ax.set_xlabel("Recall", fontsize=20)
        ax.set_ylabel("Precision", fontsize=20)
        ax.legend(loc="lower right")
        plt.tight_layout()
    if save:
        fig.savefig(save, dpi=300)
