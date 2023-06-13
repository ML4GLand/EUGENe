import numpy as np
import seaborn as sns
from .. import settings
from os import PathLike
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from typing import Union, Sequence
from sklearn.preprocessing import binarize
from ._utils import _check_input, _label_plot, _save_fig


def _plot_binary_confusion_mtx(
    sdata,
    target_key: str,
    prediction_key: str,
    threshold: float,
    title: str = None,
    xlab: str = None,
    ylab: str = None,
    figsize: tuple = (6, 6),
    save: PathLike = None,
    ax=None,
    **kwargs,
) -> None:
    """
    Plot a confusion matrix for binary classification.

    This function plots a binary confusion matrix as a seaborn heatmap.
    Pulls a target and prediction key from seqs_annot and uses the passed in
    threshold to binarize the prediction. The confusion matrix is then plotted
    using seaborn.

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
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    cf_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    ts = sdata[target_key].values.reshape(-1, 1)
    ps = binarize(sdata[prediction_key].values.reshape(-1, 1), threshold=threshold)
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
    _label_plot(
        ax,
        title,
        xlab=xlab,
        ylab=ylab,
    )
    plt.tight_layout()
    if save:
        _save_fig(save)
    return ax


def confusion_mtx(
    sdata,
    target_key: str,
    prediction_key: str,
    kind: str = "binary",
    threshold: float = 0.5,
    rc_context: dict = settings.rc_context,
    return_axes: bool = False,
    **kwargs,
) -> None:
    """
    Plot a confusion matrix for given targets and predictions within SeqData

    Creates a confusion matrix from the given target and prediction keys held
    in the seqs_annot of the passed in SeqData. The

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_key : str
        Key to use as target.
    prediction_key : str
        Key to use as prediction.
    kind : str
        Kind of confusion matrix to plot. Currently only allow for binary
    threshold : float
        Threshold to use to generate a binary confusion matrix.
    rc_context : Mapping[str, str]
        Matplotlib rc context. Defaults to eugene.settings.rc_context.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
        None
    """
    with plt.rc_context(rc_context):
        if kind == "binary":
            ax = _plot_binary_confusion_mtx(
                sdata, target_key, prediction_key, threshold, **kwargs
            )
        else:
            raise ValueError(
                f"Confusion matrix for '{kind}' classification not currently supported."
            )
    if return_axes:
        return ax


def auroc(
    sdata,
    target_keys: Union[Sequence[str], str],
    prediction_keys: Union[Sequence[str], str],
    labels: Union[Sequence[str], str] = "",
    xlab: str = "False Positive Rate",
    ylab: str = "True Positive Rate",
    figsize: tuple = (8, 8),
    save: str = None,
    ax=None,
    **kwargs,
) -> None:
    """
    Plot the area under the receiver operating characteristic curve for one or more predictions against
    a one or more targets.


    You must pass in the same number of target keys as prediction keys. If you want to compare the same target
    against multiple predictions, pass in the same target key for each predictions key.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_keys : Union[Sequence[str], str]
        Target keys to use for plotting.
    prediction_keys : Union[Sequence[str], str]
        Prediction keys to use for plotting.
    labels : Union[Sequence[str], str]
        Labels to use for each prediction. If not passed in, the labels_{i} will be used.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    figsize : tuple
        Size of the figure.
    save : str
        Path to save the figure. If none, figure will not be saved.
    **kwargs
        Additional keyword arguments to pass to matplotlib plot function
    """
    target_keys, prediction_keys, labels = _check_input(
        sdata, target_keys, prediction_keys, labels
    )
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    for label, target_key, prediction_key in zip(labels, target_keys, prediction_keys):
        ts = sdata[target_key].values.reshape(-1, 1)
        ps = sdata[prediction_key].values.reshape(-1, 1)
        fpr, tpr, _ = roc_curve(ts, ps)
        roc_auc = auc(fpr, tpr)
        print(roc_auc, label)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", **kwargs)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.legend(loc="lower right")
        plt.tight_layout()
    if save:
        _save_fig(save)
    return ax


def auprc(
    sdata,
    target_keys: Union[Sequence[str], str],
    prediction_keys: Union[Sequence[str], str],
    labels: Union[Sequence[str], str] = "",
    xlab: str = "Recall",
    ylab: str = "Precision",
    figsize: tuple = (8, 8),
    save: str = None,
    ax=None,
    **kwargs,
) -> None:
    """
    Plot the area under the precision recall curve for one or more predictions against
    a one or more targets.


    You must pass in the same number of target keys as prediction keys. If you want to compare the same target
    against multiple predictions, pass in the same target key for each predictions key.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_keys : Union[Sequence[str], str]
        Target keys to use for plotting.
    prediction_keys : Union[Sequence[str], str]
        Prediction keys to use for plotting.
    labels : Union[Sequence[str], str]
        Labels to use for each prediction. If not passed in, the labels_{i} will be used.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    figsize : tuple
        Size of the figure.
    save : str
        Path to save the figure. If none, figure will not be saved.
    **kwargs
        Additional keyword arguments to pass to matplotlib plot function
    """
    target_keys, prediction_keys, labels = _check_input(
        sdata, target_keys, prediction_keys, labels
    )
    _, ax = plt.subplots(1, 1, figsize=figsize)
    for label, target_key, prediction_key in zip(labels, target_keys, prediction_keys):
        ts = sdata[target_key].values.reshape(-1, 1)
        ps = sdata[prediction_key].values.reshape(-1, 1)
        precision, recall, _ = precision_recall_curve(ts, ps)
        average_precision = average_precision_score(ts, ps)
        ax.plot(
            recall, precision, label=f"{label} (AP = {average_precision:.3f})", **kwargs
        )
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.legend(loc="lower right")
        plt.tight_layout()
    if save:
        _save_fig(save)
    return ax
