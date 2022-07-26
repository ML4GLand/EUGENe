import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Sequence
from sklearn.preprocessing import binarize
from ._utils import _create_matplotlib_axes, _label_plot, _plot_seaborn, tflog2pandas, many_logs2pandas


def metric_curve(
    log_path,
    metric: str = None,
    hue: str = "metric",
    title: str = None,
    xlab: str = "minibatch step",
    ylab: str = "loss",
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots the loss curves from a training run

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.
    keys : str or list of str
        Keys to plot.
    title : str
        Title of plot.
    xlab : str
        Label for x-axis.
    ylab : str
        Label for y-axis.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    None
    """
    tb_event_path = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
    dataframe = many_logs2pandas(tb_event_path)
    dataframe = dataframe[dataframe["metric"].str.contains(metric)]
    ax = _plot_seaborn(dataframe, "value", sns.lineplot, groupby="step", orient='v', title=title, xlab=xlab, ylab=ylab, hue=hue, **kwargs)
    if return_axes:
        return ax


def loss_curve(
    log_path,
    title: str = None,
    xlab: str = None,
    ylab: str = "loss",
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots the loss curves from a training run

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.
    title : str
        Title of plot.
    xlab : str
        Label for x-axis.
    ylab : str
        Label for y-axis.
    return_axes : bool
        If True, return the axes object.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    None
    """
    ax = metric_curve(log_path, metric="loss", title=title, xlab=xlab, ylab=ylab, **kwargs)
    if return_axes:
        return ax


def training_summary(
    log_path: str,
    metrics: Union[str, Sequence[str]] = None,
) -> None:
    """
    Plots the training summary from a training run

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.

    Returns
    -------
    None
    """
    loss_curve(log_path, return_axes=True)
    metric_curve(log_path, metric=metrics, return_axes=True)
    return None


def performance_scatter(sdata, x, y, seq_idx=None, **kwargs):

    # Get the indices of the sequences in the subset
    if seq_idx is not None:
        sdata = sdata[seq_idx]

    _plot_performance_scatter(sdata, target=target, prediction=prediction, **kwargs)


def _plot_performance_scatter(sdata, x, y, **kwargs):
    """
    Plot a scatter plot of the performance of the model on a subset of the sequences.
    """
    # Get the predictions and targets
    targets = sdata.seqs_annot[x]
    predictions = sdata.seqs_annot[y]

    # Plot the scatter plot
    plt.scatter(targets, predictions, **kwargs)
    plt.xlabel("TARGETS")
    plt.ylabel("PREDICTIONS")
    plt.title("Performance of Model on Subset")


def confusion_mtx(sdata, **kwargs):
    _plot_confusion_mtx(sdata, **kwargs)


def _plot_confusion_mtx(sdata, target="TARGETS", prediction="PREDICTIONS", title="Sequences", xlab="Predicted Activity", ylab="True Activity", threshold=0.5):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        cf_names = ["True Neg","False Pos", "False Neg","True Pos"]
        ts = binarize(sdata.seqs_annot[target].values.reshape(-1, 1), threshold=threshold)
        ps = binarize(sdata.seqs_annot[prediction].values.reshape(-1, 1), threshold=threshold)
        cf_mtx = confusion_matrix(ts, ps)
        cf_pcts = ["{0:.2%}".format(value) for value in (cf_mtx/cf_mtx.sum(axis=1)[:,None]).flatten()]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(cf_mtx.flatten(),cf_pcts, cf_names)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cf_mtx, annot=labels, fmt='s', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.set_title(title, fontsize=24)
        ax.set_yticklabels(["Inactive", "Active"], fontsize=16)
        ax.set_xticklabels(["Inactive (Score<{})".format(str(threshold)), "Active (Score>{})".format(str(threshold))], fontsize=16)
        plt.tight_layout()