import os
import glob
import seaborn as sns
from os import PathLike
from typing import Optional, Union, Sequence, Mapping
import matplotlib.pyplot as plt
from ._utils import (
    _plot_seaborn,
    many_logs2pandas,
    _save_fig,
)
from .. import settings


def metric_curve(
    log_path,
    metric,
    hue: Optional[str] = "metric",
    title: Optional[str] = None,
    xlab: Optional[str] = "minibatch step",
    ylab: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_axes: Optional[bool] = False,
    **kwargs
) -> None:
    """
    Plots the loss curves from a PyTorch Lightning (PL) training run.

    Uses the tensorboard event file to extract the loss curves. The loss curves are extracted from the event file and
    converted to a pandas dataframe. The dataframe is then plotted using seaborn.

    Parameters
    ----------
    log_path : str, optional
        Path to tensorboard log directory.
    metric : str, optional
        Metric to plot. Should be the string name of the metric used in PL
    title : str, optional
        Title of plot.
    xlab : str, optional
        Label for x-axis.
    ylab : str, optional
        Label for y-axis.
    ax : matplotlib.pyplot.Axes, optional
        The axes object to plot on.
    return_axes : bool, optional
        If True, returns the axes object.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    None or matplotlib.pyplot.Axes
        If return_axes is True, returns the axes object.
    """
    log_path = settings.logging_dir if log_path is None else log_path
    ylab = metric if ylab is None else ylab
    tb_event_path = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
    dataframe = many_logs2pandas(tb_event_path)
    dataframe = dataframe[dataframe["metric"].str.contains(metric)]
    dataframe = dataframe[~dataframe["metric"].str.contains("step")]
    ax = _plot_seaborn(
        dataframe,
        "value",
        sns.lineplot,
        groupby="step",
        orient="v",
        title=title,
        xlab=xlab,
        ylab=ylab,
        hue=hue,
        ax=ax,
        **kwargs
    )
    if return_axes:
        return ax


def loss_curve(
    log_path,
    title: Optional[str] = None,
    xlab: Optional[str] = "minibatch_step",
    ylab: Optional[str] = "loss",
    ax: Optional[plt.Axes] = None,
    return_axes: Optional[bool] = False,
    **kwargs
) -> None:
    """
    Plots the loss curves from a PyTorch Lightning (PL) training run. Wraps metrics_curve function.

    Uses the tensorboard event file to extract the metric curves. The metric curves are extracted from the event file and
    converted to a pandas dataframe. The dataframe is then plotted using seaborn.


    Parameters
    ----------
    log_path : str, optional
        Path to tensorboard log directory.
    title : str, optional
        Title of plot.
    xlab : str, optional
        Label for x-axis.
    ylab : str, optional
        Label for y-axis.
    ax : matplotlib.pyplot.Axes, optional
        The axes object to plot on.
    return_axes : bool, optional
        If True, returns the axes object.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    None or matplotlib.pyplot.Axes
        If return_axes is True, returns the axes object.
    """
    log_path = settings.logging_dir if log_path is None else log_path
    ax = metric_curve(
        log_path, metric="loss", title=title, xlab=xlab, ylab=ylab, ax=ax, **kwargs
    )
    if return_axes:
        return ax


def training_summary(
    log_path,
    metric: Optional[str] = "r2",
    figsize: Optional[tuple] = (12, 6),
    save: Optional[str] = None,
    return_axes: Optional[bool] = False,
    **kwargs
) -> None:
    """
    Plots the training summary from a given training run

    Convenience function to plot loss and metric together.

    Parameters
    ----------
    log_path : str, optional
        Path to tensorboard log directory.
    metrics : str or list of str, optional
        Metrics to plot. Should be the string name of the metric used in PL
    figsize : tuple, optional
        The size of the figure.
    save : str, optional
        The filename to save the figure to.
    return_axes : bool, optional
        If True, returns the axes object.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    None or matplotlib.pyplot.Axes
        If return_axes is True, returns the axes object.
    """
    log_path = settings.logging_dir if log_path is None else log_path
    _, ax = plt.subplots(1, 2, figsize=figsize)
    loss_curve(log_path, ax=ax[0], **kwargs)
    metric_curve(log_path, metric=metric, ax=ax[1], **kwargs)
    if save is not None:
        _save_fig(save)
    if return_axes:
        return ax