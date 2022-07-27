import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Sequence
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
    Plots the loss curves from a PyTorch Lightning (PL) training run

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.
    metric : str
        Metric to plot. Should be the string name of the metric used in PL
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
    If return_axes is True, returns the axes object.
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
    Plots the loss curves from a PyTorch Lightning (PL) training run. Wraps metrics_curve().

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
    If return_axes is True, returns the axes object.
    """
    ax = metric_curve(log_path, metric="loss", title=title, xlab=xlab, ylab=ylab, **kwargs)
    if return_axes:
        return ax


def training_summary(
    log_path: str,
    metrics: Union[str, Sequence[str]] = None,
) -> None:
    """
    Plots the training summary from a training run. Convenience function to plot loss and metric together.

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.
    metrics : str or list of str
        Metrics to plot. Should be the string name of the metric used in PL
        
    Returns
    -------
    None
    """
    loss_curve(log_path, return_axes=True)
    metric_curve(log_path, metric=metrics, return_axes=True)
    return None
