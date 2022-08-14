import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Union, Sequence
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _create_matplotlib_axes(num_axes, subplot_size=(4, 4)) -> List[Axes]:
    """
    Creates a grid of matplotlib axes.

    Parameters
    ----------
    num_axes : int
        Number of axes to create.
    subplot_size : tuple of ints
        Size of each subplot.

    Returns
    -------
    list of matplotlib.axes.Axes
        List of axes.
    """
    num_rows = int(np.ceil(num_axes / 3))
    num_cols = int(np.ceil(num_axes / num_rows))
    _, ax = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * subplot_size[0], num_rows * subplot_size[1]),
    )
    ax = ax.flatten() if num_axes > 1 else ax
    return ax


def _label_plot(ax: Axes, title: str, xlab: str, ylab: str, xtick_rot: int = 0) -> None:
    """
    Labels a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to label.
    title : str
        Title of plot.
    xlab : str
        Label for x-axis.
    ylab : str
        Label for y-axis.

    Returns
    -------
    None
    """
    ax.set_xlabel(xlab)
    if xtick_rot != 0:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rot)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    plt.tight_layout()


def _plot_seaborn(
    dataframe: pd.DataFrame,
    keys: Union[str, Sequence[str]],
    func,
    groupby: str = None,
    orient: str = "v",
    title: str = None,
    xlab: str = None,
    xtick_rot: int = 0,
    ylab: str = None,
    figsize: tuple = (10, 5),
    save: str = None,
    ax = None,
    **kwargs,
) -> Axes:
    """
    Plots a histogram, boxplot, violin plot or scatterplot using seaborn.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to plot.
    keys : str or list of str
        Keys to plot.
    func : function
        Function to use for plotting.
    groupby : str
        Key to group by.
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
    matplotlib.axes.Axes
    """
    keys = [keys] if isinstance(keys, str) else keys
    num_axes = len(keys)
    if ax is None:
        ax = _create_matplotlib_axes(num_axes, subplot_size=figsize)
    for i, key in enumerate(keys):
        curr_ax = ax[i] if len(keys) > 1 else ax
        if groupby is None:
            if orient == "v":
                func(data=dataframe, y=key, ax=curr_ax, **kwargs)
            elif orient == "h":
                func(data=dataframe, x=key, ax=curr_ax, **kwargs)
        else:
            if orient == "v":
                func(data=dataframe, x=groupby, y=key, ax=curr_ax, **kwargs)
            elif orient == "h":
                func(data=dataframe, x=key, y=groupby, ax=curr_ax, **kwargs)
        _label_plot(
            curr_ax,
            title,
            xlab=key if xlab is None else xlab,
            xtick_rot=xtick_rot,
            ylab=ylab,
        )
    if save is not None:
        if "/" not in save:
            save = os.path.join(os.getcwd(), save)
        dir = "/".join(save.split("/")[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save)
    return ax


def _check_input(
    sdata,
    targets: Union[Sequence[str], str],
    predictions: Union[Sequence[str], str],
    labels: Union[Sequence[str], str],
):
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(labels, str):
        labels = [labels]
    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must be equal.")
    if len(targets) != len(labels):
        labels = [f"LABEL_{i}" for i in range(len(targets))]
    return targets, predictions, labels


# Extraction function modified from https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
def tflog2pandas(path: str) -> pd.DataFrame:
    """Convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data.reset_index(drop=True)


def many_logs2pandas(event_paths):
    """Convert many tensorflow log files to pandas DataFrame

    Parameters
    ----------
    event_paths : list of str
        paths to tensorflow log files

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs
