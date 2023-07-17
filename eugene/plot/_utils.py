import os
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
from .. import settings
from os import PathLike
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Union, Sequence
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _create_matplotlib_axes(num_axes, subplot_size=(4, 4)) -> List[Axes]:
    """
    Creates and returns a list of matplotlib axes.

    Uses at most 3 columns before breaking into a new row.
    By default each subplot is 4x4.

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
    Labels a passed in axes with a title, x-axis label, and y-axis label.

    Optinally rotate the x-axis if you have long labels.

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


def _save_fig(file_path: PathLike, dpi=settings.dpi):
    """
    Save a figure to a file path.

    Creates the filepath if it doesn't exist.
    Uses the dpi specified in package settings.
    """
    if "/" not in file_path:
        file_path = os.path.join(os.getcwd(), file_path)
    dir = "/".join(file_path.split("/")[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(file_path, dpi=dpi)


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
    ax: Axes = None,
    **kwargs,
) -> Axes:
    """
    Plots a histogram, boxplot, violin plot or scatterplot using seaborn.

    Uses the keys in a dataframe to create multiple subplots, one for each key.
    If a groupby key is passed in, then the subplots will be grouped by that key
    Creates an axes using _create_matplotlib_axes if ax is None and saves the
    figure if save is not None. Otherwise, the axes is returned.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to use for plotting.
    keys : str or list of str
        Keys in the dataframe columns to plot.
    func : function
        Function to use for plotting. Must be a seaborn plotting function that takes in x and y
    groupby : str
        Key to group by. If None, then no grouping is done.
    title : str
        Title of plot.
    xlab : str
        Label for x-axis.
    ylab : str
        Label for y-axis.
    figsize : tuple of ints
        Size of figure to plot. By default, (10, 5).
    save : str
        Path to save figure. If None, then figure is not saved.
        Note that this will create a directory if it does not exist.
    **kwargs
        Additional keyword arguments to pass to seaborn. This will be
        dependent on func

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
    # plt.show()
    if save is not None:
        _save_fig(save)
    return ax


def _violin_long(
    dataframe,
    groups: Union[str, Sequence[str]],
    title: str = None,
    xlab: str = "variable",
    xtick_rot: int = 0,
    ylab: str = "value",
    figsize: tuple = (8, 8),
    save: PathLike = None,
    ax: Axes = None,
    **kwargs,
):
    """
     Plots a violinplot using seaborn on an SeqData object.
     The difference between this and the _plot_seaborn function is that
     this function takes a list of groups and plots them on the same
     plot (i.e. it takes in multiple columns and turns them into a single one
     in long format. Then plots these as different groups)

    Parameters
     ----------
     sdata : SeqData
         SeqData object.
     groups : str or list of str
         Groups to plot.
     xlabel : str
         Label for x-axis.
     ylabel : str
         Label for y-axis.
     figsize : tuple
         Figure size.
     save : str
         Filepath to save figure to.
     **kwargs

     Returns
     -------
     None
    """

    groups = [groups] if isinstance(groups, str) else groups
    long = dataframe.melt(value_vars=groups)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    sns.violinplot(data=long, x="variable", y="value", ax=ax, **kwargs)
    sns.stripplot(data=long, x="variable", y="value", ax=ax, color="black", alpha=0.75)
    _label_plot(
        ax,
        title,
        xlab=xlab,
        xtick_rot=xtick_rot,
        ylab=ylab,
    )
    if save:
        _save_fig(save)
    return ax


def _check_input(
    sdata,
    targets: Union[Sequence[str], str],
    predictions: Union[Sequence[str], str],
    labels: Union[Sequence[str], str],
):
    """
    Helper function to check input for plotting functions.

    Makes sure that the targets, predictions, and labels are all the same length.
    """
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(labels, str):
        labels = [labels]
    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must be equal.")
    if len(targets) != len(labels):
        labels = [f"label_{i}" for i in range(len(targets))]
    return targets, predictions, labels


def tflog2pandas(path: str) -> pd.DataFrame:
    """
    Convert single tensorflow log file to pandas DataFrame

    Takes in the filepath to a tensorflow log file and converts it to a pandas
    DataFrame. The index of the DataFrame is the step number.

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe

    Note
    ----
    Extraction function modified from
    https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
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
    """
    Convert many tensorflow log files to pandas DataFrame.

    Wraps around tflog2pandas to convert many tensorflow log files to a single

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


def _const_line(*args, **kwargs):
    """
    Plots a constant line on current axis
    """
    x = np.arange(-1, 1.01, 0.01)
    plt.plot(x, x, c="k", ls="--")

def _collapse_pos(positions):
    """Collapse neighbor positions of array to ranges"""
    ranges = []
    start = positions[0]
    for i in range(1, len(positions)):
        if positions[i - 1] == positions[i] - 1:
            continue
        else:
            ranges.append((start, positions[i - 1] + 2))
            start = positions[i]
    ranges.append((start, positions[-1] + 2))
    return ranges
