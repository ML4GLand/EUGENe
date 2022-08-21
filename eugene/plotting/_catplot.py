import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Collection, Iterable  # Acol_indexB
from typing import Tuple, List  # Classes
from ._utils import _create_matplotlib_axes, _label_plot, _plot_seaborn


default_rc_context = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def countplot(
    sdata,
    keys: Union[str, Sequence[str]],
    groupby: str = None,
    orient: str = "h",
    rc_context: Mapping[str, str] = default_rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a countplot using seaborn. This function was written by Copilot

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    keys : str or list of str
        Keys to plot.
    groupby : str
        Key to group by.
    orient : str
        Orientation of plot.
    rc_context : Mapping[str, str]
        Matplotlib rc context.
    return_axes : bool
        Return axes.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
        None
    """
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            sdata.seqs_annot,
            keys,
            func=sns.countplot,
            groupby=groupby,
            orient=orient,
            **kwargs
        )
    if return_axes:
        return ax


def histplot(
    sdata,
    keys: Union[str, Sequence[str]],
    orient: str = "v",
    rc_context: Mapping[str, str] = default_rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a histogram using seaborn.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    keys : str or list of str
        Keys to plot.
    groupby : str
        Key to group by.
    orient : str
        Orientation of plot.
    rc_context : Mapping[str, str]
        Matplotlib rc context.
    return_axes : bool
        Return axes.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
        None
    """
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            sdata.seqs_annot,
            keys,
            func=sns.histplot,
            orient=orient,
            ylab="Frequency",
            **kwargs
        )
    if return_axes:
        return ax


def boxplot(
    sdata,
    keys: Union[str, Sequence[str]],
    groupby: str = None,
    orient: str = "v",
    jitter=False,
    rc_context: Mapping[str, str] = default_rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a boxplot using seaborn.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    keys : str or list of str
        Keys to plot.
    groupby : str
        Key to group by.
    orient : str
        Orientation of plot.
    rc_context : Mapping[str, str]
        Matplotlib rc context.
    return_axes : bool
        Return axes.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
        None
    """
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            sdata.seqs_annot,
            keys,
            func=sns.boxplot,
            groupby=groupby,
            orient=orient,
            **kwargs
        )
        if jitter == True:
            _plot_seaborn(
                sdata.seqs_annot,
                keys,
                func=sns.stripplot,
                groupby=groupby,
                orient=orient,
                ax=ax,
                **kwargs
            )
    if return_axes:
        return ax


def violinplot(
    sdata,
    keys: Union[str, Sequence[str]],
    groupby: str = None,
    orient: str = "v",
    rc_context: Mapping[str, str] = default_rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
     Plots a violinplot using seaborn.

    Parameters
     ----------
     sdata : SeqData
         SeqData object.
     keys : str or list of str
         Keys to plot.
     groupby : str
         Key to group by.
     orient : str
         Orientation of plot.
     rc_context : Mapping[str, str]
         Matplotlib rc context.
     return_axes : bool
         Return axes.
     **kwargs
         Additional keyword arguments to pass to seaborn.

     Returns
     -------
         None
    """
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            sdata.seqs_annot,
            keys,
            func=sns.violinplot,
            groupby=groupby,
            orient=orient,
            **kwargs
        )
    if return_axes:
        return ax


def scatterplot(sdata, x: str, y: str, seq_idx=None, **kwargs):
    """
    Plots a scatterplot using seaborn.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    x : str
        Key for x-axis.
    y : str
        Key for y-axis.
    seq_idx : int or list of int
        Index of sequences to plot.
    **kwargs

    Returns
    -------
    None
    """
    if seq_idx is not None:
        sdata = sdata[seq_idx]
    _plot_seaborn(sdata.seqs_annot, x, sns.scatterplot, y, ylab=y, **kwargs)
