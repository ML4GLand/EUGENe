import seaborn as sns
from .. import settings
import matplotlib.pyplot as plt
from typing import Union, Mapping
from typing import Sequence, Iterable
from ._utils import _plot_seaborn, _violin_long


def countplot(
    sdata,
    keys: Union[str, Sequence[str]],
    groupby: str = None,
    orient: str = "h",
    rc_context: Mapping[str, str] = settings.rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a countplot of a column(s) in seqs_annot using Seaborn.

    This function can be used to show the counts of observations in a single
    or multiple columns of seqs_annot within a SeqData. If a groupby is
    provided then the counts are grouped by the groupby column.

    Parameters
    ----------
    sdata : SeqData
        SeqData object that contains keys in seqs_annot.
    keys : str or list of str
        Keys to plot. Will be plotted in separate adjacent subplots.
    groupby : str
        Key to group by. If None, will plot counts of each key.
    orient : str
        Orientation of plot. Either "h" (horizontal) or "v" (vertical).
    rc_context : Mapping[str, str]
        Matplotlib rc context. Default is eugene.settings.rc_context.
    return_axes : bool
        Return axes.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
        None
    """
    keys = [keys] if isinstance(keys, str) else keys
    if groupby is None:
        sdata_df = sdata[keys].to_dataframe()
    else:
        sdata_df = sdata[keys + [groupby]].to_dataframe()
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            sdata_df, keys, func=sns.countplot, groupby=groupby, orient=orient, **kwargs
        )
    if return_axes:
        return ax


def histplot(
    sdata,
    keys: Union[str, Sequence[str]],
    orient: str = "v",
    rc_context: Mapping[str, str] = settings.rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a histogram of a column(s) in seqs_annot using seaborn.

    This function can be used to show the distribution of a single or multiple
    columns of seqs_annot within a SeqData. If a groupby is provided then the
    distribution is grouped by the groupby column.

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
    sdata_df = sdata[keys].to_dataframe()
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            sdata_df, keys, func=sns.histplot, orient=orient, ylab="Frequency", **kwargs
        )
    if return_axes:
        return ax


def boxplot(
    sdata,
    keys: Union[str, Sequence[str]],
    groupby: str = None,
    orient: str = "v",
    jitter=False,
    rc_context: Mapping[str, str] = settings.rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a boxplot of a column(s) in seqs_annot using Seaborn.

    This function can be used to show the distribution of a single or multiple
    columns of seqs_annot within a SeqData. If a groupby is provided then the
    distribution is grouped by the groupby column.

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
    keys = [keys] if isinstance(keys, str) else keys
    if groupby is None:
        sdata_df = sdata[keys].to_dataframe()
    else:
        sdata_df = sdata[keys + [groupby]].to_dataframe()
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            sdata_df, keys, func=sns.boxplot, groupby=groupby, orient=orient, **kwargs
        )
        if jitter == True:
            _plot_seaborn(
                sdata_df,
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
    keys: Union[str, Sequence[str]] = None,
    groupby: str = None,
    orient: str = "v",
    rc_context: Mapping[str, str] = settings.rc_context,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a violinplot of a column(s) in seqs_annot using Seaborn.

    This function can be used to show the distribution of a single or multiple
    columns of seqs_annot within a SeqData as a violin plot. If a groupby is provided
    then the distribution is grouped by the groupby column.

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
         Matplotlib rc context. Uses settings by default.
     return_axes : bool
         Return axes.
     **kwargs
         Additional keyword arguments to pass to seaborn.

     Returns
     -------
         None
    """
    keys = [keys] if isinstance(keys, str) else keys
    if groupby is None:
        sdata_df = sdata[keys].to_dataframe()
    elif groupby is not None and isinstance(groupby, Iterable) and keys is None:
        sdata_df = sdata[groupby].to_dataframe()
    else:
        sdata_df = sdata[keys + [groupby]].to_dataframe()
    with plt.rc_context(rc_context):
        if groupby is not None and isinstance(groupby, Iterable) and keys is None:
            ax = _violin_long(sdata_df, groupby, **kwargs)
        else:
            ax = _plot_seaborn(
                sdata_df,
                keys,
                func=sns.violinplot,
                groupby=groupby,
                orient=orient,
                **kwargs
            )
    if return_axes:
        return ax


def scatterplot(
    sdata,
    x: str,
    y: str,
    seq_idx: Sequence[int] = None,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots a scatterplot of two columns in seqs_annot using Seaborn.

    This function can be used to show the relationship between two columns of
    seqs_annot within a SeqData. If seq_idx is provided then only the sequences
    with the given indices are plotted.

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
    **kwargs: dict
        Additional keyword arguments to pass to _plot_seaborn.
    Returns
    -------
    None
    """
    if seq_idx is not None:
        sdata = sdata[seq_idx]
    keys = [keys] if isinstance(keys, str) else keys
    sdata_df = sdata[keys].to_dataframe()
    ax = _plot_seaborn(
        sdata_df, keys=x, func=sns.scatterplot, groupby=y, xlab=x, ylab=y, **kwargs
    )
    if return_axes:
        return ax
