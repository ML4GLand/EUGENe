import numpy as np
import pandas as pd
import seaborn as sns
from os import PathLike
from ._utils import _save_fig


def positional_gia_plot(
    sdata, 
    keys: list, 
    id_key: str = "id",
    xlab: str = "Position",
    ylab: str = "Predicted Score",
    save: PathLike = None, 
    return_axes: bool = False
):
    """ 
    Plot a lineplot for each position of the sequence after implanting a feature.

    Assumes that the value corresponding to each seqsm_key in the sdata.uns dictionary
    has the same shape, namely (L, ) where L are the positions where a feature was implanted
    and scores were calculated using a model. Plots the scores as a line plot with a 95% CI
    corresponding to the number of sequences used to make the plot.

    Parameters
    ----------
    sdata : SeqData
        The SeqData object with sequences and scores to plot
    seqsm_keys : list
        The keys in the sdata.uns dictionary that contain the scores to plot
    xlab : str
        The x-axis label
    ylab : str
        The y-axis label
    save : PathLike
        The path to save the figure to
    return_axes : bool
        Whether to return the matplotlib axes object
    """
    concat_df = pd.DataFrame()
    for key in keys:
        df = pd.DataFrame(index=sdata[id_key].values, data=sdata[key].values).melt(
            var_name=xlab, 
            value_name=ylab, 
            ignore_index=False
        )
        df["feature"] = key
        concat_df = pd.concat([concat_df, df])
    concat_df.reset_index(drop=True, inplace=True)
    g = sns.lineplot(data=concat_df, x=xlab, y=ylab, hue="feature")
    if save:
        _save_fig(save)
    if return_axes:
        return g