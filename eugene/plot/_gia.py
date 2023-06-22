import numpy as np
import pandas as pd
import seaborn as sns
from os import PathLike
from ._utils import _save_fig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def positional_gia_plot(
    sdata,
    keys: list,
    id_key: str = "id",
    xlab: str = "Position",
    ylab: str = "Predicted Score",
    ylim: tuple = None,
    save: PathLike = None,
    return_axes: bool = False,
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
            var_name=xlab, value_name=ylab, ignore_index=False
        )
        df["feature"] = key
        concat_df = pd.concat([concat_df, df])
    concat_df.reset_index(drop=True, inplace=True)
    g = sns.lineplot(data=concat_df, x=xlab, y=ylab, hue="feature")
    if ylim is not None:
        g.set(ylim=ylim)
    if save:
        _save_fig(save)
    if return_axes:
        return g

def distance_cooperativity_gia_plot(
    sdata,
    results_key="cooperativity",
    distance_key="distance",
    col_names=None,
    cols_to_plot=None,
    motif_a_name="",
    motif_b_name="",
):
    cooperativity_df = pd.DataFrame(np.median(sdata[results_key], axis=1))
    cooperativity_df["distance"] = [int(d[1:]) for d in sdata[distance_key].values]
    cooperativity_df[f"relative_to_{motif_a_name}"] = [d[0] for d in sdata[distance_key].values]
    if col_names is not None:
        cooperativity_df.columns = col_names + ["distance", f"relative_to_{motif_a_name}"]
    else:
        col_names = cooperativity_df.columns[:-2]

    # Plot the results
    labels = []
    handles = []
    for col in col_names:
        if cols_to_plot is not None and col not in cols_to_plot:
            continue
        sns.lineplot(
            data=cooperativity_df,
            x="distance",
            y=col,
            hue=f"relative_to_{motif_a_name}",
            linestyle="-"
        )
        labels += [f"{col}, +", f"{col}, -"]
        handles += [
            Line2D([0], [0], color="orange", linestyle="-"),
            Line2D([0], [0], color="blue", linestyle="-")
        ]

    # Denote the dev vs hk lines styles and colors for the relative to motif a with a legend
    plt.legend(
        title=f"Relative to {motif_a_name}",
        loc="upper right",
        labels=labels,
        handles=handles
    )

    # Set the x-axis label to "Distance from motif B to motif A"
    plt.ylabel("Median predicted cooperativity")
    plt.xlabel("Motif pair distance")
    plt.title(f"{motif_a_name}/{motif_b_name} cooperativity")

    plt.show()