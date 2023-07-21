import numpy as np
import pandas as pd
import seaborn as sns
from os import PathLike
from ._utils import _save_fig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Union, Sequence, Mapping, List

def positional_gia_plot(
    sdata,
    vars: list,
    id_var: str = "id",
    xlab: str = "Position",
    ylab: str = "Predicted Score",
    ylim: Optional[tuple] = None,
    save: Optional[PathLike] = None,
    return_axes: bool = False,
):
    """Plot a lineplot for each position of the sequence after implanting a feature.

    Assumes that the value corresponding to each seqsm_var in the sdata.uns dictionary
    has the same shape, namely (L, ) where L are the positions where a feature was implanted
    and scores were calculated using a model. Plots the scores as a line plot with a 95% CI
    corresponding to the number of sequences used to make the plot.

    Parameters
    ----------
    sdata : SeqData
        The SeqData object with sequences and scores to plot
    id_var : str
        The name of the variable in sdata.obs to use as the x-axis
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
    for key in vars:
        df = pd.DataFrame(index=sdata[id_var].values, data=sdata[key].values).melt(
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
    sdata: np.ndarray,
    results_var: str = "cooperativity",
    distance_var: str = "distance",
    col_names: Optional[List[str]] = None,
    cols_to_plot: Optional[List[str]] = None,
    motif_a_name: str = "",
    motif_b_name: str = "",
) -> None:
    """Plot the median predicted cooperativity as a function of motif pair distance.

    Parameters
    ----------
    sdata : np.ndarray
        The input data array.
    results_var : str, optional
        The name of the variable containing the cooperativity results, by default "cooperativity".
    distance_var : str, optional
        The name of the variable containing the motif pair distances, by default "distance".
    col_names : List[str], optional
        The names of the columns in the input data array, by default None.
    cols_to_plot : List[str], optional
        The names of the columns to plot, by default None.
    motif_a_name : str, optional
        The name of the first motif, by default "".
    motif_b_name : str, optional
        The name of the second motif, by default "".

    Returns
    -------
    None
    """
    cooperativity_df = pd.DataFrame(np.median(sdata[results_var], axis=1))
    cooperativity_df["distance"] = [int(d[1:]) for d in sdata[distance_var].values]
    cooperativity_df[f"relative_to_{motif_a_name}"] = [d[0] for d in sdata[distance_var].values]
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
    