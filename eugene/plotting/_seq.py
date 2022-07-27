import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seqlogo
from vizsequence import viz_sequence
from ..preprocessing._utils import _collapse_pos


def _plot_seq_features(
    ax: Axes,
    seq: str,
    annots: pd.DataFrame,
    additional_annots: list = [],
):
    """
    Plot sequence features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on
    seq : str
        The sequence to plot
    annots : pandas.DataFrame
        The annotations to plot
    additional_annots : list
        A list of additional annotations to plot

    Returns
    -------
    None
    """
    h = 0.1  # height of TFBS rectangles
    ax.set_ylim(0, 1)  # lims of axis
    ax.spines["bottom"].set_visible(False)  # remove axis surrounding, makes it cleaner
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)  # remove tick marks on y-axis
    ax.set_yticks([0.25, 0.525, 0.75])  # Add ticklabel positions
    ax.set_yticklabels(["Feature"] + additional_annots, weight="bold")  # Add ticklabels
    ax.hlines(0.2, 1, len(seq), color="black")  #  Backbone to plot boxes on top of

    # Build rectangles for each TFBS into a dictionary
    for row, annot in annots.iterrows():
        start = annot["Start"]
        end = annot["End"]
        name = annot["Name"] if annot["Name"] else "Unknown"
        strand = annot["Strand"] if annot["Strand"] else "Unknown"
        color = "red" if strand == "+" else "blue"
        feature_block = mpl.patches.Rectangle(
            (start, 0.2 - (h / 2)),
            width=end - start + 1,
            height=h,
            facecolor=color,
            edgecolor="black",
        )
        ax.add_artist(feature_block)
        rx, ry = feature_block.get_xy()
        ytop = ry + feature_block.get_height()
        cx = rx + feature_block.get_width() / 2.0
        ax.annotate(
            name,
            (cx, ytop),
            color="black",
            weight="bold",
            fontsize=12,
            ha="center",
            va="bottom",
        )
        for i, add_annot in enumerate(additional_annots):
            if add_annot in annot.index:
                ax.annotate(
                    annot[add_annot],
                    (cx, 0.45 + i * 0.2),
                    color="black",
                    weight="bold",
                    fontsize=12,
                    ha="center",
                    va="bottom",
                )


def _plot_seq_logo(
    ax: Axes,
    seq: str,
    imp_scores: np.ndarray = None,
    highlight: list = [],
    threshold: float = 0.5,
    **kwargs
):
    """
    Plot sequence logo.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on
    seq : str
        The sequence to plot
    imp_scores : np.ndarray
        The importance scores to plot
    highlight : list
        A list of positions to highlight
    threshold : float
        The threshold for importance scores to highlight
    **kwargs
        Additional keyword arguments to pass to seqlogo.plot_seq_logo

    Returns
    -------
    None
    """
    if imp_scores is None:
        from ..preprocessing import ohe_DNA_seq

        print("No importance scores given, outputting just sequence")
        ylab = "Sequence"
        ax.spines["left"].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
        imp_scores = ohe_DNA_seq(seq)
    else:
        ylab = "Importance Score"

    # Plot the featue importance scores
    if len(highlight) > 0:
        to_highlight = {"red": _collapse_pos(highlight)}
        print(to_highlight)
        viz_sequence.plot_weights_given_ax(
            ax,
            imp_scores,
            subticks_frequency=10,
            highlight=to_highlight,
            height_padding_factor=1,
            **kwargs
        )
    else:
        viz_sequence.plot_weights_given_ax(
            ax, imp_scores, subticks_frequency=10, height_padding_factor=1, **kwargs
        )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel(ylab)
    if threshold is not None:
        ax.hlines(1, len(seq), threshold / 10, color="red")


def seq_track(
    sdata,
    seq_id: str,
    uns_key: str,
    additional_annotations: list = ["Score", "Strand"],
    pred_key: str = None,
    threshold: float = None,
    highlight: list = [],
    cmap=None,
    norm=None,
    return_axes: bool = False,
    **kwargs
):
    """
    Function to plot tracks from a SeqData object

    Parameters
    ----------
    sdata : SeqData object
        The SeqData object to plot
    seq_id : str
        The ID of the sequence to plot
    uns_key : str
        The key in the SeqData.uns dictionary to use to get the nucleotide scores
    pred_key : str
        The key in the SeqData.seqs_annot
    threshold : float
        The threshold to use to draw a cut-off line
    highlight : list
        A list of positions to highlight in the sequence
    cmap : str
        The name of the colormap to use
    norm : str
        The name of the normalization to use
    return_axes : bool
        Whether to return the axes object
    **kwargs : dict
        Additional keyword arguments to pass to vizsequence call

    Returns
    -------
    if return_axes:
        ax : matplotlib.axes.Axes
            The axes object
    """

    # Get the sequence and annotations
    seq_idx = np.where(sdata.seqs_annot.index == seq_id)[0][0]
    seq = sdata.seqs[seq_idx]
    p_annot = sdata.pos_annot.df[sdata.pos_annot.df["Chromosome"] == seq_id]
    imp_scores = sdata.uns[uns_key][seq_idx] if uns_key in sdata.uns.keys() else None

    # Define subplots
    _, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    # Plot the sequence and annotations
    _plot_seq_features(ax[0], seq, p_annot, additional_annots=additional_annotations)
    _plot_seq_logo(
        ax[1],
        seq,
        imp_scores=imp_scores,
        highlight=highlight,
        threshold=threshold,
        **kwargs
    )

    # Add title
    title = seq_id
    if pred_key is not None:
        model_pred = sdata.seqs_annot[pred_key].iloc[seq_idx]
        if cmap is not None:
            color = cmap(norm(model_pred))
        else:
            color = "black"
        title += ": {}".format(str(round(model_pred, 3)))
    else:
        color = "black"
    plt.suptitle(title, fontsize=24, weight="bold", color=color)
    if return_axes:
        return ax


def _plot_logo(matrix, **kwargs):
    """
    Plot a sequence logo using the SeqLogo package.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to plot
    **kwargs : dict
        Additional keyword arguments to pass to the SeqLogo object

    Returns
    -------
    """
    cpm = seqlogo.CompletePm(pfm=matrix)
    logo = seqlogo.seqlogo(cpm, ic_scale=True, format="png", **kwargs)
    display(logo)


def filter_viz(sdata, filter_id=None, uns_key="pfms", **kwargs):
    """Plot the logo of the pfm generated for a passed in filter.  If a filter_id is given,
    the logo will be filtered to only show the features that match the filter_id.
    The uns_key is the key in the sdata.uns dictionary that contains the importance scores.
    The kwargs are passed to the SeqLogo object. See the SeqLogo documentation for more details.

    Parameters
    ----------
    sdata : SeqData
        The SeqData object to plot the logo for
    filter_id : str
        The filter_id to use to filter the logo
    uns_key : str
        The key in the sdata.uns dictionary that contains the importance scores
    **kwargs : dict
        The keyword arguments to pass to the SeqLogo object

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object
    """
    return _plot_logo(sdata.uns[uns_key][filter_id], **kwargs)
