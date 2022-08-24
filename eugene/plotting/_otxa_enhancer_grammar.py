import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from vizsequence import viz_sequence
from ..preprocessing._utils import _collapse_pos
from ..preprocessing._otx_preprocess import defineTFBS


def _plot_otx_seq(
    sdata,
    seq_id,
    uns_key=None,
    model_pred=None,
    threshold=None,
    highlight=[],
    cmap=None,
    norm=None,
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
        The key in the SeqData.uns dictionary to use to get the model predictions
    model_pred : str
        The key in the SeqData.uns dictionary to use to get the model predictions
    threshold : float
        The threshold to use to binarize the model predictions
    highlight : list
        A list of positions to highlight in the sequence
    cmap : str
        The name of the colormap to use
    norm : str
        The name of the normalization to use
    **kwargs : dict
        Additional keyword arguments to pass to the SeqLogo object

    Returns
    -------
    """

    # Get the sequence
    seq_idx = np.where(sdata.seqs_annot.index == seq_id)[0][0]
    seq = sdata.seqs[seq_idx]

    # Get the annotations for the seq
    tfbs_annot = defineTFBS(seq)

    # Define subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    # Build the annotations in the first subplot
    h = 0.1  # height of TFBS rectangles
    ax[0].set_ylim(0, 1)  # lims of axis
    ax[0].spines["bottom"].set_visible(
        False
    )  # remove axis surrounding, makes it cleaner
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["left"].set_visible(False)
    ax[0].tick_params(left=False)  # remove tick marks on y-axis
    ax[0].set_yticks([0.25, 0.525, 0.75])  # Add ticklabel positions
    ax[0].set_yticklabels(
        ["TFBS", "Affinity", "Closest OLS Hamming Distance"], weight="bold"
    )  # Add ticklabels
    ax[0].hlines(0.2, 1, len(seq), color="black")  #  Backbone to plot boxes on top of

    # Build rectangles for each TFBS into a dictionary
    tfbs_blocks = {}
    for pos in tfbs_annot.keys():
        if tfbs_annot[pos][0] == "GATA":
            tfbs_blocks[pos] = mpl.patches.Rectangle(
                (pos - 2, 0.2 - (h / 2)),
                width=8,
                height=h,
                facecolor="orange",
                edgecolor="black",
            )
        elif tfbs_annot[pos][0] == "ETS":
            tfbs_blocks[pos] = mpl.patches.Rectangle(
                (pos - 2, 0.2 - (h / 2)),
                width=8,
                height=h,
                facecolor="blue",
                edgecolor="black",
            )

    # Plot the TFBS with annotations, should be input into function
    for pos, r in tfbs_blocks.items():
        ax[0].add_artist(r)
        rx, ry = r.get_xy()
        ytop = ry + r.get_height()
        cx = rx + r.get_width() / 2.0
        tfbs_site = tfbs_annot[pos][0] + tfbs_annot[pos][1]
        tfbs_aff = round(tfbs_annot[pos][3], 2)
        closest_match = tfbs_annot[pos][5] + ": " + str(tfbs_annot[pos][7])
        spacing = tfbs_annot[pos][4]
        ax[0].annotate(
            tfbs_site,
            (cx, ytop),
            color="black",
            weight="bold",
            fontsize=12,
            ha="center",
            va="bottom",
        )
        ax[0].annotate(
            tfbs_aff,
            (cx, 0.45),
            color=r.get_facecolor(),
            weight="bold",
            fontsize=12,
            ha="center",
            va="bottom",
        )
        ax[0].annotate(
            closest_match,
            (cx, 0.65),
            color="black",
            weight="bold",
            fontsize=12,
            ha="center",
            va="bottom",
        )
        ax[0].annotate(
            str(spacing),
            (((rx - spacing) + rx) / 2, 0.25),
            weight="bold",
            color="black",
            fontsize=12,
            ha="center",
            va="bottom",
        )

    if uns_key is None:
        from ..preprocessing import ohe_DNA_seq

        print("No importance scores given, outputting just sequence")
        ylab = "Sequence"
        ax[1].spines["left"].set_visible(False)
        ax[1].set_yticklabels([])
        ax[1].set_yticks([])
        print(seq)
        importance_scores = ohe_DNA_seq(seq)
    else:
        importance_scores = sdata.uns[uns_key][seq_idx]
        ylab = "Importance Score"

    title = seq_id
    if model_pred is not None:
        model_pred = sdata[model_pred].iloc[seq_idx]
        color = cmap(norm(model_pred))
        title += ": {}".format(str(round(model_pred, 3)))
    else:
        color = "black"

    # Plot the featue importance scores
    if len(highlight) > 0:
        to_highlight = {"red": _collapse_pos(highlight)}
        print(to_highlight)
        viz_sequence.plot_weights_given_ax(
            ax[1],
            importance_scores,
            subticks_frequency=10,
            highlight=to_highlight,
            height_padding_factor=1,
        )
    else:
        viz_sequence.plot_weights_given_ax(
            ax[1], importance_scores, subticks_frequency=10, height_padding_factor=1
        )
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].set_xlabel("Sequence Position")
    ax[1].set_ylabel(ylab)
    if threshold is not None:
        ax[1].hlines(1, len(seq), threshold / 10, color="red")
    plt.suptitle(title, fontsize=24, weight="bold", color=color)
    return ax


def otx_seq(sdata, seq_id=None, **kwargs):
    """
    Plot a sequence of the data.
    """
    _plot_otx_seq(sdata, seq_id, **kwargs)


def prettier_boxplot(
    sdata, prediction, groupby, palette, order, xlabel=None, ylabel=None, threshold=0
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.boxplot(
        y=sdata.seqs_annot[prediction],
        x=sdata.seqs_annot[groupby],
        order=order,
        palette=palette,
        ax=ax,
    )
    sns.swarmplot(
        y=sdata.seqs_annot[prediction],
        x=sdata.seqs_annot[groupby],
        order=order,
        palette=palette,
        ax=ax,
        size=10,
        edgecolor="black",
        linewidth=2,
    )
    ax.hlines(
        threshold, ax.get_xlim()[0], ax.get_xlim()[1], color="red", linestyle="dashed"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
