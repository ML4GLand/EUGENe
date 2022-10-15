from typing import Union
from os import PathLike
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seqlogo
import logomaker as lm
from vizsequence import viz_sequence
from tqdm.auto import tqdm
from ..preprocess._utils import _collapse_pos
from ._utils import _save_fig
from .. import settings


vocab_dict = {
    "DNA": ["A", "C", "G", "T"], 
    "RNA": ["A", "C", "G", "U"]
}


def _plot_seq_features(
    ax: Axes,
    seq: str,
    annots: pd.DataFrame,
    additional_annots: list = [],
):
    """
    Plot sequence features using matplotlib.

    This uses basic matplotlib rectangles and lines to plot sequence features
    as blocks. Can be used along with importance scores to give a visual of where the 
    a prior known features of a sequence are

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
    for _, annot in annots.iterrows():
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
                if isinstance(annot[add_annot], float):
                    ann = "{:.2f}".format(annot[add_annot])
                else:
                    ann = annot[add_annot]
                ax.annotate(
                    ann,
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
    threshold: float = None,
    ylab="Importance Score",
    **kwargs,
):
    """
    Plot sequence logo using plot_weights_given_ax function from viz_sequence

    This allows for the plotting of sequence logos using the viz_sequence package.

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
        Additional keyword arguments to pass to plot_weights_given_ax

    Returns
    -------
    None

    Note
    ----
    This was adapted from the viz_sequence package:
    https://github.com/kundajelab/vizsequence
    """
    if imp_scores is None:
        from ..preprocess import ohe_seq
        print("No importance scores given, outputting just sequence")
        ylab = "Sequence" if ylab is None else ylab
        ax.spines["left"].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
        imp_scores = ohe_seq(seq)
    else:
        ylab = "Importance Score" if ylab is None else ylab

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
            **kwargs,
        )
    else:
        viz_sequence.plot_weights_given_ax(
            ax,
            imp_scores,
            subticks_frequency=int(len(seq) / 10),
            height_padding_factor=1,
            **kwargs,
        )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel(ylab)
    if threshold is not None:
        ax.hlines(1, len(seq), threshold / 10, color="red")


def seq_track_features(
    sdata,
    seq_id: str,
    uns_key: str = None,
    additional_annotations: list = ["Score", "Strand"],
    pred_key: str = None,
    threshold: float = None,
    highlight: list = [],
    cmap=None,
    norm=None,
    return_axes: bool = False,
    save: str = None,
    **kwargs,
):
    """
    Function to plot tracks from a SeqData object using matplotlib and function
    from viz_sequence package.
    
    This function allows users to also add features from the pos_annot attribute,
    which is not currently available with seq_track function.

    This also allows users to just plot the sequences with no importance scores, which is
    currently not available with seq_track function.


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
    p_annot = (
        sdata.pos_annot.df[sdata.pos_annot.df["Chromosome"] == seq_id]
        if sdata.pos_annot is not None
        else None
    )
    imp_scores = sdata.uns[uns_key][seq_idx] if uns_key in sdata.uns.keys() else None

    # Define subplots
    _, ax = (
        plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        if p_annot is not None
        else plt.subplots(1, 1, figsize=(12, 4))
    )
    plt.subplots_adjust(wspace=0, hspace=0)

    # Plot the sequence and annotations
    if p_annot is not None:
        _plot_seq_features(
            ax[0], seq, p_annot, additional_annots=additional_annotations
        )
        _plot_seq_logo(
            ax[1],
            seq,
            imp_scores=imp_scores,
            highlight=highlight,
            threshold=threshold,
            **kwargs,
        )
    else:
        _plot_seq_logo(
            ax,
            seq,
            imp_scores=imp_scores,
            highlight=highlight,
            threshold=threshold,
            **kwargs,
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
    if save is not None:
        _save_fig(save)


def multiseq_track_features(
    sdata,
    seq_ids: list,
    uns_keys: str = None,
    ylabs: list = None,
    width=None,
    height=None,
    return_axes: bool = False,
    save: str = None,
    **kwargs,
):
    """
    Wrapper around seq_track_features function to plot multiple tracks from a SeqData object
    using matplotlib and viz_sequence. This function allows users to also add features from the 
    pos_annot attribute
    
    Parameters
    ----------
    sdata : SeqData object
        The SeqData object to plot
    seq_ids : list
        The IDs of the sequences to plot
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
    ax : matplotlib.axes.Axes
        The axes object
    """
    if isinstance(seq_ids, str):
        seq_ids = [seq_ids]
    if isinstance(uns_keys, str):
        uns_keys = [uns_keys]
    ylabs = ylabs if ylabs is not None else ["Importance Score"] * len(uns_keys)
    seq_idx = np.where(sdata.seqs_annot.index.isin(seq_ids))[0]
    seqs = sdata.seqs[seq_idx]
    fig_width = (
        len(seq_ids) * int(len(seqs[0]) / 20) if width is None else width
    )  # make each sequence width proportional to its length and multiply by the number of sequences
    fig_height = (
        len(uns_keys) * 4 if height is None else height
    )  # make each sequence height proportional to the number of uns_keys
    _, ax = plt.subplots(len(uns_keys), len(seq_ids), figsize=(fig_width, fig_height))
    for i, uns_key in tqdm(enumerate(uns_keys), desc="Importance values", position=0):
        for j, seq in enumerate(seqs):
            imp_scores = (
                sdata.uns[uns_key][seq_idx[j]] if uns_key in sdata.uns.keys() else None
            )
            _plot_seq_logo(
                ax.flatten()[i * len(seq_ids) + j],
                seq,
                imp_scores=imp_scores,
                ylab=ylabs[i],
            )
            if i == 0:
                ax.flatten()[i * len(seq_ids) + j].set_title(
                    seq_ids[j], fontsize=18, weight="bold"
                )
    plt.tight_layout()
    if return_axes:
        return ax
    if save is not None:
        _save_fig(save)


def _plot_logo_seqlogo(
    matrix, 
    **kwargs
):
    """
    Plot a sequence logo of a position frequency matrix (PFM) using the SeqLogo package. 
    
    This function is deprecated because there is no easy way to save these as
    figures because they are not matplotlib axes.

    Parameters
    ----------
    matrix : numpy.ndarray
        The position frequency matrix to plot
    **kwargs : dict
        Additional keyword arguments to pass to the SeqLogo object
    """
    cpm = seqlogo.CompletePm(pfm=matrix)
    logo = seqlogo.seqlogo(
        cpm, 
        ic_scale=True, 
        format="png", 
        **kwargs
    )
    display(logo)
    return logo


def filter_viz_seqlogo(
    sdata, 
    filter_id: Union[str, list], 
    uns_key: str = "pfms", 
    return_logo: bool = False,
    **kwargs
):
    """Plot the logo of the pfm generated for a passed in filter.  
    
    This function is deprecated because there is no easy way to save these as
    figures because they are not matplotlib axes.
    
    If a filter_id is given, the logo will be filtered to only show the features that match the filter_id.
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
    logo = _plot_logo_seqlogo(sdata.uns[uns_key][filter_id], **kwargs)
    if return_logo:
        return logo


def seq_track(
    sdata,
    seq_id: str,
    uns_key: str,
    vocab: str = "DNA",
    highlights: list = [],
    highlight_colors: list = ["lavenderblush", "lightcyan", "honeydew"],
    title: str ="",
    ylab: str = "Saliency",
    xlab: str = "Position",
    return_ax: bool = False,
    save: PathLike = None,
    **kwargs,
):
    """
    Plot a track of the importance scores for a sequence using the logomaker package
    
    This function is a wrapper around the logomaker Logo function. See the logomaker documentation
    for more details on the kwargs that can be passed to this function.

    Currently does no allow for features to be plotted (users must do them themselves on returned axes) or
    for sequence only plotting (i.e. importance scores must be passed in through the uns key)

    Parameters
    ----------
    sdata : SeqData
        The SeqData object to plot the logo for
    seq_id : str
        The ID of the sequence to plot
    uns_key : str
        The key in the sdata.uns dictionary that contains the importance scores
    vocab : str
        The vocabulary to use for the sequence
    highlights : list
        A list of positions to highlight in the sequence
    highlight_colors : list 
        A list of colors to use for the highlights
    title : str
        The title to use for the plot
    ylab : str
        The y-axis label to use for the plot
    xlab : str
        The x-axis label to use for the plot
    return_ax : bool
        Whether to return the axes object

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object
    """ 
    if isinstance(highlights, tuple):
        highlights = [highlights]
    if isinstance(highlight_colors, str):
        highlight_colors = [highlight_colors] * len(highlights)
    seq_idx = np.where(sdata.seqs_annot.index == seq_id)[0][0]
    imp_scores = sdata.uns[uns_key][seq_idx] if uns_key in sdata.uns.keys() else None
    viz_seq = pd.DataFrame(imp_scores.T, columns=vocab_dict[vocab])
    viz_seq.index.name = "pos"
    y_max = np.max(viz_seq.values)
    y_min = np.min(viz_seq.values)
    nn_logo = lm.Logo(viz_seq, **kwargs)

    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=["left"], visible=True, bounds=[y_min, y_max])

    # style using Axes methods
    nn_logo.ax.set_xlim([0, len(viz_seq)])
    nn_logo.ax.set_xticks([])
    nn_logo.ax.set_ylim([y_min, y_max])
    nn_logo.ax.set_ylabel(ylab)
    nn_logo.ax.set_xlabel(xlab)
    nn_logo.ax.set_title(title)
    for i, highlight in enumerate(highlights):
        nn_logo.highlight_position_range(
            pmin=highlight[0], 
            pmax=highlight[1], 
            color=highlight_colors[i]
        )
    if save is not None:
        _save_fig(save)
    if return_ax:
        return nn_logo.ax


def multiseq_track(
    sdata,
    seq_ids: list,
    uns_keys: str = None,
    ylabs: list = None,
    width: int = None,
    height: int = None,
    return_axes: bool = False,
    save: str = None,
    **kwargs,
):
    """ 
    Plot the saliency tracks for multiple sequences across multiple importance scores in one plot.

    Wraps the seq_track function to plot multiple sequences at once across multiple importance scores. 

    Attempts to make each sequence width proportional to its length and multiply by the number of sequences
    if no width is passed in.

    Attempts to make each sequence height proportional to the number of uns_keys passed in (the number of different
    importance scores to plot) if no height is passed in.

    Parameters
    ----------
    sdata : SeqData
        The SeqData object with sequences and importances to plot a logo for
    seq_ids : list
        The sequence ids to plot
    uns_keys : list
        The keys in the sdata.uns dictionary that contain the importance scores to plot
    ylabs : list
        The ylabs to use for each importance score
    width : int
        The width of the figure to plot
    height : int
        The height of the figure to plot
    return_axes : bool
        Whether to return the matplotlib axes objects
    save : str
        The path to save the figure to
    **kwargs : dict
        Additional keyword arguments to pass to the seq_track function

    Returns
    -------
    axes : list
        The axes objects if return_axes is True
    """
    if isinstance(seq_ids, str):
        seq_ids = [seq_ids]
    if isinstance(uns_keys, str):
        uns_keys = [uns_keys]
    if isinstance(ylabs, str):
        ylabs = [ylabs]
    seq = sdata.seqs[0]
    ylabs= ylabs if ylabs is not None else ["Importance Score"] * len(uns_keys)
    fig_width = (len(seq_ids) * int(len(seq) / 20) if width is None else width)  
    fig_height = (len(uns_keys) * 4 if height is None else height)
    _, ax = plt.subplots(len(uns_keys), len(seq_ids), figsize=(fig_width, fig_height))
    for i, uns_key in tqdm(enumerate(uns_keys), desc="Importance values", position=0, total=len(uns_keys)):
        for j, seq_id in enumerate(seq_ids):
            seq_track(
                sdata,
                seq_id=seq_id,
                uns_key=uns_key,
                ax=ax.flatten()[i * len(seq_ids) + j],
                ylab=ylabs[i],
                title=seq_id,
                save=None,
                **kwargs,
            )
    plt.tight_layout()
    if save is not None:
        _save_fig(save)
    if return_axes:
        return ax


def filter_viz(
    sdata,
    filter_id: Union[str, int],
    uns_key: str = "pfms",
    vocab: str = "DNA",
    title: str = None,
    return_ax: bool = False,
    save: str = None,
    **kwargs,
):
    """ 
    Plot the PFM for a single filter in a SeqData object's uns dictionary as a PWM logo

    This function also uses logomaker to generate the PWM and plot it. Check out the logomaker documentation
    for more information on how to style the plot.

    Parameters
    ----------
    sdata : SeqData
        The SeqData object with sequences and pfms to plot a logo for
    filter_id : str or int
        The filter id to plot
    uns_key : str
        The key in the sdata.uns dictionary that contains the pfms to plot
    vocab : str
        The vocabulary to use for the logo
    title : str
        The title to use for the plot, defaults to the filter id if None
    return_ax : bool
        Whether to return the matplotlib axes object

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object if return_ax is True
    """
    pfm = sdata.uns[uns_key][filter_id]
    if isinstance(pfm, np.ndarray):
        pfm = pd.DataFrame(pfm, columns=vocab_dict[vocab])
    vocab = vocab_dict[vocab]
    if pfm[vocab[0]].dtype == "float64":
        pfm.fillna(0.25, inplace=True)
        info_mat = lm.transform_matrix(
            pfm, 
            from_type="probability", 
            to_type="information"
        )
    elif pfm[vocab.keys[0]].dtype == "int64":
        pfm.fillna(1, inplace=True)
        info_mat = lm.transform_matrix(
            pfm, 
            from_type="counts", 
            to_type="information", 
            allow_nan=True
        )
    if "N" in pfm.columns:
        info_mat = info_mat.drop("N", axis=1)
    logo = lm.Logo(info_mat, **kwargs)
    logo.style_xticks(spacing=5, anchor=25, rotation=45, fmt="%d", fontsize=14)
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True, linewidth=2)
    logo.ax.set_ylim([0, 2])
    logo.ax.set_yticks([0, 1, 2])
    logo.ax.set_yticklabels(["0", "1", "2"])
    logo.ax.set_ylabel("bits")
    logo.ax.set_title(title if title is not None else filter_id)
    if save is not None:
        _save_fig(save)
    if return_ax:
        return logo.ax


def multifilter_viz(
    sdata,
    filter_ids: list,
    num_rows: int = None,
    num_cols: int = None,
    uns_key: str = "pfms",
    titles: list = None,
    figsize=(12,10),
    save: PathLike = None,
    **kwargs,
):
    """
    Plot multiple filters in a SeqData object's uns dictionary as PWM logos.

    This function wraps filter_viz. Getting the figure to look nice it more of an art
    than a science. In experimenting so far, I've found that a 8x4 grid with a (12, 10)
    figure size works well. 

    Parameters 
    ----------
    sdata : SeqData
        The SeqData object with sequences and pfms to plot a logo for
    filter_ids : list
        The filter ids to plot
    num_rows : int
        The number of rows to use for the figure
    num_cols : int
        The number of columns to use for the figure
    uns_key : str
        The key in the sdata.uns dictionary that contains the pfms to plot
    titles : list
        The titles to use for the plots, defaults to the filter ids if None
    figsize : tuple
        The figure size to use for the plot
    save : PathLike
        The path to save the figure to
    
    Returns
    -------
    axes : list
        The axes objects if return_axes is True
    """

    _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            filter_id = filter_ids[i * num_cols + j]
            filter_viz(
                sdata,
                filter_id=filter_id,
                uns_key=uns_key,
                ax=ax.flatten()[i * num_cols + j],
                title=titles[i * num_cols + j] if titles is not None else filter_id,
                save=None,
                **kwargs,
            )

    plt.tight_layout()
    if save is not None:
        _save_fig(save)


def kipoi_ism_heatmap(
    sdata, 
    seq_id: Union[str, int], 
    uns_key: str = "NaiveISM_imps", 
    figsize: tuple = (15, 2.5),
    save: PathLike = None,
    return_axes: bool = False
):
    """ 
    Wrapper function around Kipoi's seqlogo_heatmap function that generates a really 
    nice heatmap of the importance scores for a single sequence in a SeqData object's
    uns dictionary.

    Parameters
    ----------
    sdata : SeqData
        The SeqData object with sequences and importance scores to plot a heatmap for
    seq_id : str or int
        The sequence id to plot
    uns_key : str
        The key in the sdata.uns dictionary that contains the importance scores to plot
    figsize : tuple
        The figure size to use for the plot
    save : PathLike
        The path to save the figure to
    return_axes : bool
        Whether to return the matplotlib axes object
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from ..external.kipoi.kipoi_veff.plot import seqlogo_heatmap
    seq_idx = np.where(sdata.seqs_annot.index == seq_id)[0][0]
    val = sdata.uns[uns_key][seq_idx]
    ax = plt.figure(figsize=figsize)
    seqlogo_heatmap(val.T, val, ax=plt.subplot())
    if save:
        _save_fig(save)
    if return_axes:
        return ax


def feature_implant_plot(
    sdata, 
    seqsm_keys: list, 
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
    for seqsm_key in seqsm_keys:
        df = pd.DataFrame(index=sdata.names, data=sdata.seqsm[seqsm_key]).melt(
            var_name=xlab, 
            value_name=ylab, 
            ignore_index=False
        )
        df["feature"] = seqsm_key
        concat_df = pd.concat([concat_df, df])
    concat_df.reset_index(drop=True, inplace=True)
    g = sns.lineplot(data=concat_df, x=xlab, y=ylab, hue="feature")
    if save:
        _save_fig(save)
    if return_axes:
        return g
