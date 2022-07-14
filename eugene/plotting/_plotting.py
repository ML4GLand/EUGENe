import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import seqlogo
from vizsequence import viz_sequence
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import binarize
from ..utils import collapse_pos, defineTFBS


def seq(sdata, seq_idx=None, **kwargs):
    """
    Plot a sequence of the data.
    """
    _plot_seq(sdata, seq_idx, **kwargs)


def _plot_seq(sdata, seq_id, uns_key = None, model_pred=None, threshold=None, highlight=[], cmap=None, norm=None, **kwargs):
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
    fig, ax = plt.subplots(2, 1, figsize=(12,4), sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    # Build the annotations in the first subplot
    h = 0.1  # height of TFBS rectangles
    ax[0].set_ylim(0, 1)  # lims of axis
    ax[0].spines['bottom'].set_visible(False)  #remove axis surrounding, makes it cleaner
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].tick_params(left = False)  #remove tick marks on y-axis
    ax[0].set_yticks([0.25, 0.525, 0.75])  # Add ticklabel positions
    ax[0].set_yticklabels(["TFBS", "Affinity", "Closest OLS Hamming Distance"], weight="bold")  # Add ticklabels
    ax[0].hlines(0.2, 1, len(seq), color="black")  #  Backbone to plot boxes on top of

    # Build rectangles for each TFBS into a dictionary
    tfbs_blocks = {}
    for pos in tfbs_annot.keys():
        if tfbs_annot[pos][0] == "GATA":
            tfbs_blocks[pos] = mpl.patches.Rectangle((pos-2, 0.2-(h/2)), width=8, height=h, facecolor="orange", edgecolor="black")
        elif tfbs_annot[pos][0] == "ETS":
            tfbs_blocks[pos] = mpl.patches.Rectangle((pos-2, 0.2-(h/2)), width=8, height=h, facecolor="blue", edgecolor="black")

    # Plot the TFBS with annotations, should be input into function
    for pos, r in tfbs_blocks.items():
        ax[0].add_artist(r)
        rx, ry = r.get_xy()
        ytop = ry + r.get_height()
        cx = rx + r.get_width()/2.0
        tfbs_site = tfbs_annot[pos][0] + tfbs_annot[pos][1]
        tfbs_aff = round(tfbs_annot[pos][3], 2)
        closest_match = tfbs_annot[pos][5] + ": " + str(tfbs_annot[pos][7])
        spacing = tfbs_annot[pos][4]
        ax[0].annotate(tfbs_site, (cx, ytop), color='black', weight='bold',
                    fontsize=12, ha='center', va='bottom')
        ax[0].annotate(tfbs_aff, (cx, 0.45), color=r.get_facecolor(), weight='bold',
                    fontsize=12, ha='center', va='bottom')
        ax[0].annotate(closest_match, (cx, 0.65), color="black", weight='bold',
                    fontsize=12, ha='center', va='bottom')
        ax[0].annotate(str(spacing), (((rx-spacing) + rx)/2, 0.25), weight='bold', color="black",
                fontsize=12, ha='center', va='bottom')

    if uns_key is None:
        from ..preprocessing import oheDNA
        print("No importance scores given, outputting just sequence")
        ylab = "Sequence"
        ax[1].spines['left'].set_visible(False)
        ax[1].set_yticklabels([])
        ax[1].set_yticks([])
        print(seq)
        importance_scores = oheDNA(seq)
    else:
        importance_scores = sdata.uns[uns_key][seq_idx]
        ylab = "Importance Score"

    title = seq_id
    if model_pred is not None:
        color = cmap(norm(model_pred))
        title += ": {}".format(str(round(model_pred, 3)))
    else:
        color = "black"

    # Plot the featue importance scores
    if len(highlight) > 0:
        to_highlight = {"red": collapse_pos(highlight)}
        print(to_highlight)
        viz_sequence.plot_weights_given_ax(ax[1], importance_scores, subticks_frequency=10, highlight=to_highlight, height_padding_factor=1)
    else:
        viz_sequence.plot_weights_given_ax(ax[1], importance_scores, subticks_frequency=10, height_padding_factor=1)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_xlabel("Sequence Position")
    ax[1].set_ylabel(ylab)
    if threshold is not None:
        ax[1].hlines(1, len(seq), threshold/10, color="red")
    plt.suptitle(title, fontsize=24, weight="bold", color=color)
    return ax


def logo(sdata, filter_id=None, uns_key="pfms", **kwargs):
    """ Plot the logo of the sequence.  If a filter_id is given,
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
    cpm = seqlogo.CompletePm(pfm = matrix)
    logo = seqlogo.seqlogo(cpm, ic_scale = True, format="png", **kwargs)
    display(logo)


def performance_scatter(sdata, seq_idx=None, target="TARGETS", prediction="PREDICTIONS", **kwargs):

    # Get the indices of the sequences in the subset
    if seq_idx is None:
        sdata = sdata[seq_idx]

    _plot_performance_scatter(sdata, **kwargs)


def _plot_performance_scatter(sdata, target="TARGETS", prediction="PREDICTIONS", **kwargs):
    """
    Plot a scatter plot of the performance of the model on a subset of the sequences.
    """
    # Get the predictions and targets
    targets = sdata.seqs_annot[target]
    predictions = sdata.seqs_annot[prediction]

    # Plot the scatter plot
    plt.scatter(targets, predictions, **kwargs)
    plt.xlabel("TARGETS")
    plt.ylabel("PREDICTIONS")
    plt.title("Performance of Model on Subset")
    plt.show()


def confusion_mtx(sdata, **kwargs):
    _plot_confusion_mtx(sdata, **kwargs)


def _plot_confusion_mtx(sdata, target="TARGETS", prediction="PREDICTIONS", title="Sequences", xlab="Predicted Activity", ylab="True Activity", threshold=0.5):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        cf_names = ["True Neg","False Pos", "False Neg","True Pos"]
        ts = binarize(sdata.seqs_annot[target].values.reshape(-1, 1), threshold=threshold)
        ps = binarize(sdata.seqs_annot[prediction].values.reshape(-1, 1), threshold=threshold)
        cf_mtx = confusion_matrix(ts, ps)
        cf_pcts = ["{0:.2%}".format(value) for value in (cf_mtx/cf_mtx.sum(axis=1)[:,None]).flatten()]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(cf_mtx.flatten(),cf_pcts, cf_names)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cf_mtx, annot=labels, fmt='s', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.set_title(title, fontsize=24)
        ax.set_yticklabels(["Inactive", "Active"], fontsize=16)
        ax.set_xticklabels(["Inactive (Score<{})".format(str(threshold)), "Active (Score>{})".format(str(threshold))], fontsize=16)
        plt.tight_layout()


def auroc(sdata, **kwargs):
    _plot_auroc(sdata, **kwargs)


def _plot_auroc(sdata, target="TARGETS", prediction="PREDICTIONS", title="ROC Curve", xlab="False Positive Rate", ylab="True Positive Rate", threshold=0.5):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        ts = sdata.seqs_annot[target].values.reshape(-1, 1)
        ps = sdata.seqs_annot[prediction].values.reshape(-1, 1)
        fpr, tpr, _ = roc_curve(ts, ps)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.set_title(title, fontsize=24)
        ax.legend(loc="lower right")
        plt.tight_layout()


def auprc(sdata, **kwargs):
    _plot_auprc(sdata, **kwargs)


def _plot_auprc(sdata, target="TARGETS", prediction="PREDICTIONS", title="AUPRC Curve", xlab="Recall", ylab="Precision", threshold=0.5):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        ts = sdata.seqs_annot[target].values.reshape(-1, 1)
        ps = sdata.seqs_annot[prediction].values.reshape(-1, 1)
        precision, recall, _ = precision_recall_curve(ts, ps)
        average_precision = average_precision_score(ts, ps)
        ax.plot(recall, precision, label=f"AUPRC curve (AP = {average_precision:.3f})")
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.set_title(title, fontsize=24)
        ax.legend(loc="lower right")
        plt.tight_layout()


def violin(sdata, **kwargs):
    _plot_violin(sdata, **kwargs)


def _plot_violin(sdata, category, value="PREDICTIONS", title="Distribution of Predictions", ylab="Predicted Activity", **kwargs):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        sns.violinplot(data=sdata.seqs_annot, x=category, y=value, ax=ax, **kwargs)
        ax.set_xlabel(category, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.set_title(title, fontsize=24)
        plt.tight_layout()


def boxplot(sdata, **kwargs):
    _plot_boxplot(sdata, **kwargs)


def _plot_boxplot(sdata, category, value="PREDICTIONS", title="Distribution of Predictions", ylab="Predicted Activity", **kwargs):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        sns.boxplot(data=sdata.seqs_annot, x=category, y=value, ax=ax, **kwargs)
        ax.set_xlabel(category, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.set_title(title, fontsize=24)
        plt.tight_layout()


def performance_summary(sdata, task, **kwargs):
    _plot_performance_summary(sdata, task, **kwargs)


def _plot_performance_summary(sdata, task, title="Performance Summary", **kwargs):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        if task == "classification":
            _plot_classification_summary(sdata, ax, **kwargs)
        elif task == "regression":
            _plot_regression_summary(sdata, ax, **kwargs)
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")
        ax.set_xlabel("", fontsize=20)
        ax.set_ylabel("", fontsize=20)
        ax.set_title(title, fontsize=24)
        plt.tight_layout()


def _plot_classification_summary(sdata, ax, target="TARGETS", prediction="PREDICTIONS", title="Classification Summary", xlab="", ylab="", threshold=0.5):
    ts = sdata.seqs_annot[target].values.reshape(-1, 1)
    ps = sdata.seqs_annot[prediction].values.reshape(-1, 1)
    fpr, tpr, _ = roc_curve(ts, ps)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(ts, ps)
    average_precision = average_precision_score(ts, ps)
    ax.plot(recall, precision, label=f"AUPRC curve (AP = {average_precision:.3f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.legend(loc="lower right")
    plt.tight_layout()


def _plot_regression_summary(sdata, ax, target="TARGETS", prediction="PREDICTIONS", title="Regression Summary", xlab="", ylab="", threshold=0.5):
    ts = sdata.seqs_annot[target].values.reshape(-1, 1)
    ps = sdata.seqs_annot[prediction].values.reshape(-1, 1)
    fpr, tpr, _ = roc_curve(ts, ps)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(ts, ps)
    average_precision = average_precision_score(ts, ps)
    ax.plot(recall, precision, label=f"AUPRC curve (AP = {average_precision:.3f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.legend(loc="lower right")
    plt.tight_layout()


def pca(sdata, seqsm_key, pc1=0, pc2=1, color="b", loadings=None, labels=None, n=5):
    pc_data = sdata.seqsm[seqsm_key]
    xs = pc_data[:, pc1]
    ys = pc_data[:, pc2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    ax = plt.scatter(xs * scalex, ys * scaley, c=color)
    if loadings is not None:
        if n > loadings.shape[0]:
            n = loadings.shape[0]
        for i in range(n):
            plt.arrow(0, 0, loadings[0, i], loadings[1, i], color='r', alpha=0.5, head_width=0.07, head_length=0.07, overhang=0.7)
        if labels is None:
            plt.text(loadings[0, i] * 1.2, loadings[1, i] * 1.2, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(loadings[0, i] * 1.2, loadings[1, i] * 1.2, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.show()
    return ax


def skree(sdata, uns_key="sklearn_pca_obj", n_comp=30):
    """
    Function to generate and output a Skree plot using matplotlib barplot
    Parameters
    ----------
    pca_obj : scikit-learn pca object
    n_comp : number of components to show in the plot

    Returns
    -------

    """
    variance={}
    for i,val in enumerate(sdata.uns[uns_key].explained_variance_ratio_.tolist()):
        key="PC"+str(i+1)
        variance[key]=val
    plt.bar(["PC"+str(i) for i in range(1,n_comp+1)],sdata.uns[uns_key].explained_variance_ratio_.tolist())
    plt.xticks(rotation=90)
    plt.ylabel("Variance Explained")
    plt.xlabel("Principal Component")
    return variance


def umap(sdata, seqsm_key, umap1=0, umap2=1, color="b", n=5):
    umap_data = sdata.seqsm[seqsm_key]
    xs = umap_data[:, umap1]
    ys = umap_data[:, umap2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    ax = plt.scatter(xs * scalex, ys * scaley, c=color)
    plt.xlabel("UMAP{}".format(1))
    plt.ylabel("UMAP{}".format(2))
    plt.show()
    return ax
