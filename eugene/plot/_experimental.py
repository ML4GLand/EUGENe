import matplotlib.pyplot as plt


def pca(
    sdata,
    seqsm_key,
    pc1=0,
    pc2=1,
    loadings=None,
    labels=None,
    n=5,
    return_axes=False,
    **kwargs
):
    """
    Plot the PCA of the data.

    Parameters
    ----------
    sdata :
        SeqData The SeqData object.
    seqsm_key :
    str The key of the SeqSM object to use.
    pc1 : int
        The first PC to plot.
    pc2 : int
        The second PC to plot.
    color : str
        The color of the points.
    loadings :
        list of floats The loadings of the PCs.
    labels :
        list of str The labels of the points.
    n :
        int The number of points to plot.

    Returns
    -------
    None
    """
    pc_data = sdata.seqsm[seqsm_key]
    xs = pc_data[:, pc1]
    ys = pc_data[:, pc2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    ax = plt.scatter(xs * scalex, ys * scaley, **kwargs)
    if loadings is not None:
        if n > loadings.shape[0]:
            n = loadings.shape[0]
        for i in range(n):
            plt.arrow(
                0,
                0,
                loadings[0, i],
                loadings[1, i],
                color="r",
                alpha=0.5,
                head_width=0.07,
                head_length=0.07,
                overhang=0.7,
            )
        if labels is None:
            plt.text(
                loadings[0, i] * 1.2,
                loadings[1, i] * 1.2,
                "Var" + str(i + 1),
                color="g",
                ha="center",
                va="center",
            )
        else:
            plt.text(
                loadings[0, i] * 1.2,
                loadings[1, i] * 1.2,
                labels[i],
                color="g",
                ha="center",
                va="center",
            )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    if return_axes:
        return ax


def skree(sdata, uns_key, n_comp=30, return_variance=False):
    """
    Function to generate and output a Skree plot using matplotlib barplot
    Parameters
    ----------
    pca_obj : scikit-learn pca object
    n_comp : number of components to show in the plot

    Returns
    -------

    """
    variance = {}
    for i, val in enumerate(sdata.uns[uns_key].explained_variance_ratio_.tolist()):
        key = "PC" + str(i + 1)
        variance[key] = val * 100
    plt.bar(
        ["PC" + str(i) for i in range(1, n_comp + 1)],
        sdata.uns[uns_key].explained_variance_ratio_ * 100,
    )
    plt.xticks(rotation=90)
    plt.ylabel("Variance Explained")
    plt.xlabel("Principal Component")
    if return_variance:
        return variance


def umap(sdata, seqsm_key, umap1=0, umap2=1, n=5, return_axes=False, **kwargs):
    """
    Plot the UMAP of the data.

    Parameters
    ----------
    sdata : SeqData
        The SeqData object.
    seqsm_key : str
        The key of the SeqSM object to use.
    umap1 : int
        The first UMAP to plot.
    umap2 : int
        The second UMAP to plot.
    color : str
        The color of the points.
    n : int
        The number of points to plot.

    Returns
    -------
    None
    """
    umap_data = sdata.seqsm[seqsm_key]
    xs = umap_data[:, umap1]
    ys = umap_data[:, umap2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    ax = plt.scatter(xs * scalex, ys * scaley, **kwargs)
    plt.xlabel("UMAP{}".format(1))
    plt.ylabel("UMAP{}".format(2))
    plt.show()
    if return_axes:
        return ax
