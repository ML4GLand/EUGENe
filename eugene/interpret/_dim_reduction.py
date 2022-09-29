import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def pca(sdata, uns_key, n_comp=30, copy=False):
    """Function to perform scaling and PCA on an input matrix

    Parameters
    ----------
    mtx : sample by feature
    n_comp : number of pcs to return
    index_name : name of index if part of input

    Returns
    sklearn pca object and pca dataframe
    -------

    """
    print("Make sure your matrix is sample by feature")
    sdata = sdata.copy if copy else sdata
    mtx = sdata.uns[uns_key]
    if len(mtx.shape) == 3:
        mtx = mtx.max(axis=1)
    scaler = StandardScaler()
    scaler.fit(mtx)
    mtx_scaled = scaler.transform(mtx)
    pca_obj = PCA(n_components=n_comp)
    pca_obj.fit(mtx_scaled)
    sdata.seqsm[f"{uns_key}_pca"] = pca_obj.fit_transform(mtx_scaled)
    sdata.uns[f"{uns_key}_pca"] = pca_obj
    return sdata if copy else None


def umap(sdata, seqsm_key=None, uns_key=None, copy=False, **kwargs):
    """Function to perform scaling and UMAP on an input matrix

    Parameters
    ----------
    mtx : sample by feature
    index_name : name of index if part of input

    Returns
    umap object and umap dataframe
    -------

    """
    print("Make sure your matrix is sample by feature")
    sdata = sdata.copy if copy else sdata
    if seqsm_key is not None:
        mtx = sdata.seqsm[seqsm_key]
    elif uns_key is not None:
        mtx = sdata.uns[uns_key]
    else:
        raise ValueError("Must specify either seqsm_key or uns_key")
    if len(mtx.shape) == 3:
        mtx = mtx.max(axis=1)
    scaler = StandardScaler()
    scaler.fit(mtx)
    mtx_scaled = scaler.transform(mtx)
    umap_obj = UMAP(**kwargs)
    umap_obj.fit(mtx_scaled)
    sdata.seqsm[f"{uns_key}_umap"] = umap_obj.fit_transform(mtx_scaled)
    sdata.uns[f"{uns_key}_umap"] = umap_obj
    return sdata if copy else None
