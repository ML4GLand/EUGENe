from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from ..utils import track


@track
def pca(
    sdata, 
    uns_key: str, 
    n_comp: int = 30, 
    copy: bool = False
):
    """
    Function to perform scaling and PCA on an input matrix

    Parameters
    ----------
    sdata: SeqData
        SeqData object to pull data to dimensionally reduce
    uns_key: str
        Key in sdata.uns to pull data from
    n_comp: int
        Number of components to reduce to
    copy: bool
        Whether to copy the SeqData object or not
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


@track
def umap(
    sdata, 
    seqsm_key=None, 
    uns_key=None, 
    copy=False, 
    **kwargs
):
    """
    Function to perform scaling and UMAP on SeqData object data
    
    Parameters
    ----------
    sdata: SeqData
        SeqData object to pull data to dimensionally reduce
    seqsm_key: str
        Key in sdata.seqsm to pull data from
    uns_key: str
        Key in sdata.uns to store data in
    copy: bool
        Whether to copy the SeqData object or not
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
