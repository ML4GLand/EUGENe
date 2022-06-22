# -*- coding: utf-8 -*-

"""
Python script with functions for preprocessing data
"""

# Basic imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


# Function definitions
def scaled_PCA(mtx, n_comp=30, index_name='index'):
    """
    Function to perform scaling and PCA on an input matrix

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
    scaler = StandardScaler()
    scaler.fit(mtx)
    mtx_scaled = scaler.transform(mtx)
    pca_obj = PCA(n_components=n_comp)
    pca_obj.fit(mtx_scaled)
    pca_df = pd.DataFrame(pca_obj.fit_transform(mtx_scaled))
    pca_df.columns = ['PC' + str(col+1) for col in pca_df.columns]
    pca_df.index = mtx.index
    pca_df.index.name = index_name
    return pca_obj, pca_df

# Function definitions
def scaled_UMAP(mtx, index_name='index'):
    """
    Function to perform scaling and UMAP on an input matrix

    Parameters
    ----------
    mtx : sample by feature
    index_name : name of index if part of input

    Returns
    umap object and umap dataframe
    -------

    """
    print("Make sure your matrix is sample by feature")
    scaler = StandardScaler()
    scaler.fit(mtx)
    mtx_scaled = scaler.transform(mtx)
    umap_obj = umap.UMAP()
    umap_obj.fit(mtx_scaled)
    umap_df = pd.DataFrame(umap_obj.transform(mtx_scaled))
    umap_df.columns = ['UMAP' + str(col+1) for col in umap_df.columns]
    umap_df.index = mtx.index
    umap_df.index.name = index_name
    return umap_obj, umap_df


