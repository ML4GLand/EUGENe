import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features_sdata(
    sdata, 
    indeces=None,
    features_cols = None,
    seqsm_key=None,
    train_key=None,
    scaler=None,
    store_scaler=True,
    suffix=False,
    copy=False,
):
    """
    Function to standardize features based on passed in indeces and optionally save stats

    Parameters
    ----------
    train_X: numpy.ndarray
        The training data
    test_X: numpy.ndarray
        The testing data
    indeces: list of int, optional
        The indeces of the features to standardize
    stats_file: str, optional
        The file to save the stats to
    """
    if features_cols is not None:
        sdata.seqsm[f"{model_name}_features"] = sdata.seqs_annot[feature_cols].values
        seqsm_key = f"{model_name}_features"
    else:
        assert seqsm_key is not None, "Must provide either features_cols or seqsm_key"
    if train_key is not None:
        train_idx = np.where(sdata.seqs_annot[train_key] == True)[0]
        #print(len(train_idx), "training sequences")
        scale_data = sdata[train_idx].seqsm[seqsm_key].copy()
        #print(scale_data.shape, "data to create scale shape")
    else:
        scale_data = sdata.seqsm[seqsm_key].copy()
    to_scale = sdata.seqsm[seqsm_key].copy()
    #print(to_scale.shape, "data to scale shape")

    if indeces is None:
        indeces = np.array(range(scale_data.shape[1]))
    elif len(indeces) == 0:
        raise ValueError("No features to scale")
    else:
        scale_data = scale_data[:, indeces]
        to_scale = to_scale[:, indeces]
    ##print(scale_data.shape, "data to create scale shape")
    ##print(to_scale.shape, "data to scale shape")
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(scale_data)
    scaled = scaler.transform(to_scale)
    suffix = f"_{suffix}" if suffix else "scaled"
    sdata.seqsm[f"{seqsm_key}_{suffix}"] = sdata.seqsm[seqsm_key].copy()
    sdata.seqsm[f"{seqsm_key}_{suffix}"][:, indeces] = scaled
    #print(np.mean(sdata.seqsm[f"{seqsm_key}_{suffix}"][:, indeces], axis=0), "mean of scaled data")
    #print(np.std(sdata.seqsm[f"{seqsm_key}_{suffix}"][:, indeces], axis=0), "std of scaled data")
    if store_scaler:
        sdata.uns["scaler"] = scaler
    return sdata if copy else None