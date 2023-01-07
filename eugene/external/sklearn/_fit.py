import os
import numpy as np
import pandas as pd
import threading
from ... import settings

def fit(
    model,
    sdata,
    target_keys,
    train_key="train_val",
    features_cols = None,
    seqsm_key = None,
    threads=None,
    log_dir=None,
    name=None,
    version="",
    seed=None,
    verbosity=1
):
    # Set-up the run
    threads = threads if threads is not None else threading.active_count()
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    model_name = model.__class__.__name__
    name = name if name is not None else model_name
    np.random.seed(seed) if seed is not None else np.random.seed(settings.seed)
    model.verbose = verbosity

    # Remove seqs with NaN targets
    targs = sdata.seqs_annot[target_keys].values  
    if len(targs.shape) == 1:
        nan_mask = np.isnan(targs)
    else:
        nan_mask = np.any(np.isnan(targs), axis=1)
    print(f"Dropping {nan_mask.sum()} sequences with NaN targets.")
    sdata = sdata[~nan_mask]
    targs = targs[~nan_mask]

    
    # Get train and val indeces
    train_idx = np.where(sdata.seqs_annot[train_key] == True)[0]
    val_idx = np.where(sdata.seqs_annot[train_key] == False)[0]
    
    # Get train and val targets
    train_Y = targs[train_idx].squeeze()
    val_Y = targs[val_idx].squeeze()
    
    # Get train and val features
    if features_cols is not None:
        sdata.seqsm[f"{model_name}_features" if seqsm_key is None else seqsm_key] = sdata.seqs_annot[feature_cols].values
    else:
        assert seqsm_key is not None
    features = sdata.seqsm[seqsm_key]
    train_X = features[train_idx]
    model.fit(train_X, train_Y)
    
    if not os.path.exists(os.path.join(log_dir, name, version)):
        os.makedirs(os.path.join(log_dir, name, version))
        
    pd.DataFrame(pd.Series(model.get_params())).T.to_csv(os.path.join(log_dir, name, version, "hyperparams.tsv"), index=False, sep="\t")



