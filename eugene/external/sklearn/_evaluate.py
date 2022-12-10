import os
import numpy as np
import pandas as pd
from ... import settings
import threading

def predictions(
    model,
    sdata,
    target_keys,
    features_cols = None,
    seqsm_key = None,
    probability=True,
    threads=None,
    store_only=False,
    out_dir=None,
    name=None,
    version="",
    file_label="",
    prefix="",
    suffix="",
    copy=False
):
    threads = threads if threads is not None else threading.active_count()
    target_keys = [target_keys] if type(target_keys) == str else target_keys
    out_dir = out_dir if out_dir is not None else settings.output_dir
    model_name = model.__class__.__name__
    name = name if name is not None else model_name
    out_dir = os.path.join(out_dir, name, version)
    
    if features_cols is not None:
        sdata.seqsm[f"{model_name}_features"] = sdata.seqs_annot[feature_cols].values
        seqsm_key = f"{model_name}_features"
    else:
        assert seqsm_key is not None
    features = sdata.seqsm[seqsm_key]
    if probability:
        ps = model.predict_proba(features)[:, 1].squeeze()
        print(ps.shape)
    else:
        ps = model.predict(features)
    ts = sdata.seqs_annot[target_keys].values.squeeze()
    inds = sdata.seqs_annot.index
    new_cols = [f"{prefix}{lab}_predictions{suffix}" for lab in target_keys]
    sdata.seqs_annot[new_cols] = np.expand_dims(ps, axis=1)
    if not store_only:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df = pd.DataFrame(index=inds, data={"targets": ts, "predictions": ps})
        df.to_csv(os.path.join(out_dir, f"{file_label}_predictions.tsv"), sep="\t")
    return sdata if copy else None


def train_val_predictions(
    model,
    sdata,
    target_keys,
    features_cols = None,
    seqsm_key = None,
    probability=True,
    threads=None,
    store_only=False,
    out_dir=None,
    name=None,
    version="",
    file_label="",
    prefix="",
    suffix="",
    copy=False
):
    threads = threads if threads is not None else threading.active_count()
    target_keys = [target_keys] if type(target_keys) == str else target_keys
    out_dir = out_dir if out_dir is not None else settings.output_dir
    model_name = model.__class__.__name__
    name = name if name is not None else model_name
    out_dir = os.path.join(out_dir, name, version)