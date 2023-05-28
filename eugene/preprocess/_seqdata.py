import os 
import sys
bin_dir = os.path.dirname(sys.executable)
os.environ["PATH"] += os.pathsep + bin_dir
from sklearn.model_selection import train_test_split
import xarray as xr
import seqpro as sp
import numpy as np

def make_unique_ids_sdata(
    sdata,
    id_var="id",
    copy=False
):
    """Make unique ids for each sequence in a sdata object."""
    sdata = sdata.copy() if copy else sdata
    n_digits = len(str(sdata.dims["_sequence"]))
    sdata[id_var] = xr.DataArray(["seq{num:0{width}}".format(num=i, width=n_digits)for i in range(sdata.dims["_sequence"])], dims=["_sequence"])
    return sdata if copy else None

def ohe_seqs_sdata(
    sdata,
    alphabet="DNA",
    var_key="ohe_seq",
    copy=False
):
    sdata = sdata.copy() if copy else sdata
    sdata[var_key] = xr.DataArray(sp.ohe(sdata["seq"].to_numpy(), sp.ALPHABETS[alphabet]), dims=['_sequence', "length", "_ohe"])
    return sdata if copy else None

def train_test_split_sdata(
    sdata, 
    train_key="train_val", 
    id_var="id",
    test_size=0.2,
    homology_threshold=None,
    shuffle=True,
    chr=None, 
    copy=False
):
    """
    Train test split ,object.

    Parameters
    ----------
    sdata : 
        object to split
    train_key : str, optional
        Key to use for train/val split, by default "train_val"
    chr : str, optional
        Chromosome to use for train/val split, by default None
    copy : bool, optional
        Whether to return a copy of th,object, by default False
    """
    sdata = sdata.copy() if copy else sdata
    if homology_threshold is not None:
        try:
            from graph_part import train_test_validation_split
        except ImportError:
            raise ImportError("Please install graph_part to use homology_treshold (https://github.com/graph-part/graph-part))")
        seqs = sdata["seq"].to_numpy().astype("U")
        ids = sdata["id"].to_numpy()
        seq_id_dict = dict(zip(ids, seqs))
        train_indeces, _, = train_test_validation_split(seq_id_dict, threshold=homology_threshold, test_size=test_size, alignment_mode="needle", nucleotide=True)
        sdata[train_key] = sdata[id_var].isin(train_indeces)
        return sdata if copy else None
    elif chr is not None:
        chr = [chr] if isinstance(chr, str) else chr
        sdata[train_key] = ~sdata["chr"].isin(chr)
        return sdata if copy else None
    else:
        train_indeces, _, = train_test_split(sdata[id_var], test_size=test_size, shuffle=shuffle)
        sdata[train_key] = sdata[id_var].isin(train_indeces)
        return sdata if copy else None
    