import os
import sys
from typing import Any, Dict, Generic, List, Literal, Optional, Type, Union, cast

import dask.array as da
import dask_ml as dml
import numpy as np
import seqpro as sp
import xarray as xr
from sklearn.preprocessing import StandardScaler

bin_dir = os.path.dirname(sys.executable)
os.environ["PATH"] += os.pathsep + bin_dir

alphabets ={
    "DNA": sp.alphabets.DNA,
    "RNA": sp.alphabets.RNA,
}


def make_unique_ids_sdata(
    sdata: xr.Dataset,
    id_var: str = "id",
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """Make a set of unique ids for each sequence in a SeqData object and store as new xarray variable. 

    Expects that the dimension for the number of sequences is named "_sequence". Otherwise,
    it will fail. Will also overwrite any existing variable with the same name.

    Parameters
    ----------
    sdata : xr.Dataset
        SeqData object.
    id_var : str, optional
        Name of the variable to store the ids in, by default "id"
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    xr.Dataset
        SeqData object with unique ids. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    n_digits = len(str(sdata.dims["_sequence"]))
    sdata[id_var] = xr.DataArray(
        [
            "seq{num:0{width}}".format(num=i, width=n_digits)
            for i in range(sdata.dims["_sequence"])
        ],
        dims=["_sequence"],
    )
    return sdata if copy else None


def pad_seqs_sdata(
    sdata: xr.Dataset,
    length: int,
    seq_var: str = "seq", 
    pad: Literal["left", "both", "right"] = "right",
    pad_value: Optional[str] = None,
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """Pad sequences in a SeqData object.
    
    Wraps the pad_seqs function from SeqPro on the sequences in a SeqData object. Automatically
    adds a new variable to the SeqData object with the padded sequences called "{seq_var}_padded".
    Assumes that the dimension for the number of sequences is named "_sequence" and will add dimension
    called length to the padded sequences. Will also overwrite any existing variable with the same name.

    Parameters
    ----------
    sdata : xr.Dataset
        SeqData object.
    length : int
        Length to pad or truncate sequences to.
    seq_var : str, optional
        Name of the variable holding the sequences, by default "seq"
    pad : Literal["left", "both", "right"], optional
        How to pad. If padding on both sides and an odd amount of padding is needed, 1
        more pad value will be on the right side, by default "right"
    pad_val : str, optional
        Single character to pad sequences with. Needed for string input. Ignored for OHE
        sequences, by default None
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    xr.Dataset
        SeqData object with padded sequences. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    padded_seqs = sp.pad_seqs(seqs=sdata["seq"].values, pad=pad, pad_value=pad_value, length=length)
    sdata[f"{seq_var}_padded"] = xr.DataArray(padded_seqs, dims=["_sequence", "length"])
    return sdata if copy else None


def ohe_seqs_sdata(
    sdata: xr.Dataset,
    alphabet: str = "DNA",
    seq_var: str = "seq",
    ohe_var: str = "ohe_seq",
    fill_value: Union[int, float] = 0,
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """One-hot encode sequences in a SeqData object.
    
    Wraps the ohe function from SeqPro on the sequences in a SeqData object. Automatically
    adds a new variable to the SeqData object with the one-hot encoded sequences called "ohe_seq".
    with dimensions ()"_sequence", "length", "_ohe"). Will also overwrite any existing variable
    with the same name.

    Parameters
    ----------
    sdata : xr.Dataset
        SeqData object.
    alphabet : str, optional
        Alphabet to use for one-hot encoding, by default "DNA"
    seq_var : str, optional
        Name of the variable holding the sequences to be encoded, by default "seq"
    ohe_var : str, optional
        Name of the variable to store the one-hot encoded sequences in, by default "ohe_seq"
    fill_value : Union[int, float], optional
        Value to fill the one-hot encoded sequences with, by default 0
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False
    
    Returns
    -------
    xr.Dataset
        SeqData object with one-hot encoded sequences. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    ohe_seqs = sp.ohe(sdata[seq_var].values, alphabet=alphabets[alphabet])
    if fill_value != 0:
        ohe_seqs = ohe_seqs.astype(type(fill_value))
        ohe_seqs[(ohe_seqs == 0).all(-1)] = np.array(np.repeat(fill_value, ohe_seqs.shape[-1]), dtype=type(fill_value))
    sdata[ohe_var] = xr.DataArray(ohe_seqs, dims=["_sequence", "length", "_ohe"])
    return sdata if copy else None


def train_test_chrom_split(
    sdata: xr.Dataset, 
    test_chroms: List[str],
    train_var: str = "train_val",
):
    """Add a variable labeling sequences as part of the train or test split based on chromosome.

    Parameters
    ----------
    sdata : xr.Dataset
        SeqData object.
    test_chroms : list[str]
        List of chromosomes to put into test split.
    train_var : str, optional
        Name of the variable holding the labels such that True = train and False = test, by default "train_val"
    """
    train_mask = (~sdata.chrom.isin(test_chroms)).compute()
    sdata[train_var] = train_mask


def train_test_random_split(
    sdata: xr.Dataset,
    dim: str,
    train_var: str = "train_val",
    groups: Optional[Any] = None,
    test_size: float = 0.1,
    random_state: Optional[int] = None,
):
    """Add a variable labeling sequences as part of the train or test split, splitting randomly.

    Parameters
    ----------
    sdata : xr.Dataset
        SeqData object.
    dim : str
        Dimension to split randomly.
    train_var : str, optional
        Name of the variable holding the labels such that True = train and False = test, by default "train_val"
    groups : ArrayLike, optional
        Groups to stratify the splits by, by default None
    test_size : float, optional
        Proportion of data to put in the test set, by default 0.1
    random_state : int, optional
        Random seed, by default None
    """
    splitter = dml.model_selection.ShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(
        splitter.split(da.arange(sdata.sizes[dim]), groups=groups)
    )
    train_mask = np.full(sdata.sizes[dim], False)
    train_mask[train_idx] = True
    sdata[train_var] = xr.DataArray(train_mask, dims=[dim])


def train_test_homology_split(
    sdata: xr.Dataset,
    seq_var: str,
    train_var: str = "train_val",
    test_size: float = 0.1,
    nucleotide: bool = True,
):
    """Add a variable labeling sequences as part of the train or test split, splitting by homology.

    Parameters
    ----------
    sdata : xr.Dataset
        SeqData object.
    seq_var : str
        Variable containing the sequences.
    train_var : str, optional
        Name of the variable holding the labels, by default "train_val"
    test_size : float, optional
        Proportion of data to put in the test set, by default 0.1
    nucleotide : bool, optional
        Whether the input sequences are nucleotides or not, by default True

    Raises
    ------
    ImportError
        If [graph-part](https://github.com/graph-part/graph-part) is not installed.
    """
    try:
        from graph_part import train_test_validation_split
    except ImportError:
        raise ImportError(
            "Install [graph-part](https://github.com/graph-part/graph-part) to split by homology."
        )
    seq_length = sdata.sizes[sdata.attrs["length_dim"]]
    seqs = (
        sdata[seq_var]
        .to_numpy()
        .view(f"S{seq_length}")
        .squeeze()
        .astype("U")
        .astype(object)
    )
    outs = train_test_validation_split(
        seqs,
        test_size=test_size,
        initialization_mode="fast-nn",
        nucleotide=nucleotide,
        prefilter=True,
        denominator="shortest",
    )
    train_idx, test_idx = map(np.array, outs)
    train_group = np.full(sdata.sizes[sdata.attrs["sequence_dim"]], "removed")
    train_group[train_idx] = "train"
    train_group[test_idx] = "val"
    sdata[train_var] = train_group


def clamp_targets_sdata(
    sdata: xr.Dataset,
    target_vars: Union[str, List[str]],
    percentile: float = 0.995,
    train_var: Optional[str] = None,
    clamp_nums: Optional[List[float]] = None,
    store_clamp_nums: bool = False,
    suffix: bool = False,
    copy: bool = False,
):
    """
    Clamp targets to a given percentile in a SeqData object.

    Parameters
    ----------
    sdata : xr.Dataset
        SeqData object.
    target_vars : list
        List of target variables to clamp.
    percentile : float, optional
        Percentile to clamp to, by default 0.995
    train_var : str, optional
        Key to use if you only want to calculate percentiles on training data, by default None
    clamp_nums : list, optional
        You can provide numbers to clamp to, by default None
    store_clamp_nums : bool, optional
        Whether to store the clamp numbers in the SeqData object, by default False
    suffix : bool, optional
        Whether to add a suffix to the variable name, by default False
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        SeqData object with clamped targets. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    if type(target_vars) is str:
        target_vars = [target_vars]
    if clamp_nums is None:
        if train_var is not None:
            train_idx = np.where(sdata["train_val"])[0]
            clamp_nums = (
                sdata.isel(_sequence=train_idx)[target_vars]
                .to_pandas()
                .quantile(percentile)
            )
        else:
            clamp_nums = sdata[target_vars].to_pandas().quantile(percentile)
    else:
        assert len(clamp_nums) == len(target_vars)
    for target_var in target_vars:
        if suffix:
            sdata[f"{target_var}_clamped"] = xr.DataArray(
                sdata[target_var].to_pandas().clip(upper=clamp_nums[target_var]),
                dims=["_sequence"],
            )
        else:
            sdata[target_var].values = (
                sdata[target_var].to_pandas().clip(upper=clamp_nums[target_var])
            )
    if store_clamp_nums:
        sdata["clamp_nums"] = xr.DataArray(clamp_nums, dims=["_targets"])
    return sdata if copy else None


def scale_targets_sdata(
    sdata: xr.Dataset,
    target_vars: Union[str, List[str]],
    train_var: Optional[str] = None,
    scaler: Optional[StandardScaler] = None,
    return_scaler: bool = False,
    suffix: bool = False,
    copy: bool = False,
):
    """
    Scale targets in a SeqData object.
    """
    sdata = sdata.copy() if copy else sdata
    if type(target_vars) is str:
        target_vars = [target_vars]
    if train_var is not None:
        scale_data = sdata.isel(_sequence=np.where(sdata[train_var])[0])[
            target_vars
        ].to_pandas()
    else:
        scale_data = sdata[target_vars].to_pandas()
    to_scale = sdata[target_vars].to_pandas()
    if len(target_vars) == 1:
        scale_data = scale_data.values.reshape(-1, 1)
        to_scale.values.reshape(-1, 1)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(scale_data)
    assert scaler.n_features_in_ == len(target_vars)
    to_scale = scaler.transform(to_scale)
    for i, target_var in enumerate(target_vars):
        if suffix:
            sdata[f"{target_var}_scaled"] = xr.DataArray(
                to_scale[:, i], dims=["_sequence"]
            )
        else:
            sdata[target_var].values = to_scale[:, i]
    if return_scaler and copy:
        return scaler, sdata
    elif return_scaler and not copy:
        return scaler
    else:
        return sdata if copy else None
