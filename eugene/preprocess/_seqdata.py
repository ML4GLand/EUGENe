import os
import sys
from typing import Optional

bin_dir = os.path.dirname(sys.executable)
os.environ["PATH"] += os.pathsep + bin_dir
import dask.array as da
import dask_ml as dml
import numpy as np
import seqpro as sp
import xarray as xr
from sklearn.preprocessing import StandardScaler


def make_unique_ids_sdata(sdata, id_var="id", copy=False):
    """Make unique ids for each sequence in a sdata object."""
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
    sdata, length, seq_key="seq", pad="right", pad_value="N", copy=False
):
    """Pad sequences in a SeqData object."""
    sdata = sdata.copy() if copy else sdata
    padded_seqs = sp.pad_seqs(
        seqs=sdata["seq"].values, pad=pad, pad_value=pad_value, length=length
    )
    sdata[f"{seq_key}_padded"] = xr.DataArray(padded_seqs, dims=["_sequence", "length"])
    return sdata if copy else None


def ohe_seqs_sdata(
    sdata, alphabet="DNA", seq_key="seq", ohe_key="ohe_seq", fill_value=0, copy=False
):
    sdata = sdata.copy() if copy else sdata
    ohe_seqs = sp.ohe(sdata[seq_key].values, sp.ALPHABETS[alphabet])
    if fill_value != 0:
        ohe_seqs = ohe_seqs.astype(type(fill_value))
        ohe_seqs[(ohe_seqs == 0).all(-1)] = np.array(
            np.repeat(fill_value, ohe_seqs.shape[-1]), dtype=type(fill_value)
        )
    sdata[ohe_key] = xr.DataArray(ohe_seqs, dims=["_sequence", "length", "_ohe"])

    return sdata if copy else None


def train_test_chrom_split(
    sdata: xr.Dataset, test_chroms: list[str], train_key="train_val"
):
    """Add a variable labeling sequences as part of the train or test split based on chromosome.

    Parameters
    ----------
    sdata : xr.Dataset
    test_chroms : list[str]
        List of chromosomes to put into test split.
    train_key : str, optional
        Name of the variable holding the labels such that True = train and False = test, by default "train_val"
    """
    train_mask = (~sdata.chrom.isin(test_chroms)).compute()
    sdata[train_key] = train_mask


def train_test_random_split(
    sdata: xr.Dataset,
    dim: str,
    train_key="train_val",
    groups=None,
    test_size=0.1,
    random_state: Optional[int] = None,
):
    """Add a variable labeling sequences as part of the train or test split, splitting randomly.

    Parameters
    ----------
    sdata : xr.Dataset
    dim : str
        Dimension to split randomly.
    train_key : str, optional
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
    sdata[train_key] = train_mask


def train_test_homology_split(
    sdata: xr.Dataset,
    seq_var: str,
    train_key="train_val",
    test_size=0.1,
    nucleotide=True,
):
    """Add a variable labeling sequences as part of the train or test split, splitting by homology.

    Parameters
    ----------
    sdata : xr.Dataset
    seq_var : str
        Variable containing the sequences.
    train_key : str, optional
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
    sdata[train_key] = train_group


def clamp_targets_sdata(
    sdata,
    target_keys: list,
    percentile: float = 0.995,
    train_key: str = None,
    clamp_nums: list = None,
    store_clamp_nums=False,
    suffix=False,
    copy=False,
):
    """
    Clamp targets to a given percentile if they are above that percentile in a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_keys : list
        List of target keys to clamp.
    percentile : float, optional
        Percentile to clamp to, by default 0.995
    train_key : str, optional
        Key to use if you only want to calculate percentiles on training data, by default None
    clamp_nums : list, optional
        You can provide numbers to clamp to, by default None
    store_clamp_nums : bool, optional
        Whether to store the clamp numbers in the SeqData object, by default False
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        SeqData object with clamped targets. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    if type(target_keys) is str:
        target_keys = [target_keys]
    if clamp_nums is None:
        if train_key is not None:
            train_idx = np.where(sdata["train_val"])[0]
            clamp_nums = (
                sdata.isel(_sequence=train_idx)[target_keys]
                .to_pandas()
                .quantile(percentile)
            )
        else:
            clamp_nums = sdata[target_keys].to_pandas().quantile(percentile)
    else:
        assert len(clamp_nums) == len(target_keys)
    for target_key in target_keys:
        if suffix:
            sdata[f"{target_key}_clamped"] = xr.DataArray(
                sdata[target_key].to_pandas().clip(upper=clamp_nums[target_key]),
                dims=["_sequence"],
            )
        else:
            sdata[target_key].values = (
                sdata[target_key].to_pandas().clip(upper=clamp_nums[target_key])
            )
    if store_clamp_nums:
        sdata["clamp_nums"] = xr.DataArray(clamp_nums, dims=["_targets"])
    return sdata if copy else None


def scale_targets_sdata(
    sdata,
    target_keys,
    train_key=None,
    scaler=None,
    return_scaler=True,
    suffix=False,
    copy=False,
):
    """
    Scale targets in a SeqData object.
    """
    sdata = sdata.copy() if copy else sdata
    if type(target_keys) is str:
        target_keys = [target_keys]
    if train_key is not None:
        scale_data = sdata.isel(_sequence=np.where(sdata[train_key])[0])[
            target_keys
        ].to_pandas()
    else:
        scale_data = sdata[target_keys].to_pandas()
    to_scale = sdata[target_keys].to_pandas()
    if len(target_keys) == 1:
        scale_data = scale_data.values.reshape(-1, 1)
        to_scale.values.reshape(-1, 1)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(scale_data)
    assert scaler.n_features_in_ == len(target_keys)
    to_scale = scaler.transform(to_scale)
    for i, target_key in enumerate(target_keys):
        if suffix:
            sdata[f"{target_key}_scaled"] = xr.DataArray(
                to_scale[:, i], dims=["_sequence"]
            )
        else:
            sdata[target_key].values = to_scale[:, i]
    if return_scaler and copy:
        return scaler, sdata
    elif return_scaler and not copy:
        return scaler
    else:
        return sdata if copy else None
