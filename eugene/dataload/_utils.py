from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Literal,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    cast,
)
import pandas as pd
import xarray as xr


def concat_sdatas(
    sdatas: Iterable[xr.Dataset],
    keys: Optional[List] = None,
) -> xr.Dataset:
    """Concatenate multiple SeqDatas into one.

    Adds a "batch" variable to concatenated SeqData along the "_sequence" dimension.
    Assumes that there is a _sequence dimension in each SeqData to concatenate on.
    If there is not it will raise an error.

    Parameters
    """
    for i, s in enumerate(sdatas):
        s["batch"] = keys[i]
    return xr.concat(sdatas, dim="_sequence")


def add_obs(
    sdata: xr.Dataset,
    obs: pd.DataFrame,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
) -> None:
    """Add observational metadata to a SeqData.

    Parameters
    ----------
    sdata : xr.Dataset
        The SeqData to add observations to.
    obs : pd.DataFrame
        The observations to add.
    on : str, optional
        The column name to join on. If not given, left_on and right_on must be given.
    left_on : str, optional
        The column name in the SeqData to join on. If not given, on must be given.
    right_on : str, optional
        The column name in the observations to join on. If not given, on must be given.
    
    Raises
    ------
    ValueError
        If on is not given and left_on or right_on are not given.
        If on is given and left_on or right_on are given.
    """
    if on is None and (left_on is None or right_on is None):
        raise ValueError
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError

    if on is None:
        assert left_on is not None
        assert right_on is not None
    else:
        left_on = on
        right_on = on

    sdata[left_on] = sdata[left_on].astype("U").astype("O")
    df = sdata[left_on].to_dataframe()
    merged_df = df.merge(obs, left_on=left_on, right_on=right_on, how="left")

    for col in merged_df.columns:
        if col == left_on:
            continue
        sdata[col] = xr.DataArray(merged_df[col].values, dims=["_sequence"])
