from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    cast,
)
import pandas as pd
import xarray as xr


def concat_seqdatas(seqdatas, keys):
    for i, s in enumerate(seqdatas):
        s["batch"] = keys[i]
    return xr.concat(seqdatas, dim="_sequence")


def add_obs(
    sdata: xr.Dataset,
    obs: pd.DataFrame,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
):
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

    # Add each column of the merged_df back to the xarray
    for col in merged_df.columns:
        if col == left_on:
            continue
        sdata[col] = xr.DataArray(merged_df[col].values, dims=["_sequence"])
