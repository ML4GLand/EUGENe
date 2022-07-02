from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from .._compat import Literal
from .._settings import settings
from ._utils import check_datasetdir_exists

from ..dataloading._io import read
HERE = Path(__file__).parent
pkg_resources = None

def get_dataset_info():
    """Return DataFrame with info about builtin datasets.
    Returns
    -------
    DataFrame
        Info about builtin datasets indexed by dataset name.
    """
    global pkg_resources
    if pkg_resources is None:
        import pkg_resources


    stream = pkg_resources.resource_stream(__name__, "datasets.csv")
    return pd.read_csv(stream, index_col=0)


def random1000(**kwargs: dict) -> pd.DataFrame:
    """
    reads the random1000 dataset.
    """
    filename = f"{HERE}/test_1000seqs_66/test_seqs.tsv"
    data = read(filename, seq_col="SEQ", name_col="NAME", target_col="LABEL", **kwargs)
    return data


#@check_datasetdir_exists
def ols(**kwargs: dict) -> pd.DataFrame:
    """
    reads the OLS dataset.
    """
    filename = "/cellar/users/aklie/projects/EUGENE/data/2021_OLS_Library/2021_OLS_Library.tsv"
    data = read(filename, seq_col="SEQ", name_col="NAME", target_col="ACTIVITY_SUMRNA_NUMDNA", **kwargs)
    return data

def binary_ols(**kwargs: dict) -> pd.DataFrame:
    """
    reads the OLS dataset.
    """
    filename = "/cellar/users/aklie/projects/EUGENE/data/2021_OLS_Library/2021_OLS_Library.tsv"
    data = read(filename, seq_col="SEQ", name_col="NAME", target_col="ACTIVITY_SUMRNA_NUMDNA", binarize=True, **kwargs)
    return data

def Khoueiry10(**kwargs: dict) -> pd.DataFrame:
    """
    reads the Khoueiry10 dataset.
    """
    filename = "/cellar/users/aklie/projects/EUGENE/data/2010_Khoueiry_CellPress/2010_Khoueiry_CellPress.tsv"
    data = read(filename, seq_col="SEQ", name_col="NAME", target_col="FXN_LABEL", **kwargs)
    return data
