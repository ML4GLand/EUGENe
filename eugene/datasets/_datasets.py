from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import pyranges as pr

from .._compat import Literal
from .._settings import settings
from ._utils import check_datasetdir_exists, try_download_urls

from ..dataloading._io import read, read_csv
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
    data.pos_annot = pr.read_bed(f"{HERE}/test_1000seqs_66/test_seq_features.bed")
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

def deBoer20(datasets: list, **kwargs: dict) -> pd.DataFrame:
    """
    reads the deBoer20 dataset.
    """
    urls_list = [
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160503_average_promoter_ELs_per_seq_atLeast100Counts.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160609_average_promoter_ELs_per_seq_Abf1TATA_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160609_average_promoter_ELs_per_seq_pTpA_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20161024_average_promoter_ELs_per_seq_3p1E7_Gal_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20161024_average_promoter_ELs_per_seq_3p1E7_Gly_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20170811_average_promoter_ELs_per_seq_OLS_Glu_goodCores_ALL.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20180808_processed_Native80_and_N80_spikein.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_pTpA_random_design_tiling_etc_YPD_expression.txt.gz"
    ]

    if type(datasets) is int:
        datasets = [datasets]

    paths = try_download_urls(datasets, urls_list, "deBoer20", is_gz = True)

    seq_col="SEQ"
    target_col="TARGET"

    data = read_csv(paths, seq_col=seq_col, target_col=target_col, sep=",", col_names=[seq_col,target_col], **kwargs)
    return data
