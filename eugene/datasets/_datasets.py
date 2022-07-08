from pathlib import Path
from typing import Optional
import warnings
import wget, os, gzip, io

import numpy as np
import pandas as pd

from .._compat import Literal
from .._settings import settings
from ._utils import check_datasetdir_exists

from ..dataloading._io import read
HERE = Path(__file__).parent
pkg_resources = None

def try_download_urls(data_idxs: list, url_list: list, ds_name: str, is_gz: bool = False) -> list:
    paths = []
    for i in data_idxs:
        csv_name = os.path.basename(url_list[i]).split(".")[0] + ".csv"
        if not os.path.exists(os.path.join(HERE.parent, settings.datasetdir, ds_name, csv_name)):
            ds_path = os.path.join(HERE.parent, settings.datasetdir, ds_name)
            if not os.path.isdir(ds_path):
                print(f"Path {ds_path} does not exist, creating new folder.")
                os.mkdir(ds_path)

            print(f"Downloading {ds_name} {os.path.basename(url_list[i])} to {ds_path}...")
            path = wget.download(url_list[i], os.path.relpath(ds_path))
            print(f"Finished downloading {os.path.basename(url_list[i])}")

            if is_gz:
                with gzip.open(path) as gz:
                    with io.TextIOWrapper(gz, encoding="utf-8") as file:
                        file = pd.read_csv(file, delimiter=r"\t", engine="python")

                        if ds_name == "deBoer20":
                            file = deBoerCleanup(file, i)

                        save_path = os.path.join(ds_path, csv_name)
                        file.to_csv(save_path, index = False)
                        print(f"Saved csv file to {save_path}")
                        paths.append(save_path)
                os.remove(os.path.join(ds_path, os.path.basename(url_list[i])))
            else:
                # If file is not packed, same logic but without gzip
                pass
        else:
            print(f"Dataset {ds_name} {csv_name} has already been dowloaded.")
            paths.append(os.path.join(HERE.parent, settings.datasetdir, ds_name, csv_name))
    return paths

def deBoerCleanup(file: pd.DataFrame, index: int) -> pd.DataFrame:
    return file

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
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20180808_processed_Native80_and_N80_spikein.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_pTpA_random_design_tiling_etc_YPD_expression.txt.gz"
    ]

    paths = try_download_urls(datasets, urls_list, "deBoer20", is_gz = True)

    print(paths)