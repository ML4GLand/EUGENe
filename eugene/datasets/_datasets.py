from pathlib import Path
import pandas as pd
import numpy as np
import pyranges as pr
from .._compat import Literal
from ._utils import try_download_urls

from ..dataloading._io import read, read_csv
from ..dataloading import SeqData

HERE = Path(__file__).parent
pkg_resources = None

def get_dataset_info():
    """
    Return DataFrame with info about builtin datasets.
    Returns
        Info about builtin datasets indexed by dataset name as dataframe.
    """
    global pkg_resources
    if pkg_resources is None:
        import pkg_resources
    stream = pkg_resources.resource_stream(__name__, "datasets.csv")
    return pd.read_csv(stream, index_col=0)


def random1000(binary=False, **kwargs: dict) -> pd.DataFrame:
    """
    Reads the random1000 dataset.
    """
    filename = f"{HERE}/random1000/random1000_seqs.tsv"
    if binary:
        sdata = read(filename, seq_col="SEQ", name_col="NAME", target_col="LABEL", **kwargs)
    else:
        sdata = read(filename, seq_col="SEQ", name_col="NAME", target_col="ACTIVITY", **kwargs)
    sdata.pos_annot = pr.read_bed(f"{HERE}/random1000/random1000_pos_annot.bed")
    return sdata


def farley15(binary=False, **kwargs: dict) -> pd.DataFrame:
    """
    Reads the Farley15 dataset.
    """
    urls_list = ["https://zenodo.org/record/6863861/files/farley2015_seqs.csv?download=1",
                 "https://zenodo.org/record/6863861/files/farley2015_seqs_annot.csv?download=1"]

    paths = try_download_urls([0,1], urls_list, "farley15", compression = "")
    path = paths[0]

    seq_col="Enhancer"
    if binary:
        raise NotImplementedError("Farley15 dataset is not yet implemented for non-binary data.")
    else:
        data = read_csv(path, sep=",", seq_col=seq_col, auto_name=True, return_dataframe=True, **kwargs)
        n_digits = len(str(len(data)-1))
        ids = np.array(["seq{num:0{width}}".format(num=i, width=n_digits) for i in range(len(data))])
        sdata = SeqData(seqs=data[seq_col], names=ids, seqs_annot=data[["Barcode", "Biological Replicate 1 (RPM)", "Biological Replicate 2 (RPM)"]])
    return sdata


def deBoer20(datasets: list, binary=False, **kwargs: dict) -> pd.DataFrame:
    """
    Reads the deBoer20 dataset.
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

    paths = try_download_urls(datasets, urls_list, "deBoer20", compression = "gz")

    seq_col="SEQ"
    target_col="TARGET"
    if binary:
        raise NotImplementedError("deBoer20 dataset is not yet implemented for non-binary data.")
    else:
        data = read_csv(paths, sep=",", seq_col=seq_col, target_col=target_col, col_names=[seq_col,target_col], auto_name=True, compression="gzip", **kwargs)
    return data
