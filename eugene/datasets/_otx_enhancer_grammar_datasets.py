from pathlib import Path
import pandas as pd
import pyranges as pr
from .._compat import Literal
from ..dataloading._io import read_csv
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


def ols(filename, **kwargs: dict) -> pd.DataFrame:
    """
    reads the OLS dataset.
    """
    sdataframe = read_csv(
        filename,
        return_dataframe=True, 
        auto_name=False
    )
    sdata = SeqData(seqs=sdataframe["SEQ"], names=sdataframe["NAME"], seqs_annot=sdataframe[["ACTIVITY_SUMRNA_NUMDNA", "SEQ_LEN", "MICROSCOPE_FXN"]])
    sdata.seqs_annot.index = sdata.names
    sdata.seqs_annot.rename(columns={"ACTIVITY_SUMRNA_NUMDNA": "target"}, inplace=True)
    return sdata


def khoueiry10(filename, **kwargs: dict) -> pd.DataFrame:
    """
    Reads the Khoueiry10 dataset.
    """
    sdata = read_csv(filename, target_col="FXN_LABEL", name_col="NAME")
    return sdata


def gata_ets_clusters(filename, **kwargs: dict) -> pd.DataFrame:
    """
    Reads the Khoueiry10 dataset.
    """
    sdata = read_csv(filename, target_col="FXN_LABEL", name_col="NAME")
    return sdata


def exact_syntax_match(filename, **kwargs: dict) -> pd.DataFrame:
    """
    reads the OLS dataset.
    """
    sdataframe = read_csv(
        filename,
        return_dataframe=True, 
        auto_name=False
    )
    sdata = SeqData(seqs=sdataframe["SEQ"], names=sdataframe["NAME"], seqs_annot=sdataframe[["FXN_LABEL", "FXN_DESCRIPTION", "SEQ_LEN"]])
    sdata.seqs_annot.index = sdata.names
    return sdata