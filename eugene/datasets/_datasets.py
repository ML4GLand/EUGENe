from pathlib import Path
import pandas as pd
import numpy as np
import pyranges as pr
from .._compat import Literal
from ._utils import try_download_urls
from ..dataloading._io import read, read_csv, read_fasta
from ..dataloading import SeqData


HERE = Path(__file__).parent
pkg_resources = None


def get_dataset_info():
    """Return DataFrame with info about builtin datasets.

    Returns
    -------
    df : pd.DataFrame
        Info about builtin datasets indexed by dataset name as dataframe.
    """
    global pkg_resources
    if pkg_resources is None:
        import pkg_resources
    stream = pkg_resources.resource_stream(__name__, "datasets.csv")
    return pd.read_csv(stream, index_col=0)


def random1000(binary=False, **kwargs: dict) -> pd.DataFrame:
    """Reads the random1000 dataset.

    Parameters
    ----------
    binary : bool, optional
        If True, return binary target data. The default is False.

    Returns
    -------
    sdata : SeqData
        SeqData object with the random1000 dataset.
    """
    filename = f"{HERE}/random1000/random1000_seqs.tsv"
    if binary:
        sdata = read(
            filename, seq_col="SEQ", name_col="NAME", target_col="LABEL", **kwargs
        )
    else:
        sdata = read(
            filename, seq_col="SEQ", name_col="NAME", target_col="ACTIVITY", **kwargs
        )
    sdata.pos_annot = pr.read_bed(f"{HERE}/random1000/random1000_pos_annot.bed")
    return sdata


def random1000_10(binary=False, **kwargs: dict) -> pd.DataFrame:
    """Reads the random1000_10 dataset.

    Parameters
    ----------
    binary : bool, optional
        If True, return binary target data. The default is False.

    Returns
    -------
    sdata : SeqData
        SeqData object with the random1000_10 dataset.
    """
    filename = f"{HERE}/random1000_10/random1000_10_seqs.tsv"
    if binary:
        raise NotImplementedError(
            "random1000_10 dataset does not need to support binary data."
        )
    else:
        sdataframe = read(filename, return_dataframe=True)
        n_digits = len(str(len(sdataframe) - 1))
        ids = np.array(
            [
                "seq{num:0{width}}".format(num=i, width=n_digits)
                for i in range(len(sdataframe))
            ]
        )
        sdata = SeqData(
            seqs=sdataframe["SEQ"],
            names=ids,
            seqs_annot=sdataframe.drop(columns=["NAME", "SEQ"]),
        )
    return sdata


def farley15(return_sdata=True, **kwargs: dict) -> pd.DataFrame:
    """Reads the Farley15 dataset.

    Parameters
    ----------
    return_sdata : bool, optional
        If True, return SeqData object with the Farley15 dataset. The default is True.
    **kwargs : kwargs, dict
        Keyword arguments to pass to read_csv.

    Returns
    -------
    sdata : SeqData
        SeqData object with the Farley15 dataset.
    """
    urls_list = [
        "https://zenodo.org/record/6863861/files/farley2015_seqs.csv?download=1",
        "https://zenodo.org/record/6863861/files/farley2015_seqs_annot.csv?download=1",
    ]
    paths = try_download_urls([0, 1], urls_list, "farley15")
    if return_sdata:
        path = paths[0]
        seq_col = "Enhancer"
        data = read_csv(
            path,
            sep=",",
            seq_col=seq_col,
            auto_name=True,
            return_dataframe=True,
            **kwargs,
        )
        n_digits = len(str(len(data) - 1))
        ids = np.array(
            [
                "seq{num:0{width}}".format(num=i, width=n_digits)
                for i in range(len(data))
            ]
        )
        sdata = SeqData(
            seqs=data[seq_col],
            names=ids,
            seqs_annot=data[
                [
                    "Barcode",
                    "Biological Replicate 1 (RPM)",
                    "Biological Replicate 2 (RPM)",
                ]
            ],
        )
        return sdata
    else:
        return paths


def deBoer20(datasets: list, return_sdata=True, **kwargs: dict) -> pd.DataFrame:
    """Reads the deBoer20 dataset.

    Parameters
    ----------
    datasets : list of ints
        List of datasets indices to read.
    return_sdata : bool, optional
        If True, return SeqData object with the deBoer20 dataset. The default is True.
    **kwargs : kwargs, dict
        Keyword arguments to pass to read_csv.

    Returns
    -------
    sdata : SeqData
        SeqData object with the deBoer20 dataset.
    """
    urls_list = [
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160503_average_promoter_ELs_per_seq_atLeast100Counts.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160609_average_promoter_ELs_per_seq_Abf1TATA_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160609_average_promoter_ELs_per_seq_pTpA_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20161024_average_promoter_ELs_per_seq_3p1E7_Gal_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20161024_average_promoter_ELs_per_seq_3p1E7_Gly_ALL.shuffled.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20170811_average_promoter_ELs_per_seq_OLS_Glu_goodCores_ALL.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20180808_processed_Native80_and_N80_spikein.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_pTpA_random_design_tiling_etc_YPD_expression.txt.gz",
    ]

    if type(datasets) is int:
        datasets = [datasets]

    paths = try_download_urls(datasets, urls_list, "deBoer20", processing="gz")

    if return_sdata:
        seq_col = "seq"
        target_col = "target"
        sdata = read_csv(
            paths,
            sep=",",
            seq_col=seq_col,
            target_col=target_col,
            col_names=[seq_col, target_col],
            auto_name=True,
            compression="gzip",
            **kwargs,
        )
        return sdata
    else:
        return paths


def ray13(dataset="norm", return_sdata=True, **kwargs: dict) -> pd.DataFrame:
    """Reads the RNAcomplete dataset.

    Parameters
    ----------
    dataset : str
        Dataset to read, can either be "norm" or "raw". The default is "norm".
    return_sdata : bool, optional
        If True, return SeqData object with the RNAcomplete dataset. The default is True.
    **kwargs : kwargs, dict
        Keyword arguments to pass to read_csv.

    Returns
    -------
    sdata : SeqData
        SeqData object with the RNAcomplete dataset.
    """
    urls_list = [
        "http://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/norm_data.txt.gz",
        "http://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/raw_data.txt.gz",
    ]

    if dataset == "norm":
        dataset = [0]
    elif dataset == "raw":
        dataset = [1]
    else:
        raise ValueError("dataset must be either 'norm' or 'raw'.")

    paths = try_download_urls(dataset, urls_list, "ray13")

    if return_sdata:
        seq_col = "RNA_Seq"
        sdataframe = read_csv(
            paths,
            sep="\t",
            seq_col=seq_col,
            return_dataframe=True,
            compression="gzip",
            na_values=" NaN",
            **kwargs,
        )
        sdata = SeqData(
            seqs=sdataframe["RNA_Seq"],
            names=sdataframe["Probe_ID"],
            seqs_annot=sdataframe[sdataframe.columns.drop(pd.array(data=["RNA_Seq"]))],
        )
        sdata.seqs_annot.set_index("Probe_ID", inplace=True)
        return sdata
    else:
        return paths


def jores21(
    dataset="leaf", add_metadata=False, return_sdata=True, **kwargs: dict
) -> pd.DataFrame:
    """Reads the Jores21 dataset.

    Parameters
    ----------
    dataset : str, optional
        Dataset to read. Either leaf or proto. The default is "leaf".
    add_metadata : bool, optional
        If True, add metadata to the SeqData object. The default is False.
    return_sdata : bool, optional
        If True, return SeqData object with the Jores21 dataset. The default is True.
    **kwargs : kwargs, dict
        Keyword arguments to pass to read_csv.

    Returns
    -------
    sdata : SeqData
        SeqData object with the Jores21 dataset.
    """
    urls_list = [
        "https://raw.githubusercontent.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/main/CNN/CNN_test_leaf.tsv",
        "https://raw.githubusercontent.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/main/CNN/CNN_train_leaf.tsv",
        "https://raw.githubusercontent.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/main/CNN/CNN_train_proto.tsv",
        "https://raw.githubusercontent.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/main/CNN/CNN_test_proto.tsv",
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41477-021-00932-y/MediaObjects/41477_2021_932_MOESM3_ESM.xlsx",
    ]
    if dataset == "leaf":
        urls = [0, 1]
    elif dataset == "proto":
        urls = [2, 3]
    else:
        raise ValueError("dataset must be either 'leaf' or 'proto'.")
    paths = try_download_urls(urls, urls_list, "jores21")
    if return_sdata:
        seq_col = "sequence"
        data = read_csv(
            paths,
            sep="\t",
            seq_col=seq_col,
            auto_name=True,
            return_dataframe=True,
            **kwargs,
        )
        n_digits = len(str(len(data) - 1))
        ids = np.array(
            [
                "seq{num:0{width}}".format(num=i, width=n_digits)
                for i in range(len(data))
            ]
        )
        sdata = SeqData(
            seqs=data[seq_col],
            names=ids,
            seqs_annot=data[["set", "sp", "gene", "enrichment"]],
        )
        if add_metadata:
            metadata_path = try_download_urls(
                [4], urls_list, "jores21", compression=""
            )[0]
            smetadata = pd.read_excel(metadata_path, sheet_name=0, skiprows=3)
            sdata["sequence"] = sdata.seqs
            sdata.seqs_annot = sdata.seqs_annot.merge(
                smetadata, on="sequence", how="left"
            )
            sdata.seqs_annot.drop("sequence", axis=1, inplace=True)
        return sdata
    else:
        return paths


def deAlmeida22(dataset="train", return_sdata=True, **kwargs: dict) -> pd.DataFrame:
    """Reads the deAlmeida22 dataset.

    Parameters
    ----------
    dataset : str, optional
        Dataset to read. Either train or test. The default is "train".
    return_sdata : bool, optional
        If True, return SeqData object with the deAlmeida22 dataset. The default is True.
    **kwargs : kwargs, dict
        Keyword arguments to pass to read_csv.

    Returns
    -------
    sdata : SeqData
        SeqData object with the deAlmeida22 dataset.
    """
    urls_list = [
        "https://zenodo.org/record/5502060/files/Sequences_Train.fa?download=1",
        "https://zenodo.org/record/5502060/files/Sequences_Val.fa?download=1",
        "https://zenodo.org/record/5502060/files/Sequences_Test.fa?download=1",
        "https://zenodo.org/record/5502060/files/Sequences_activity_Train.txt?download=1",
        "https://zenodo.org/record/5502060/files/Sequences_activity_Val.txt?download=1",
        "https://zenodo.org/record/5502060/files/Sequences_activity_Test.txt?download=1",
    ]
    if dataset == "train":
        urls = [0, 3]
    elif dataset == "val":
        urls = [1, 4]
    elif dataset == "test":
        urls = [2, 5]
    paths = try_download_urls(urls, urls_list, "deAlmeida22")
    if return_sdata:
        sdata = read_fasta(seq_file=paths[0])
        sdata.seqs_annot = pd.read_csv(paths[1], sep="\t", **kwargs)
        sdata.seqs_annot.index = sdata.names
        return sdata
    else:
        return paths
