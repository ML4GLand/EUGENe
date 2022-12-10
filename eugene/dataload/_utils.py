from os import PathLike
from pathlib import Path
from typing import Union, List
import numpy as np
import pandas as pd


def _read_and_concat_dataframes(
    file_names: Union[PathLike, List[PathLike]],
    col_names: Union[str, list] = None,
    sep: str = "\t",
    low_memory: bool = False,
    compression: str = "infer",
    **kwargs
) -> pd.DataFrame:
    """Reads a list of files and concatenates them into a single dataframe.

    Parameters
    ----------
    file_names : str or list
        Path to file or list of paths to files.
    col_names : str or list, optional
        Column names to use for the dataframe. If not provided, the column names will be the first row of the file.
    sep : str, optional
        Separator to use for the dataframe. Defaults to "\t".
    low_memory : bool, optional
        If True, the dataframe will be stored in memory. If False, the dataframe will be stored on disk. Defaults to False.
    compression : str, optional
        Compression to use for the dataframe. Defaults to "infer".
    **kwargs
        Additional arguments to pass to pd.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    file_names = [file_names] if isinstance(file_names, str) else file_names
    dataframe = pd.DataFrame()
    for file_name in file_names:
        x = pd.read_csv(
            file_name,
            sep=sep,
            low_memory=low_memory,
            names=col_names,
            compression=compression,
            header=0,
            **kwargs
        )
        dataframe = pd.concat([dataframe, x], ignore_index=True)
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe


def _seq2Fasta(seqs, IDs, name="seqs"):
    """Utility function to generate a fasta file from a list of sequences and identifiers

    Parameters
    ----------
    seqs (list-like):
        list of sequences
    IDs (list-like):
        list of identifiers
    name (str, optional):
        name of file. Defaults to "seqs".
    """
    file = open("{}.fa".format(name), "w")
    for i in range(len(seqs)):
        file.write(">" + IDs[i] + "\n" + seqs[i] + "\n")
    file.close()

def _concat_seqsm(seqsms, keys):
    res = {}
    for i, seqsm in enumerate(seqsms):
        for key in seqsm:
            if key in res:
                #print(res[key].shape[1], seqsm[key].shape[1])
                if res[key].shape[1] == seqsm[key].shape[1]:
                    res[key] = np.concatenate([res[key], seqsm[key]])
                else:
                    print(f"{keys[i]}'s {key} is not the same shape as previous, skipping")
                    continue
            elif i == 0:
                res[key] = seqsm[key]
            else:
                print(f"{keys[i]} does not contain {key}, skipping {key}")
                continue
    return res

def concat(
    sdatas,
    keys: Union[str, list] = None,
):
    """Concatenates a list of SeqData objects together without merging.

    Does not currently support merging of uns and seqsm.
    Only objects present in the first sdata of the list will be merged

    Parameters
    ----------
    sdatas : list of SeqData objects
        List of SeqData objects to concatenate together
    keys : str or list, optional
        Names to add in seqs_annot column "batch"
    """
    from . import SeqData

    concat_seqs = (
        np.concatenate([s.seqs for s in sdatas]) if sdatas[0].seqs is not None else None
    )
    concat_names = (
        np.concatenate([s.names for s in sdatas])
        if sdatas[0].names is not None
        else None
    )
    concat_ohe_seqs = (
        np.concatenate([s.ohe_seqs for s in sdatas])
        if sdatas[0].ohe_seqs is not None
        else None
    )
    concat_rev_seqs = (
        np.concatenate([s.rev_seqs for s in sdatas])
        if sdatas[0].rev_seqs is not None
        else None
    )
    concat_rev_ohe_seqs = (
        np.concatenate([s.ohe_rev_seqs for s in sdatas])
        if sdatas[0].ohe_rev_seqs is not None
        else None
    )
    concat_seqsm = _concat_seqsm([s.seqsm for s in sdatas], keys=keys)
    for i, s in enumerate(sdatas):
        s["batch"] = keys[i]
    concat_seqs_annot = pd.concat([s.seqs_annot for s in sdatas])
    return SeqData(
        seqs=concat_seqs,
        names=concat_names,
        ohe_seqs=concat_ohe_seqs,
        rev_seqs=concat_rev_seqs,
        ohe_rev_seqs=concat_rev_ohe_seqs,
        seqs_annot=concat_seqs_annot,
        seqsm=concat_seqsm
    )
