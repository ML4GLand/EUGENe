import pandas as pd
from os import PathLike
from typing import Union, List


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
