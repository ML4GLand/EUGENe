import numpy as np
import pandas as pd

# EUGENE
from eugene.utils import seq_utils

def load_csv(file, seq_col, name_col=None, target_col=None, sep="\t", rev_comp=False, low_thresh=None, high_thresh=None, low_memory=False):
    """Function for loading sequences into numpy objects from csv/tsv files

    Args:
        file (str): tsv/csv file path to read
        seq_col (str, optional): name of column containing sequences. Defaults to None.
        name_col (str, optional): name of column containing sequence names. Defaults to None.
        target_col (str, optional): name of column containing sequence names. Defaults to None.
        sep (str, optional): column separator. Defaults to "\t".
        rev_comp (bool, optional): whether to generate reverse complements for sequences. Defaults to False.
        low_thresh (float, optional): if specified all activities under this threshold are considered inactive. Defaults to None.
        high_thresh (float, optional): if specified all activities above this threshold are considered inactive. Defaults to None.
        low_memory (bool, optional): whether to read file in low_memory mode. Defaults to False.

    Returns:
        tuple: numpy arrays of identifiers, sequences, reverse complement sequences and targets. 
               if any are not provided they are set to none
    """
    # Load as pandas dataframe
    dataframe = pd.read_csv(file, sep=sep, low_memory=low_memory)

    # Add names if available
    if name_col is not None:
        ids = dataframe[name_col].to_numpy(dtype=str)
    else:
        ids = None

    # Create own target column if thresholds passed in
    if low_thresh != None or high_thresh != None:
        assert low_thresh != None and high_thresh != None
        dataframe["FXN_LABEL"] = np.nan
        dataframe.loc[dataframe[target_col] <= low_thresh, "FXN_LABEL"] = 0
        dataframe.loc[dataframe[target_col] >= high_thresh, "FXN_LABEL"] = 1
        dataframe = dataframe[~dataframe["FXN_LABEL"].isna()]
        seqs = dataframe[seq_col].to_numpy(dtype=str)
        targets =  dataframe["FXN_LABEL"].to_numpy()
    
    # Otherwise use passed in column if there
    else:
        seqs = dataframe[seq_col].to_numpy(dtype=str)
        if target_col is not None:
            targets = dataframe[target_col].to_numpy(float)
        else:
           targets = None 

    # Grab reverse complement if asked for 
    if rev_comp:
       rev_seqs = [seq_utils.reverse_complement(seq) for seq in seqs]
    else:
        rev_seqs = None

    # Return it all
    return ids, seqs, rev_seqs, targets

def load_fasta(seq_file, target_file=None, rev_comp=False, is_target_text=False):
    """Function for loading sequences into numpy objects from fasta

    Args:
        seq_file (str): fasta file path to read 
        target_file (str): .npy file path containing sequences. Defaults to None.
        rev_comp (bool, optional): whether to generate reverse complements for sequences. Defaults to False.
        is_target_text (bool, optional): whether the file is compressed or plaintext. Defaults to False.

    Returns:
        tuple: numpy arrays of identifiers, sequences, reverse complement sequences and targets. 
               if any are not provided they are set to none
    """
    
    seqs = [x.rstrip() for (i,x) in enumerate(open(seq_file)) if i%2==1]
    ids = [x.rstrip().replace(">", "") for (i,x) in enumerate(open(seq_file)) if i%2==0]

    if target_file is not None:
        if is_target_text:
            targets = np.loadtxt(target_file, dtype=float)
        else:
            targets = np.load(target_file)
    else:
        targets = None

    if rev_comp:
        rev_seqs =  [seq_utils.reverse_complement(seq) for seq in seqs]
    else:
        rev_seqs = None

    return ids, seqs, rev_seqs, targets

def load_numpy(seq_file, names_file=None, target_file=None, rev_seq_file=None, is_names_text=False, is_seq_text=False, is_target_text=False, delim="\n"):
    """Function for loading sequences into numpy objects from numpy compressed files.
       Note if you pass one hot encoded sequences in, you must pass in reverse complements
       if you want them to be included

    Args:
        seq_file (str): .npy file path containing sequences
        names_file (str): .npy file path containing identifiers. Defaults to None.
        target_file (str): .npy file path containing sequences. Defaults to None.
        rev_seq_file (str, optional):  .npy file path containing reverse complement sequences. Defaults to None.
        is_names_text (bool, optional): whether the file is compressed or plaintext. Defaults to False.
        is_seq_text (bool, optional): whether the file is compressed or plaintext. Defaults to False.
        is_target_text (bool, optional): whether the file is compressed or plaintext. Defaults to False.
        delim (str, optional): _description_. Defaults to "\n".

    Returns:
        tuple: numpy arrays of identifiers, sequences, reverse complement sequences and targets. 
               if any are not provided they are set to none
    """    
    if is_seq_text:
        seqs = np.loadtxt(seq_file, dtype=str, delim=delim)
        if rev_seq_file != None:
            rev_seqs = np.loadtxt(rev_seq_file, dtype=str, delim=delim)
        else:
            rev_seqs = None
    else:
        seqs = np.load(seq_file)
        if rev_seq_file != None:
            rev_seqs = np.load(rev_seq_file)
        else:
            rev_seqs = None

    if names_file is not None:
        if is_names_text:
            ids = np.loadtxt(names_file, dtype=str, delim=delim)
        else:
            ids = np.load(names_file) 
    else:
        ids = None
    
    if target_file is not None:
        if is_target_text:
            targets = np.loadtxt(target_file, dtype=float, delim=delim)
        else:
            targets = np.load(target_file)
    else:
        targets = None

    return ids, seqs, rev_seqs, targets

def load(seq_file, *args, **kwargs):
    """Wrapper function around load_csv, load_fasta, load_numpy to read sequence based input

    Args:
        seq_file (str): file path containing sequences 
        *args: positional arguments from load_csv, load_fasta, load_numpy
        kwargs: keyword arguments from load_csv, load_fasta, load_numpy 

    Returns:
        tuple: numpy arrays of identifiers, sequences, reverse complement sequences and targets. 
               if any are not provided they are set to none
    """
    seq_file_extension = seq_file.split(".")[-1]
    if seq_file_extension in ["csv", "tsv"]:
        return load_csv(seq_file, *args, **kwargs)
    elif seq_file_extension in ["npy"]:
        return load_numpy(seq_file, *args, **kwargs)  
    elif seq_file_extension in ["fasta", "fa"]:
        return load_fasta(seq_file, *args, **kwargs)
    else:
        print("Sequence file type not currently supported")
        return