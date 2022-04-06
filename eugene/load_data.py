import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/adamklie/Desktop/research/lab/dev/EUGENE/eugene")
import seq_utils

def load_csv(file, seq_col, name_col=None, target_col=None, sep="\t", rev_comp=False, low_thresh=None, high_thresh=None, low_memory=False):
    dataframe = pd.read_csv(file, sep=sep, low_memory=low_memory)
    seqs = dataframe[seq_col].to_numpy(dtype=str)

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
        targets =  dataframe["FXN_LABEL"].to_numpy()
    
    # Otherwise use passed in column if there
    else:
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