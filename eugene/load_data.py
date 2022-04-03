import numpy as np
import pandas as pd


def load_csv(file, seq_col="SEQ", target_col="FXN", name_col=None, sep="\t", low_thresh=None, high_thresh=None, low_memory=False):
    dataframe = pd.read_csv(file, sep=sep, low_memory=low_memory)
    
    if low_thresh != None or high_thresh != None:
        assert low_thresh != None and high_thresh != None
        dataframe["FXN_LABEL"] = np.nan
        dataframe.loc[dataframe[target_col] <= low_thresh, "FXN_LABEL"] = 0
        dataframe.loc[dataframe[target_col] >= high_thresh, "FXN_LABEL"] = 1
        dataframe = dataframe[~dataframe["FXN_LABEL"].isna()]
        seqs, targets = dataframe[seq_col].to_numpy(dtype=str), dataframe["FXN_LABEL"].to_numpy()
    
    else:
        seqs, targets = dataframe[seq_col].to_numpy(dtype=str), dataframe[target_col].to_numpy(float)
        
    return seqs, targets

def load_fasta(seq_file, target_file, is_target_text=True):
    seqs = [x.rstrip() for (i,x) in enumerate(open(seq_file)) if i%2==1]
    #ids = [x.rstrip().replace(">", "") for (i,x) in enumerate(open(seq_file)) if i%2==0]
    targets = np.loadtxt(target_file, dtype=float)
    return seqs, targets

def load_numpy(seq_file, target_file, is_seq_text=False, is_target_text=True, delim="\n"):
    if is_seq_text:
        seqs = np.loadtxt(seq_file, dtype=str)
    else:
        seqs = np.load(seq_file)
    if is_target_text:
        targets = np.loadtxt(target_file, dtype=float)
    else:
        targets = np.load(target_file, dtype=float)
    return seqs, targets

def load(seq_file, *args, **kwargs):
    seq_file_extension = seq_file.split(".")[-1]
    if seq_file_extension in ["csv", "tsv"]:
        return load_csv(seq_file, **kwargs)
    elif seq_file_extension in ["npy"]:
        assert len(args) != 0
        target_file_extension = args[0].split(".")[-1]
        return load_numpy(seq_file, *args, **kwargs)  
    elif seq_file_extension in ["fasta", "fa"]:
        assert len(args) != 0
        target_file_extension = args[0].split(".")[-1]
        return load_fasta(seq_file, *args, **kwargs)
    else:
        logging.error("Sequence file type not currently supported")
        return