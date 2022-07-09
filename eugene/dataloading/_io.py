# Absolute imports
import h5py
import numpy as np
import pandas as pd
from typing import Optional
from os import PathLike

# Relative imports
from .dataloaders import SeqData

def read_csv(file, seq_col="SEQ", name_col=None, target_col=None, binarize=False, rev_comp=False, sep="\t", low_thresh=None, high_thresh=None, low_memory=False, return_numpy=False):
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
        return_numpy (bool, optional): whether to return numpy arrays. Defaults to False.

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

    # Subset if thresholds are passed in
    if low_thresh != None or high_thresh != None:
        assert low_thresh != None and high_thresh != None and target_col != None
        dataframe["FXN_LABEL"] = np.nan
        dataframe.loc[dataframe[target_col] <= low_thresh, "FXN_LABEL"] = 0
        dataframe.loc[dataframe[target_col] >= high_thresh, "FXN_LABEL"] = 1
        dataframe = dataframe[~dataframe["FXN_LABEL"].isna()]

    # Grab sequences
    seqs = dataframe[seq_col].to_numpy(dtype=str)

    # Grab targets if column is provided
    if target_col is not None:
        if binarize:
            assert low_thresh != None and high_thresh != None
            targets = dataframe["FXN_LABEL"].to_numpy(float)
        else:
            targets = dataframe[target_col].to_numpy(float)
            targets = targets[~np.isnan(targets) & ~np.isinf(targets)]
    else:
        targets = None

    # Grab reverse complement if asked for
    if rev_comp:
        from ..preprocessing import reverse_complement
        rev_seqs = [reverse_complement(seq) for seq in seqs]
    else:
        rev_seqs = None

    # Return it all
    if return_numpy:
        return ids, seqs, rev_seqs, targets
    else:
        return SeqData(names=ids, seqs=seqs, rev_seqs=rev_seqs, seqs_annot=pd.DataFrame(data=targets, index=ids, columns=["TARGETS"]))


def read_fasta(seq_file, target_file=None, rev_comp=False, is_target_text=False, return_numpy=False):
    """Function for loading sequences into numpy objects from fasta

    Args:
        seq_file (str): fasta file path to read
        target_file (str): .npy or .txt file path containing targets. Defaults to None.
        rev_comp (bool, optional): whether to generate reverse complements for sequences. Defaults to False.
        is_target_text (bool, optional): whether the file is compressed or plaintext. Defaults to False.
        return_numpy (bool, optional): whether to return numpy arrays. Defaults to False.

    Returns:
        tuple: numpy arrays of identifiers, sequences, reverse complement sequences and targets.
               if any are not provided they are set to none
    """

    seqs = np.array([x.rstrip() for (i,x) in enumerate(open(seq_file)) if i%2==1])
    ids = np.array([x.rstrip().replace(">", "") for (i,x) in enumerate(open(seq_file)) if i%2==0])

    if target_file is not None:
        if is_target_text:
            targets = np.loadtxt(target_file, dtype=float)
        else:
            targets = np.load(target_file)
    else:
        targets = None

    if rev_comp:
        from ..preprocessing import reverse_complement
        rev_seqs =  [reverse_complement(seq) for seq in seqs]
    else:
        rev_seqs = None

    if return_numpy:
        return ids, seqs, rev_seqs, targets
    else:
        return  SeqData(names=ids, seqs=seqs, rev_seqs=rev_seqs, seqs_annot=pd.DataFrame(data=targets, columns=["TARGETS"]))


def read_numpy(seq_file, names_file=None, target_file=None, rev_seq_file=None, is_names_text=False, is_seq_text=False, is_target_text=False, delim="\n", ohe_encoded=False, return_numpy=False):
    """Function for loading sequences into numpy objects from numpy compressed files.
       Note if you pass one hot encoded sequences in, you must pass in reverse complements
       if you want them to be included

    Args:
        seq_file (str): .npy file path containing sequences
        names_file (str): .npy or .txt file path containing identifiers. Defaults to None.
        target_file (str): .npy or .txt file path containing targets. Defaults to None.
        rev_seq_file (str, optional): .npy or .txt file path containing reverse complement sequences. Defaults to None.
        is_names_text (bool, optional): whether the file is compressed (.npy) or plaintext (.txt). Defaults to False.
        is_seq_text (bool, optional): whether the file is (.npy) or plaintext (.txt). Defaults to False.
        is_target_text (bool, optional): whether the file is (.npy) or plaintext (.txt). Defaults to False.
        delim (str, optional):  Defaults to "\n".
        ohe_encoded (bool, optional): whether the sequences are one hot encoded. Defaults to False.
        return_numpy (bool, optional): whether to return numpy arrays. Defaults to False.

    Returns:
        tuple: numpy arrays of identifiers, sequences, reverse complement sequences and targets.
               if any are not provided they are set to none
    """
    if is_seq_text:
        seqs = np.loadtxt(seq_file, dtype=str, delim=delim)
        if rev_seq_file != None:
            rev_seqs = np.loadtxt(rev_seq_file, dtype=str)
        else:
            rev_seqs = None
    else:
        seqs = np.load(seq_file, allow_pickle=True)
        if rev_seq_file != None:
            rev_seqs = np.load(rev_seq_file, allow_pickle=True)
        else:
            rev_seqs = None

    if names_file is not None:
        if is_names_text:
            ids = np.loadtxt(names_file, dtype=str)
        else:
            ids = np.load(names_file)
    else:
        ids = None

    if target_file is not None:
        if is_target_text:
            targets = np.loadtxt(target_file, dtype=float)
        else:
            targets = np.load(target_file)
    else:
        targets = None

    if return_numpy:
        return ids, seqs, rev_seqs, targets
    elif ohe_encoded:
        return SeqData(names=ids, ohe_seqs=seqs, rev_seqs=rev_seqs, seqs_annot=pd.DataFrame(data=targets, columns=["TARGETS"]))
    else:
        return  SeqData(names=ids, seqs=seqs, rev_seqs=rev_seqs, seqs_annot=pd.DataFrame(data=targets, columns=["TARGETS"]))


def read_h5sd(filename: Optional[PathLike], sdata = None, mode: str = "r"):
    """Function for loading sequences into SeqData objects from h5sd files.

    Args:
        filename (str): .h5sd file path to read
        sdata (SeqData, optional): SeqData object to load data into. Defaults to None.
        mode (str, optional): mode to open file. Defaults to "r".

    Returns:
        SeqData: SeqData object containing sequences and identifiers
    """
    with h5py.File(filename, "r") as f:
        d = {}
        for k in f.keys():
            if "seqs" in f:
                d["seqs"] = np.array([n.decode("ascii", "ignore") for n in f["seqs"][:]])
            if "names" in f:
                d["names"] = np.array([n.decode("ascii", "ignore") for n in f["names"][:]])
            if "ohe_seqs" in f:
                d["ohe_seqs"] = f["ohe_seqs"][:]
            if "ohe_rev_seqs" in f:
                d["ohe_rev_seqs"] = f["ohe_rev_seqs"][:]
            if "rev_seqs" in f:
                d["rev_seqs"] = f["rev_seqs"][:]
            if "ohe_rev_seqs" in f:
                d["ohe_rev_seqs"] = f["ohe_rev_seqs"][:]
            if "seqs_annot" in f:
                out_dict = {}
                for key in f["seqs_annot"].keys():
                    out_dict[key] = f["seqs_annot"][key][()]
                    d["seqs_annot"] = pd.DataFrame(out_dict)
            if "pos_annot" in f:
                d["pos_annot"] = f["pos_annot"].attrs
            if "seqsm" in f:
                d["seqsm"] = f["seqsm"].attrs
    return SeqData(**d)


def read(seq_file, *args, **kwargs):
    """Wrapper function around read_csv, read_fasta, read_numpy to read sequence based input

    Args:
        seq_file (str): file path containing sequences
        args: positional arguments from read_csv, read_fasta, read_numpy, read_h5sd
        kwargs: keyword arguments from read_csv, read_fasta, read_numpy, read_h5sd

    Returns:
        tuple: numpy arrays of identifiers, sequences, reverse complement sequences and targets.
               if any are not provided they are set to none
    """
    seq_file_extension = seq_file.split(".")[-1]
    if seq_file_extension in ["csv", "tsv"]:
        return read_csv(seq_file, *args, **kwargs)
    elif seq_file_extension in ["npy"]:
        return read_numpy(seq_file, *args, **kwargs)
    elif seq_file_extension in ["fasta", "fa"]:
        return read_fasta(seq_file, *args, **kwargs)
    elif seq_file_extension in ["h5sd", "h5"]:
        return read_h5sd(seq_file, *args, **kwargs)
    else:
        print("Sequence file type not currently supported")
        return


def write_h5sd(sdata, filename: Optional[PathLike] = None, mode: str = "w"):
    """Write SeqData object to h5sd file."""
    with h5py.File(filename, mode) as f:
        f = f["/"]
        f.attrs.setdefault("encoding-type", "SeqData")
        f.attrs.setdefault("encoding-version", "0.0.0")
        if sdata.seqs is not None:
            f.create_dataset("seqs", data=np.array([n.encode("ascii", "ignore") for n in sdata.seqs]))
        if sdata.names is not None:
            f.create_dataset("names", data=np.array([n.encode("ascii", "ignore") for n in sdata.names]))
        if sdata.ohe_seqs is not None:
            f.create_dataset("ohe_seqs", data=sdata.ohe_seqs)
        if sdata.ohe_rev_seqs is not None:
            f.create_dataset("ohe_rev_seqs", data=sdata.ohe_rev_seqs)
        if sdata.rev_seqs is not None:
            f.create_dataset("rev_seqs", data=np.array([n.encode("ascii", "ignore") for n in sdata.rev_seqs]))
        if sdata.ohe_rev_seqs is not None:
            f.create_dataset("ohe_rev_seqs", data=sdata.ohe_rev_seqs)
        if sdata.seqs_annot is not None:
           for key, item in dict(sdata.seqs_annot).items():
                # note that not all variable types are supported but string and int are
                f["seqs_annot/" + str(key)] = item


def seq2Fasta(seqs, IDs, name="seqs"):
    """Utility function to generate a fasta file from a list of sequences and identifiers

    Args:
        seqs (list-like): list of sequences
        IDs (list-like):  list of identifiers
        name (str, optional): name of file. Defaults to "seqs".
    """
    file = open("{}.fa".format(name), "w")
    for i in range(len(seqs)):
        file.write(">" + IDs[i] + "\n" + seqs[i] + "\n")
    file.close()


def gkmSeq2Fasta(seqs, IDs, ys, name="seqs"):
    """Utility function to generate a fasta file from a list of sequences, identifiers and targets but splits into
         two files, one for positive label and one for negative label. Useful for running gkm-SVM.

    Args:
        seqs (_type_): _description_
        IDs (_type_): _description_
        ys (_type_): _description_
        name (str, optional): _description_. Defaults to "seqs".
    """
    neg_mask = (ys==0)

    neg_seqs, neg_ys, neg_IDs = seqs[neg_mask], ys[neg_mask], IDs[neg_mask]
    neg_file = open("{}-neg.fa".format(name), "w")
    for i in range(len(neg_seqs)):
        neg_file.write(">" + neg_IDs[i] + "\n" + neg_seqs[i] + "\n")
    neg_file.close()

    pos_seqs, pos_ys, pos_IDs = seqs[~neg_mask], ys[~neg_mask], IDs[~neg_mask]
    pos_file = open("{}-pos.fa".format(name), "w")
    for i in range(len(pos_seqs)):
        pos_file.write(">" + pos_IDs[i] + "\n" + pos_seqs[i] + "\n")
    pos_file.close()
