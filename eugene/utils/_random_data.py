import os
import numpy as np
import pandas as pd
from .._settings import settings
from ..preprocess import reverse_complement_seqs, ohe_seqs
from ..dataload._utils import _seq2Fasta


alphabet = np.array(["A", "G", "C", "T"])


def random_base():
    """
    Generate a random base from the AGCT alpahbet.
    """
    return np.random.choice(alphabet)


def random_seq(seq_len):
    """
    Generate a random sequence of length seq_len.

    Parameters
    ----------
    seq_len : int
        Length of sequence to return.

    Returns
    -------
    Random sequence.
    """
    return "".join([np.random.choice(alphabet) for i in range(seq_len)])


def random_seqs(seq_num, seq_len):
    """
    Generate seq_num random sequences of length seq_len

    Parameters
    ----------
    seq_num (int):
        number of sequences to return
    seq_len (int):
        length of sequence to return

    Returns
    -------
    numpy array of random sequences.
    """
    return np.array([random_seq(seq_len) for i in range(seq_num)])


def generate_random_data(
    num_seqs: int, 
    seq_len: int, 
    vocab: str = "DNA", 
    num_outputs: int = 1, 
    out_dir: str = None, 
    dataset_name: str = None
):
    """
    Simple function tp generate commonly used file types for testing EUGENe models

    Parameters
    ----------
    num_seqs (int):
        number of sequences to generate
    seq_len (int):
        length of sequences to generate
    vocab (str):
        vocabulary to use for sequence generation. Default is DNA.
    num_outputs (int):
        number of outputs to generate. Default is 1.
    out_dir (str):
        directory to save files to. Default is None.
    dataset_name (str):
        name of dataset. Default is None.
    """
    out_dir = out_dir if out_dir is not None else settings.dataset_dir

    if dataset_name is None:
        dataset_name = f"random{num_seqs}seqs_{seq_len}bp"

    out_dir = os.path.join(out_dir, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seqs = random_seqs(num_seqs, seq_len)
    oheseqs = ohe_seqs(seqs, vocab=vocab)
    rev_seqs = reverse_complement_seqs(seqs)
    rev_ohe_seqs = ohe_seqs(rev_seqs, vocab=vocab)
    n_digits = len(str(num_seqs - 1))
    ids = np.array(
        ["seq{num:0{width}}".format(num=i, width=n_digits) for i in range(num_seqs)]
    )
    labels = np.random.randint(0, 2, size=(num_seqs, num_outputs))
    activities = np.random.rand(num_seqs, num_outputs)
    label_cols = ["label_{}".format(i) for i in range(num_outputs)]
    activity_cols = ["activity_{}".format(i) for i in range(num_outputs)]
    d = dict(
        dict(name=ids, seq=seqs),
        **dict(zip(label_cols, labels.T)),
        **dict(zip(activity_cols, activities.T)),
    )
    pd.DataFrame(d).to_csv(
        os.path.join(out_dir, f"{dataset_name}_seqs.tsv"), sep="\t", index=False
    )
    np.save(os.path.join(out_dir, f"{dataset_name}_seqs"), seqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_ohe_seqs"), oheseqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_rev_seqs"), rev_seqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_ohe_rev_seqs"), rev_ohe_seqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_ids"), ids)
    np.save(os.path.join(out_dir, f"{dataset_name}_labels"), labels)
    np.save(os.path.join(out_dir, f"{dataset_name}_activities"), activities)
    _seq2Fasta(seqs, ids, name=os.path.join(out_dir, f"{dataset_name}_seqs"))
