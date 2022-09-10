# Classics
import os
import numpy as np
import pandas as pd

# EUGENE
from .._settings import settings
from ..preprocessing import reverse_complement_seqs, ohe_seqs
from ..dataloading._utils import _seq2Fasta


alphabet = np.array(["A", "G", "C", "T"])


def random_base():
    """
    Generate a random base.
    Args:
        None
    Returns:
        Random base.
    """
    return np.random.choice(alphabet)


def random_seq(seq_len):
    """
    Generate a random sequence of length seq_len.
    Args:
        seq_len (int): length of sequence to return
    Returns:
        Random sequence.
    """
    return "".join([np.random.choice(alphabet) for i in range(seq_len)])


def random_seqs(seq_num, seq_len):
    """
    Generate seq_num random sequences of length seq_len
    Args
        seq_num (int): number of sequences to return
        seq_len (int): length of sequence to return
    Returns:
        numpy array of random sequences.
    """
    return np.array([random_seq(seq_len) for i in range(seq_num)])


def random_seqs_to_file(file, ext="csv", **kwargs):
    """
    Generate a random sequence of length seq_len and save to file
    Args
        seq_len (int): length of sequence to return
    Returns:
        Random sequence.
    """
    pass


def generate_random_data(num_seqs, seq_len, vocab="DNA", num_outputs=1, out_dir=None, dataset_name=None):
    """Simple function tp generate commonly used file types for testing EUGENE models"""
    out_dir = out_dir if out_dir is not None else settings.dataset_dir

    if dataset_name is None:
        dataset_name = f"random{num_seqs}seqs_{seq_len}bp"

    out_dir = os.path.join(out_dir, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seqs = random_seqs(num_seqs, seq_len)
    ohe_seqs = ohe_seqs(seqs, vocab=vocab)
    rev_seqs = reverse_complement_seqs(seqs)
    rev_ohe_seqs = ohe_seqs(rev_seqs, vocab=vocab)
    n_digits = len(str(num_seqs-1))
    ids = np.array(["seq{num:0{width}}".format(num=i, width=n_digits) for i in range(num_seqs)])
    labels = np.random.randint(0,2,size=(num_seqs, num_outputs))
    activities = np.random.rand(num_seqs, num_outputs)
    label_cols = ["LABEL_{}".format(i) for i in range(num_outputs)]
    activity_cols = ["ACTIVITY_{}".format(i) for i in range(num_outputs)]
    d = dict(dict(NAME=ids, SEQ=seqs), **dict(zip(label_cols, labels.T)), **dict(zip(activity_cols, activities.T)))
    pd.DataFrame(d).to_csv(os.path.join(out_dir, f"{dataset_name}_seqs.tsv"), sep="\t", index=False)
    np.save(os.path.join(out_dir, f"{dataset_name}_seqs"), seqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_ohe_seqs"), ohe_seqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_rev_seqs"), rev_seqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_rev_ohe_seqs"), rev_ohe_seqs)
    np.save(os.path.join(out_dir, f"{dataset_name}_ids"), ids)
    np.save(os.path.join(out_dir, f"{dataset_name}_labels"), labels)
    np.save(os.path.join(out_dir, f"{dataset_name}_activities"), activities)
    _seq2Fasta(seqs, ids, name=os.path.join(out_dir, f"{dataset_name}_seqs"))
