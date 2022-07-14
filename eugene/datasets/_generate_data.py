# Classics
import os
import numpy as np
import pandas as pd

# EUGENE
from ..preprocessing import reverse_complement_seq, ohe_DNA_seq
from ..dataloading._io import seq2Fasta


alphabet = np.array(["A", "G", "C", "T"])


def random_base():
    """
    Generate a random base.
    :return: Random base.
    """
    return np.random.choice(alphabet)


def random_seq(seq_len):
    """
    Generate a random sequence of length seq_len.
    :args
    seq_len (int): length of sequence to return
    :return: Random sequence.
    """
    return "".join([np.random.choice(alphabet) for i in range(seq_len)])


def random_seqs(seq_num, seq_len):
    """
    Generate seq_num random sequences of length seq_len
    :args
    seq_num (int): number of sequences to return
    seq_len (int): length of sequence to return
    :return: numpy array of random sequences.
    """
    return [random_seq(seq_len) for i in range(seq_num)]


def random_seqs_to_file(file, ext="csv", **kwargs):
    """
    Generate a random sequence of length seq_len and save to file
    :args
    seq_len (int): length of sequence to return
    :return: Random sequence.
    """
    pass


def generate_random_data(num_seqs, seq_len, out_dir="./random_data/"):
    """Simple script tp generate commonly used file types for testing EUGENE models"""
    out_dir = os.path.join(out_dir, "random{0}seqs_{1}bp/".format(num_seqs, seq_len))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seqs = random_seqs(num_seqs, seq_len)
    ohe_seqs = ohe_DNA_seq(seqs)
    rev_seqs = [reverse_complement(seq) for seq in seqs]
    rev_ohe_seqs = ohe_DNA_seq(rev_seqs)
    ids = np.array(["seq{0:03}".format(i+1) for i in range(num_seqs)])
    labels = np.array([np.random.randint(0,2) for i in range(num_seqs)])
    activities = np.array([np.random.rand() for i in range(num_seqs)])

    pd.DataFrame(data={"NAME": ids, "SEQ":seqs, "LABEL": labels, "ACTIVITY": activities}).to_csv(out_dir + "random_seqs.tsv", sep="\t", index=False)
    np.save(out_dir + "random_seqs", seqs)
    np.save(out_dir + "random_ohe_seqs", ohe_seqs)
    np.save(out_dir + "random_rev_seqs", rev_seqs)
    np.save(out_dir + "random_rev_ohe_seqs", rev_ohe_seqs)
    np.save(out_dir + "random_ids", ids)
    np.save(out_dir + "random_labels", labels)
    np.save(out_dir + "random_activities", activities)
    seq2Fasta(seqs, ids, name=out_dir + "random_seqs")
