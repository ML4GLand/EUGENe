# Classics
import os
import sys
import numpy as np
import pandas as pd

# EUGENE
from ..preprocessing import random_seqs, reverse_complement, encodeDNA
from ._io import seq2Fasta

# Simple script tp generate commonly used file types for testing EUGENE models
def generate_random_data(num_seqs, seq_len, out_dir="./random_data/"):
    out_dir = os.path.join(out_dir, "random{0}seqs_{1}bp/".format(num_seqs, seq_len))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seqs = random_seqs(num_seqs, seq_len)
    ohe_seqs = encodeDNA(seqs)
    rev_seqs = [reverse_complement(seq) for seq in seqs]
    rev_ohe_seqs = encodeDNA(rev_seqs)
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
