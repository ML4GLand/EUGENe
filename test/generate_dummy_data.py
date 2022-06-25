import os
import sys
import numpy as np
import pandas as pd

# EUGENE
from ..preprocessing import random_seqs, reverse_complement, encodeDNA

# Simple script tp generate commonly used file types for testing EUGENE models
if __name__ == "__main__":
    args = sys.argv
    num_seqs, seq_len = int(args[1]), int(args[2])
    out_dir = "test_{0}seqs_{1}/".format(num_seqs, seq_len)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seqs = random_seqs(num_seqs, seq_len)
    ohe_seqs = np.array([ohe(seq) for seq in seqs])
    rev_seqs = [reverse_complement(seq) for seq in seqs]
    rev_ohe_seqs = np.array([encodeDNA(rev_seq) for rev_seq in rev_seqs])
    ids = np.array(["seq{0:03}".format(i+1) for i in range(num_seqs)])
    labels = np.array([np.random.randint(0,2) for i in range(num_seqs)])
    activities = np.array([np.random.rand() for i in range(num_seqs)])

    pd.DataFrame(data={"NAME": ids, "SEQ":seqs, "LABEL": labels, "ACTIVITY": activities}).to_csv(out_dir + "test_seqs.tsv", sep="\t", index=False)
    np.save(out_dir + "test_seqs", seqs)
    np.save(out_dir + "test_ohe_seqs", ohe_seqs)
    np.save(out_dir + "test_rev_seqs", rev_seqs)
    np.save(out_dir + "test_rev_ohe_seqs", rev_ohe_seqs)
    np.save(out_dir + "test_ids", ids)
    np.save(out_dir + "test_labels", labels)
    np.save(out_dir + "test_activities", activities)
    seq2Fasta(seqs, ids, name=out_dir + "test_seqs")
