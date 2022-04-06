import numpy as np
import pandas as pd

from eugene import seq_utils

seqs = seq_utils.random_seqs(100, 1000)
ohe_seqs = np.array([seq_utils.ohe(seq) for seq in seqs])
rev_seqs = [seq_utils.reverse_complement(seq) for seq in seqs]
rev_ohe_seqs = np.array([seq_utils.ohe(rev_seq) for rev_seq in rev_seqs]) 
ids = np.array(["seq{0:03}".format(i+1) for i in range(100)])
labels = np.array([np.random.randint(0,2) for i in range(100)])
activities = np.array([np.random.rand() for i in range(100)])

pd.DataFrame(data={"NAME": ids, "SEQ":seqs, "LABEL": labels, "ACTIVITY": activities}).to_csv("test_seqs.tsv", sep="\t", index=False)
np.save("test_seqs", seqs)
np.save("test_ohe_seqs", ohe_seqs)
np.save("test_rev_seqs", rev_seqs)
np.save("test_rev_ohe_seqs", rev_ohe_seqs)
np.save("test_ids", ids)
np.save("test_labels", labels)
np.save("test_activities", activities)
seq_utils.seq2Fasta(seqs, ids, name="test_seqs")
