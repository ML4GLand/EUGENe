import numpy as np

# Function to grab scores from output of gkmtest
# name => filepath to read from
def get_scores(fname):
    f = open(fname)
    d = np.array([float(x.strip().split('\t')[1]) for x in f])
    f.close()
    return d


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
