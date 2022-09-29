import pandas as pd
from itertools import product


# Useful helpers for generating and checking for kmers
def generate_all_possible_kmers(n=7, alphabet="AGCU"):
    """Generate all possible kmers of length and alphabet provided
    """
    return ["".join(c) for c in product(alphabet, repeat=n)]


def kmer_in_seqs(seqs, kmer):
    """Return a 0/1 array of whether a kmer is in each of the passed in sequences
    """
    seqs_s = pd.Series(seqs)
    kmer_binary = seqs_s.str.contains(kmer).astype(int).values
    return kmer_binary