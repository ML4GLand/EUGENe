def _seq2Fasta(seqs, IDs, name="seqs"):
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
