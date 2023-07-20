# EUGENe on sdata
def count_kmers_sdata(sdata: SeqData, k: int, frequency: bool = False) -> dict:
    """
    Counts k-mers in a given sequence from a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences.
    k : int
        k value for k-mers (e.g. k=3 generates 3-mers).
    frequency : bool
        Whether to return relative k-mer frequency in place of count.
        Default is False.

    Returns
    -------
    kmers : dict
        k-mers and their counts expressed in a dictionary.
    """
    data = {}
    for seq in sdata.seqs:
        data = count_kmers(seq, k, data)
    if frequency:
        total = sum(data.values())
        for kmer in data:
            data[kmer] = data[kmer] / total
    return data


# EUGENe on sdata?
def edit_distance_sdata(
    sdata1: SeqData, sdata2: SeqData, dual: bool = False, average: bool = False
) -> list:
    """
    Calculates the nucleotide edit distance between pairs of sequences from a SeqData object.

    Parameters
    ----------
    sdata1 : SeqData
        First SeqData object containing sequences.
    sdata2 : SeqData
        Second SeqData object containing sequences.
    dual : bool
        Whether to calculate the forwards and backwards edit distance, and return the lesser.
        Defaults to False.
    average : bool
        Whether to average all edit distances and return in place of a list.
        Defaults to False.

    Returns
    -------
    edits : list
        List containing itemized amounts of edits between sequences.
    """
    assert len(sdata1.seqs) == len(
        sdata2.seqs
    ), "Both SeqData objects must be of the same length."
    distances = []
    for i in range(len(sdata1.seqs)):
        distances.append(edit_distance(sdata1.seqs[i], sdata2.seqs[i], dual))
    if average:
        return sum(distances) / len(distances)
    return distances
