import tqdm
import pandas as pd
import numpy as np
from ._utils import _get_vocab_dict, _get_index_dict, _one_hot2token, _tokenize, _token2one_hot, _pad_sequences


# Vocabularies
DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
AMINO_ACIDS = ["A", "R", "N", "D", "B", "C", "E", "Q", "Z", "G", "H",
               "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
CODONS = ["AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT", "AGA",
          "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT", "CAA", "CAC",
          "CAG", "CAT", "CCA", "CCC", "CCG", "CCT", "CGA", "CGC", "CGG",
          "CGT", "CTA", "CTC", "CTG", "CTT", "GAA", "GAC", "GAG", "GAT",
          "GCA", "GCC", "GCG", "GCT", "GGA", "GGC", "GGG", "GGT", "GTA",
          "GTC", "GTG", "GTT", "TAC", "TAT", "TCA", "TCC", "TCG", "TCT",
          "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT"]
STOP_CODONS = ["TAG", "TAA", "TGA"]


def _ohe_seq(seq_vec, vocab, neutral_vocab, maxlen=None, seq_align="start", pad_value="N", encode_type="one_hot"):
    """Convert a list of genetic sequences into one-hot-encoded array.
        Arguments
            seq_vec: list of strings (genetic sequences)
            vocab: list of chars: List of "words" to use as the vocabulary. Can be strings of length>0, but all need to have the same length. For DNA, this is: ["A", "C", "G", "T"].
            neutral_vocab: list of chars: Values used to pad the sequence or represent unknown-values. For DNA, this is: ["N"].
            maxlen: int or None, should we trim (subset) the resulting sequence. If None don't trim. Note that trims wrt the align parameter. It should be smaller than the longest sequence.
            seq_align: character; 'end' or 'start' To which end should we align sequences?
            encode_type: "one_hot" or "token". "token" represents each vocab element as a positive integer from 1 to len(vocab) + 1. neutral_vocab is represented with 0.

        Returns
            Array with shape for encode_type:
                - "one_hot": `(len(seq_vec), maxlen, len(vocab))`
                - "token": `(len(seq_vec), maxlen)`
            If `maxlen=None`, it gets the value of the longest sequence length from `seq_vec`.
    """
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]
    if isinstance(seq_vec, str):
        raise ValueError("seq_vec should be an iterable returning " +
                         "strings not a string itself")
    assert len(vocab[0]) == len(pad_value)
    assert pad_value in neutral_vocab

    assert encode_type in ["one_hot", "token"]

    seq_vec = pad_sequences(seq_vec, maxlen=maxlen,
                            align=seq_align, value=pad_value)

    if encode_type == "one_hot":
        arr_list = [token2one_hot(tokenize(seq, vocab, neutral_vocab), len(vocab))
                    for i, seq in enumerate(seq_vec)]
    elif encode_type == "token":
        arr_list = [1 + np.array(tokenize(seq, vocab, neutral_vocab)) for seq in seq_vec]
        # we add 1 to be compatible with keras: https://keras.io/layers/embeddings/
        # indexes > 0, 0 = padding element

    return np.stack(arr_list)


def ohe_DNA_seq(seq, vocab=DNA, neutral_vocab="N"):
    seq = seq.strip().upper()
    return token2one_hot(tokenize(seq, vocab, neutral_vocab), len(vocab))


def decode_DNA_seq(arr, vocab=DNA, neutral_vocab="N"):
    tokens = arr.argmax(axis=1)
    indexToLetter = _get_index_dict(vocab)
    return ''.join([indexToLetter[x] for x in tokens])


def ohe_DNA_seqs(seq_vec, maxlen=None, seq_align="start", copy=False):
    """Convert the DNA sequence into 1-hot-encoding numpy array
        Arguments
            seq_vec: list of chars. List of sequences that can have different lengths
            maxlen: int or None, Should we trim (subset) the resulting sequence. If None don't trim. Note that trims wrt the align parameter. It should be smaller than the longest sequence.
            seq_align: character; 'end' or 'start' To which end should we align sequences?

        Returns
            3D numpy array of shape (len(seq_vec), trim_seq_len(or maximal sequence length if None), 4)"""
    return encodeSequence(seq_vec,
                          vocab=DNA,
                          neutral_vocab="N",
                          maxlen=maxlen,
                          seq_align=seq_align,
                          pad_value="N",
                          encode_type="one_hot")


def decode_DNA_seqs(arr, vocab=DNA):
    """Convert a one-hot encoded array back to string"""
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)
    return [''.join([indexToLetter[x] for x in row]) for row in tokens]


def reverse_complement_seq(seq, copy=False):
    return "".join(COMPLEMENT.get(base, base) for base in reversed(seq))


def reverse_complement_seqs(seqs, copy=False):
    return np.array([reverse_complement(seq) for seq in seqs])


def dinuc_shuffle_seq(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str or type(seq) is np.str_:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str or type(seq) is np.str_:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str or type(seq) is np.str_:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]
