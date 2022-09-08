from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
from ._utils import (
    _get_index_dict,
    _one_hot2token,
    _tokenize,
    _token2one_hot,
    _pad_sequences,
)  # concise
from ._utils import (
    _string_to_char_array,
    _one_hot_to_tokens,
    _char_array_to_string,
    _tokens_to_one_hot,
)  # dinuc_shuffle


# Vocabularies
DNA = ["A", "C", "G", "T"]
COMPLEMENT_DNA = {"A": "T", "C": "G", "G": "C", "T": "A"}
COMPLEMENT_RNA = {"A": "U", "C": "G", "G": "C", "U": "A"}
RNA = ["A", "C", "G", "U"]
AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "B",
    "C",
    "E",
    "Q",
    "Z",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
CODONS = [
    "AAA",
    "AAC",
    "AAG",
    "AAT",
    "ACA",
    "ACC",
    "ACG",
    "ACT",
    "AGA",
    "AGC",
    "AGG",
    "AGT",
    "ATA",
    "ATC",
    "ATG",
    "ATT",
    "CAA",
    "CAC",
    "CAG",
    "CAT",
    "CCA",
    "CCC",
    "CCG",
    "CCT",
    "CGA",
    "CGC",
    "CGG",
    "CGT",
    "CTA",
    "CTC",
    "CTG",
    "CTT",
    "GAA",
    "GAC",
    "GAG",
    "GAT",
    "GCA",
    "GCC",
    "GCG",
    "GCT",
    "GGA",
    "GGC",
    "GGG",
    "GGT",
    "GTA",
    "GTC",
    "GTG",
    "GTT",
    "TAC",
    "TAT",
    "TCA",
    "TCC",
    "TCG",
    "TCT",
    "TGC",
    "TGG",
    "TGT",
    "TTA",
    "TTC",
    "TTG",
    "TTT",
]
STOP_CODONS = ["TAG", "TAA", "TGA"]


def sanitize_seq(seq):
    return seq.strip().upper()


def sanitize_seqs(seqs):
    return np.array([seq.strip().upper() for seq in seqs])


def reverse_complement_seq(seq, alphabet, copy=False):
    """Reverse complement a DNA sequence."""
    if alphabet == "DNA":
        return "".join(COMPLEMENT_DNA.get(base, base) for base in reversed(seq))
    elif alphabet == "RNA":
        return "".join(COMPLEMENT_RNA.get(base, base) for base in reversed(seq))
    else:
        raise ValueError("Invalid alphabet")


def reverse_complement_seqs(seqs, alphabet="DNA", copy=False):
    """Reverse complement a list of DNA sequences."""
    return np.array(
        [
            reverse_complement_seq(seq, alphabet)
            for i, seq in tqdm(
                enumerate(seqs),
                total=len(seqs),
                desc="Reverse complementing sequences",
            )
        ]
    )


def ohe_DNA_seq(seq, vocab=DNA, fill_value=0, neutral_vocab="N"):
    """Convert a DNA sequence into one-hot-encoded array."""
    seq = seq.strip().upper()
    return _token2one_hot(_tokenize(seq, vocab, neutral_vocab), len(vocab), fill_value=0)


def decode_DNA_seq(arr, vocab=DNA, neutral_vocab="N"):
    """Convert a one-hot encoded array back to string"""
    tokens = arr.argmax(axis=1)
    indexToLetter = _get_index_dict(vocab)
    return "".join([indexToLetter[x] for x in tokens])


def _ohe_seqs(
    seq_vec,
    vocab,
    neutral_vocab,
    maxlen=None,
    seq_align="start",
    pad_value="N",
    encode_type="one_hot",
    fill_value=None,
    pad=True,
    verbose=True
):
    """
    Convert a list of genetic sequences into one-hot-encoded array.
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
        raise ValueError(
            "seq_vec should be an iterable returning " + "strings not a string itself"
        )
    assert len(vocab[0]) == len(pad_value)
    assert pad_value in neutral_vocab
    assert encode_type in ["one_hot", "token"]

    if pad:
        seq_vec = _pad_sequences(
            seq_vec, maxlen=maxlen, align=seq_align, value=pad_value
        )

    if encode_type == "one_hot":
        arr_list = [
            _token2one_hot(_tokenize(seq, vocab, neutral_vocab), len(vocab), fill_value)
            for i, seq in tqdm(
                enumerate(seq_vec),
                total=len(seq_vec),
                desc="One-hot-encoding sequences",
                disable=not verbose
            )
        ]
    elif encode_type == "token":
        arr_list = [
            1 + np.array(_tokenize(seq, vocab, neutral_vocab)) for seq in seq_vec
        ]
        # we add 1 to be compatible with keras: https://keras.io/layers/embeddings/
        # indexes > 0, 0 = padding element

    if pad:
        return np.stack(arr_list)
    else:
        return np.array(arr_list, dtype=object)


def ohe_alphabet_seqs(
    seq_vec,
    alphabet,
    maxlen=None,
    seq_align="start",
    pad=True,
    fill_value=None,
    verbose=True,
    copy=False,
):
    """
    Convert the sequence into 1-hot-encoding np array
    Arguments
        seq_vec: list of chars. List of sequences that can have different lengths
        RNAbases : bool, Whether or not to use RNA bases in place of DNA bases. Default is false.
        maxlen: int or None, Should we trim (subset) the resulting sequence. If None don't trim. Note that trims wrt the align parameter. It should be smaller than the longest sequence.
        seq_align: character; 'end' or 'start' To which end should we align sequences?

    Returns
        3D np array of shape (len(seq_vec), trim_seq_len(or maximal sequence length if None), 4)"""
    return _ohe_seqs(
        seq_vec,
        vocab=DNA if alphabet == "DNA" else RNA,
        neutral_vocab="N",
        maxlen=maxlen,
        seq_align=seq_align,
        pad_value="N",
        encode_type="one_hot",
        fill_value=fill_value,
        pad=pad,
        verbose=verbose,
    )


def decode_DNA_seqs(arr, vocab=DNA, verbose=True):
    """Convert a one-hot encoded array back to string"""
    tokens = _one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)
    if verbose:
        return np.array(
            [
                "".join([indexToLetter[x] for x in row])
                for i, row in tqdm(
                    enumerate(tokens), total=len(tokens), desc="Decoding DNA sequences"
                )
            ]
        )
    else:
        return ["".join([indexToLetter[x] for x in row]) for row in tokens]


def dinuc_shuffle_seq(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D np array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a np RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D np array, then the
    result is an N x L x D np array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str or type(seq) is np.str_:
        arr = _string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = _one_hot_to_tokens(seq)
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
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
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
            all_results.append(_char_array_to_string(chars[result]))
        else:
            all_results[i] = _tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def dinuc_shuffle_seqs(seqs, num_shufs=None, rng=None):
    """
    Shuffle the sequences in `seqs` in the same way as `dinuc_shuffle_seq`.
    If `num_shufs` is not specified, then the first dimension of N will not be
    present (i.e. a single string will be returned, or an L x D array).
    """
    if not rng:
        rng = np.random.RandomState()

    if type(seqs) is str or type(seqs) is np.str_:
        seqs = [seqs]

    all_results = []
    for i in range(len(seqs)):
        all_results.append(dinuc_shuffle_seq(seqs[i], num_shufs=num_shufs, rng=rng))
    return np.array(all_results)


def perturb_seqs(X_0, ds=False):
    """Produce all edit-distance-one pertuabtions of a sequence.
    This function will take in a single one-hot encoded sequence of length N
    and return a batch of N*(n_choices-1) sequences, each containing a single
    perturbation from the given sequence.
    Parameters
    ----------
    X_0: np.ndarray, shape=(n_seqs, n_choices, seq_len)
        A one-hot encoded sequence to generate all potential perturbations.
    Returns
    -------
    X: torch.Tensor, shape=(n_seqs, (n_choices-1)*seq_len, n_choices, seq_len)
        Each single-position perturbation of seq.
    """

    if not isinstance(X_0, np.ndarray):
        raise ValueError("X_0 must be of type np.ndarray, not {}".format(type(X_0)))

    if len(X_0.shape) != 3:
        raise ValueError(
            "X_0 must have three dimensions: (n_seqs, n_choices, seq_len)."
        )

    n_seqs, n_choices, seq_len = X_0.shape
    idxs = X_0.argmax(axis=1)

    X_0 = torch.from_numpy(X_0)

    n = seq_len * (n_choices - 1)
    X = torch.tile(X_0, (n, 1, 1))
    X = X.reshape(n, n_seqs, n_choices, seq_len).permute(1, 0, 2, 3)

    for i in range(n_seqs):
        for k in range(1, n_choices):
            idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)

            X[i, idx, idxs[i], np.arange(seq_len)] = 0
            X[i, idx, (idxs[i] + k) % n_choices, np.arange(seq_len)] = 1

    return X


def feature_implant_seq(seq, feature, position, encoding="str", onehot=False):
    """
    Insert a feature at a given position in a sequence.
    """
    if encoding == "str":
        return seq[:position] + feature + seq[position + len(feature):]
    elif encoding == "onehot":
        if onehot:
            feature = _token2one_hot(feature.argmax(axis=1), vocab_size=4, fill_value=0)
        return np.concatenate(
            (seq[:position], feature, seq[position + len(feature):]), axis=0
        )
    else:
        raise ValueError("Encoding not recognized.")


def feature_implant_across_seq(seq, feature, **kwargs):
    """
    Insert a feature at every position in a sequence.
    """
    implanted_seqs = []
    for pos in range(len(seq) - len(feature) + 1):
        implanted_seqs.append(feature_implant_seq(seq, feature, pos, **kwargs))
    return np.array(implanted_seqs)

