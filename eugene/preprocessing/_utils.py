import numpy as np
np.random.seed(13)


DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
COMPLEMENT_DNA = {"A": "T", "C": "G", "G": "C", "T": "A"}
COMPLEMENT_RNA = {"A": "U", "C": "G", "G": "C", "U": "A"}


def _get_vocab(vocab):
    if vocab == "DNA":
        return DNA
    elif vocab == "RNA":
        return RNA
    else:
        raise ValueError("Invalid vocab, only DNA or RNA are currently supported")


# exact concise
def _get_vocab_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    Used in `_tokenize`.
    """
    return {l: i for i, l in enumerate(vocab)}


# exact concise
def _get_index_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    """
    return {i: l for i, l in enumerate(vocab)}


# modified concise
def _tokenize(seq, vocab="DNA", neutral_vocab=["N"]):
    """
    Convert sequence to integers based on a vocab

    Parameters
    ----------
    seq: 
        sequence to encode
    vocab: 
        vocabulary to use
    neutral_vocab: 
        neutral vocabulary -> assign those values to -1
    
    Returns
    -------
        List of length `len(seq)` with integers from `-1` to `len(vocab) - 1`
    """
    vocab = _get_vocab(vocab)
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]

    nchar = len(vocab[0])
    for l in vocab + neutral_vocab:
        assert len(l) == nchar
    assert len(seq) % nchar == 0  # since we are using striding

    vocab_dict = _get_vocab_dict(vocab)
    for l in neutral_vocab:
        vocab_dict[l] = -1

    # current performance bottleneck
    return [
        vocab_dict[seq[(i * nchar) : ((i + 1) * nchar)]]
        for i in range(len(seq) // nchar)
    ]


def _sequencize(tvec, vocab="DNA", neutral_value=-1, neutral_char="N"):
    """
    Converts a token vector into a sequence of symbols of a vocab.
    """
    vocab = _get_vocab(vocab) 
    index_dict = _get_index_dict(vocab)
    index_dict[neutral_value] = neutral_char
    return "".join([index_dict[i] for i in tvec])


# modified concise
def _token2one_hot(tvec, vocab="DNA", fill_value=None):
    """
    Converts an L-vector of integers in the range [0, D] into an L x D one-hot
    encoding. If fill_value is not None, then the one-hot encoding is filled
    with this value instead of 0.

    Parameters
    ----------
    tvec : np.array
        L-vector of integers in the range [0, D]
    vocab_size : int
        D
    fill_value : float, optional
        Value to fill the one-hot encoding with. If None, then the one-hot
    """
    vocab = _get_vocab(vocab)
    vocab_size = len(vocab)
    arr = np.zeros((len(tvec), vocab_size))
    tvec_range = np.arange(len(tvec))
    tvec = np.asarray(tvec)
    arr[tvec_range[tvec >= 0], tvec[tvec >= 0]] = 1
    if fill_value is not None:
        arr[tvec_range[tvec < 0]] = fill_value
    return arr.astype(np.int8) if fill_value is None else arr.astype(np.float16)


# modified dinuc_shuffle
def _one_hot2token(one_hot, neutral_value=-1):
    """
    Converts a one-hot encoding into a vector of integers in the range [0, D]
    where D is the number of classes in the one-hot encoding.

    Parameters
    ----------
    one_hot : np.array
        L x D one-hot encoding
    neutral_value : int, optional
        Value to use for neutral values.
    
    Returns
    -------
    np.array
        L-vector of integers in the range [0, D]
    """
    tokens = np.tile(neutral_value, one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot==1)
    tokens[seq_inds] = dim_inds
    return tokens


# pad and subset, exact concise
def _pad(seq, max_seq_len, value="N", align="end"):
    seq_len = len(seq)
    assert max_seq_len >= seq_len
    if align == "end":
        n_left = max_seq_len - seq_len
        n_right = 0
    elif align == "start":
        n_right = max_seq_len - seq_len
        n_left = 0
    elif align == "center":
        n_left = (max_seq_len - seq_len) // 2 + (max_seq_len - seq_len) % 2
        n_right = (max_seq_len - seq_len) // 2
    else:
        raise ValueError("align can be of: end, start or center")

    # normalize for the length
    n_left = n_left // len(value)
    n_right = n_right // len(value)

    return value * n_left + seq + value * n_right


# exact concise
def _trim(seq, maxlen, align="end"):
    seq_len = len(seq)

    assert maxlen <= seq_len
    if align == "end":
        return seq[-maxlen:]
    elif align == "start":
        return seq[0:maxlen]
    elif align == "center":
        dl = seq_len - maxlen
        n_left = dl // 2 + dl % 2
        n_right = seq_len - dl // 2
        return seq[n_left:n_right]
    else:
        raise ValueError("align can be of: end, start or center")


# modified concise
def _pad_sequences(
    seqs, 
    maxlen=None, 
    align="end", 
    value="N"
):
    """
    Pads sequences to the same length.

    Parameters
    ----------
    seqs : list of str
        Sequences to pad
    maxlen : int, optional
        Length to pad to. If None, then pad to the length of the longest sequence.
    align : str, optional
        Alignment of the sequences. One of "start", "end", "center"
    value : str, optional
        Value to pad with

    Returns
    -------
    np.array
        Array of padded sequences
    """

    # neutral element type checking
    assert isinstance(value, list) or isinstance(value, str)
    assert isinstance(value, type(seqs[0])) or type(seqs[0]) is np.str_
    assert not isinstance(seqs, str)
    assert isinstance(seqs[0], list) or isinstance(seqs[0], str)

    max_seq_len = max([len(seq) for seq in seqs])

    if maxlen is None:
        maxlen = max_seq_len
    else:
        maxlen = int(maxlen)

    if max_seq_len < maxlen:
        import warnings
        warnings.warn(
            f"Maximum sequence length ({max_seq_len}) is smaller than maxlen ({maxlen})."
        )
        max_seq_len = maxlen

    # check the case when len > 1
    for seq in seqs:
        if not len(seq) % len(value) == 0:
            raise ValueError("All sequences need to be dividable by len(value)")
    if not maxlen % len(value) == 0:
        raise ValueError("maxlen needs to be dividable by len(value)")

    padded_seqs = [
        _trim(_pad(seq, max(max_seq_len, maxlen), value=value, align=align), maxlen, align=align)
        for seq in seqs 
    ]
    return padded_seqs


def _is_overlapping(a, b):
    """Returns True if two intervals overlap"""
    if b[0] >= a[0] and b[0] <= a[1]:
        return True
    else:
        return False


def _merge_intervals(intervals):
    """Merges a list of overlapping intervals"""
    if len(intervals) == 0:
        return None
    merged_list = []
    merged_list.append(intervals[0])
    for i in range(1, len(intervals)):
        pop_element = merged_list.pop()
        if _is_overlapping(pop_element, intervals[i]):
            new_element = pop_element[0], max(pop_element[1], intervals[i][1])
            merged_list.append(new_element)
        else:
            merged_list.append(pop_element)
            merged_list.append(intervals[i])
    return merged_list


def _hamming_distance(string1, string2):
    """Find hamming distance between two strings. Returns inf if they are different lengths"""
    distance = 0
    L = len(string1)
    if L != len(string2):
        return np.inf
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance


def _collapse_pos(positions):
    """Collapse neighbor positions of array to ranges"""
    ranges = []
    start = positions[0]
    for i in range(1, len(positions)):
        if positions[i - 1] == positions[i] - 1:
            continue
        else:
            ranges.append((start, positions[i - 1] + 2))
            start = positions[i]
    ranges.append((start, positions[-1] + 2))
    return ranges


# all below are exact dinuc_shuffle
def _string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def _char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def _one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def _tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]
