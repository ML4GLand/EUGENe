from __future__ import division, print_function
import numpy as np

np.random.seed(13)


def ascii_encode(seq, pad=0):
    """
    Converts a string of characters to a NumPy array of byte-long ASCII codes.
    """
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(
            encode_seq, pad_width=(0, pad), mode="constant", constant_values=36
        )
    return encode_seq


def ascii_decode(seq):
    """
    Converts a NumPy array of byte-long ASCII codes to a string of characters.
    """
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")


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


def _get_vocab_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    """
    return {l: i for i, l in enumerate(vocab)}


def _get_index_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    """
    return {i: l for i, l in enumerate(vocab)}


def _one_hot2token(arr):
    return arr.argmax(axis=2)


def _tokenize(seq, vocab, neutral_vocab=[]):
    """
    Convert sequence to integers
    Arguments
        seq: Sequence to encode
        vocab: Vocabulary to use
        neutral_vocab: Neutral vocabulary -> assign those values to -1

    Returns
        List of length `len(seq)` with integers from `-1` to `len(vocab) - 1`
    """
    # Req: all vocabs have the same length
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


def _token2one_hot(tvec, vocab_size, fill_value):
    """
    Note: everything out of the vocabulary is transformed into `np.zeros(vocab_size)`
    """
    arr = np.zeros((len(tvec), vocab_size))

    tvec_range = np.arange(len(tvec))
    tvec = np.asarray(tvec)
    arr[tvec_range[tvec >= 0], tvec[tvec >= 0]] = 1
    if fill_value is not None:
        arr[tvec_range[tvec < 0]] = fill_value

    return arr.astype(np.int8) if fill_value is None else arr.astype(np.float16)


def _pad_sequences(sequence_vec, maxlen=None, align="end", value="N"):
    """
    Pad and/or trim a list of sequences to have common length. Procedure:
        1. Pad the sequence with N's or any other string or list element (`value`)
        2. Subset the sequence
    Note
        See also: https://keras.io/preprocessing/sequence/
        Aplicable also for lists of characters
    Arguments
        sequence_vec: list of chars or lists
            List of sequences that can have various lengths
        value: Neutral element to pad the sequence with. Can be `str` or `list`.
        maxlen: int or None; Final lenght of sequences.
             If None, maxlen is set to the longest sequence length.
        align: character; 'start', 'end' or 'center'
            To which end to align the sequences when triming/padding. See examples bellow.
    Returns
        List of sequences of the same class as sequence_vec
    # Example
        ```python
            >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
            >>> pad_sequences(sequence_vec, 10, align="start", value="N")
            ['CTTACTCAGA', 'TCTTTANNNN']
            >>> pad_sequences(sequence_vec, 10, align="end", value="N")
            ['CTTACTCAGA', 'NNNNTCTTTA']
            >>> pad_sequences(sequence_vec, 4, align="center", value="N")
            ['ACTC', 'CTTT']
        ```
    """

    # neutral element type checking
    assert isinstance(value, list) or isinstance(value, str)
    assert isinstance(value, type(sequence_vec[0])) or type(sequence_vec[0]) is np.str_
    assert not isinstance(sequence_vec, str)
    assert isinstance(sequence_vec[0], list) or isinstance(sequence_vec[0], str)

    max_seq_len = max([len(seq) for seq in sequence_vec])

    if maxlen is None:
        maxlen = max_seq_len
    else:
        maxlen = int(maxlen)

    if max_seq_len < maxlen:
        import warnings

        warnings.warn(
            "Maximum sequence length (%s) is less than maxlen (%s)"
            % (max_seq_len, maxlen)
        )
        max_seq_len = maxlen

    # check the case when len > 1
    for seq in sequence_vec:
        if not len(seq) % len(value) == 0:
            raise ValueError("All sequences need to be dividable by len(value)")
    if not maxlen % len(value) == 0:
        raise ValueError("maxlen needs to be dividable by len(value)")

    # pad and subset
    def pad(seq, max_seq_len, value="N", align="end"):
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

    def trim(seq, maxlen, align="end"):
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

    padded_sequence_vec = [
        pad(seq, max(max_seq_len, maxlen), value=value, align=align)
        for seq in sequence_vec
    ]
    padded_sequence_vec = [
        trim(seq, maxlen, align=align) for seq in padded_sequence_vec
    ]

    return padded_sequence_vec


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
