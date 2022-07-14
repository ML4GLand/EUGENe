from __future__ import division, print_function
import re
import logging
import numpy as np
np.random.seed(13)

# Define a few constants:
alphabet = np.array(["A", "G", "C", "T"])
COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


# Find hamming distance between two strings. Returns inf if they are different lengths
def hamming_distance(string1, string2):
    distance = 0
    L = len(string1)
    if L != len(string2):
        return np.inf
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance


# Collapse neighbor positions of array to ranges
def collapse_pos(positions):
    ranges = []
    start = positions[0]
    for i in range(1, len(positions)):
        if positions[i-1] == positions[i]-1:
            continue
        else:
            ranges.append((start, positions[i-1]+2))
            start = positions[i]
    ranges.append((start, positions[-1]+2))
    return ranges

### Sequence encoding from concise

def _get_vocab_dict(vocab):
    return {l: i for i, l in enumerate(vocab)}

def _get_index_dict(vocab):
    return {i: l for i, l in enumerate(vocab)}

def one_hot2token(arr):
    return arr.argmax(axis=2)

def tokenize(seq, vocab, neutral_vocab=[]):
    """Convert sequence to integers
    # Arguments
       seq: Sequence to encode
       vocab: Vocabulary to use
       neutral_vocab: Neutral vocabulary -> assign those values to -1
    # Returns
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
    return [vocab_dict[seq[(i * nchar):((i + 1) * nchar)]] for i in range(len(seq) // nchar)]

def token2one_hot(tvec, vocab_size):
    """
    Note: everything out of the vucabulary is transformed into `np.zeros(vocab_size)`
    """
    arr = np.zeros((len(tvec), vocab_size))

    tvec_range = np.arange(len(tvec))
    tvec = np.asarray(tvec)
    arr[tvec_range[tvec >= 0], tvec[tvec >= 0]] = 1
    return arr

def pad_sequences(sequence_vec, maxlen=None, align="end", value="N"):
    """Pad and/or trim a list of sequences to have common length. Procedure:
        1. Pad the sequence with N's or any other string or list element (`value`)
        2. Subset the sequence
    # Note
        See also: https://keras.io/preprocessing/sequence/
        Aplicable also for lists of characters
    # Arguments
        sequence_vec: list of chars or lists
            List of sequences that can have various lengths
        value: Neutral element to pad the sequence with. Can be `str` or `list`.
        maxlen: int or None; Final lenght of sequences.
             If None, maxlen is set to the longest sequence length.
        align: character; 'start', 'end' or 'center'
            To which end to align the sequences when triming/padding. See examples bellow.
    # Returns
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
        warnings.warn("Maximum sequence length (%s) is less than maxlen (%s)" % (max_seq_len, maxlen))
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

    padded_sequence_vec = [pad(seq, max(max_seq_len, maxlen),
                               value=value, align=align) for seq in sequence_vec]
    padded_sequence_vec = [trim(seq, maxlen, align=align) for seq in padded_sequence_vec]

    return padded_sequence_vec


### Random sequence generations

def random_base():
    """
    Generate a random base.
    :return: Random base.
    """
    return np.random.choice(alphabet)

def random_seq(seq_len):
    """
    Generate a random sequence of length seq_len.
    :args
    seq_len (int): length of sequence to return
    :return: Random sequence.
    """
    return "".join([np.random.choice(alphabet) for i in range(seq_len)])

def random_seqs(seq_num, seq_len):
    """
    Generate seq_num random sequences of length seq_len
    :args
    seq_num (int): number of sequences to return
    seq_len (int): length of sequence to return
    :return: numpy array of random sequences.
    """
    return [random_seq(seq_len) for i in range(seq_num)]

def random_seqs_to_file(file, ext="csv", **kwargs):
    """
    Generate a random sequence of length seq_len and save to file
    :args
    seq_len (int): length of sequence to return
    :return: Random sequence.
    """
    pass


### Useful functions on sequences

def reverse_complement(seq, copy=False):
    return "".join(COMPLEMENT.get(base, base) for base in reversed(seq))

def reverse_complement_seqs(seqs, copy=False):
    return np.array([reverse_complement(seq) for seq in seqs])


### Dinuc shuffle from Kundaje lab

def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)

def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")

def one_hot_to_tokens(one_hot):
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

def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

def dinuc_shuffle(seq, num_shufs=None, rng=None):
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


### Enhancer Utils

enhancer_binding_sites = {"Core-otx-a": ["..GGAA..", "..GGAT..", "..TTCC..", "..ATCC..", "..GATA..", "..TATC.."],
                          "WT-otx-a": ["GTTATCTC", "ACGGAAGT", "AAGGAAAT", "AATATCT", "AAGATAGG", "GAGATAAC", "ACTTCCGT", "ATTTCCTT", "AGATATT", "CCTATCTT"]}

def is_overlapping(a, b):
    if b[0] >= a[0] and b[0] <= a[1]:
        return True
    else:
        return False

def merge_intervals(intervals):
    if len(intervals) == 0:
        return None
    merged_list= []
    merged_list.append(intervals[0])
    for i in range(1, len(intervals)):
        pop_element = merged_list.pop()
        if is_overlapping(pop_element, intervals[i]):
            new_element = pop_element[0], max(pop_element[1], intervals[i][1])
            merged_list.append(new_element)
        else:
            merged_list.append(pop_element)
            merged_list.append(intervals[i])
    return merged_list

def randomizeLinkers(seq, features=None, enhancer=None):
    if features == None:
        assert enhancer != None
        features = enhancer_binding_sites[enhancer]

    transformed_seq = []
    feature_spans = merge_intervals([x.span() for x in re.finditer(r"("+'|'.join(features)+r")", seq)])
    if feature_spans is None:
        return seq
    for i, span in enumerate(feature_spans):
        if i == 0:
            linker_len = span[0]
        else:
            linker_len = feature_spans[i][0]-feature_spans[i-1][1]
        transformed_seq.append("".join(np.random.choice(alphabet, size=linker_len)))
        transformed_seq.append(seq[span[0]:span[1]])
    transformed_seq.append("".join(np.random.choice(alphabet, size=len(seq)-feature_spans[-1][1])))
    transformed_seq = "".join(transformed_seq)
    if len(transformed_seq) != len(seq):
        logging.warning('Transformed sequence is length {}'.format(len(transformed_seq)))
    return transformed_seq
