import tqdm
import pandas as pd
import numpy as np
from ._utils import _get_vocab_dict, _get_index_dict, one_hot2token, tokenize, token2one_hot, pad_sequences
from ..utils import loadSiteName2bindingSiteSequence, loadBindingSiteName2affinities, encode_seq, encode_OLS_seq

### One-hot feature encoding (not sequence)

# Fit to overall dataframe
def encodeBlock():
    ohe_block = OneHotEncoder(sparse=False)
    X = OLS_data[block_features]
    ohe_block.fit(X)
    X_block = ohe_block.fit_transform(X)


### Mixed encodings

# Wrapper function to generate mixed 1.0-3.0 encodings
# Currently supports encoding into mixed 1.0, 2.0 and 3.0
def mixed_encode(data):
    mixed1_encoding, mixed2_encoding, mixed3_encoding, valid_idx = [], [], [], []
    for i, (row_num, enh_data) in tqdm.tqdm(enumerate(data.iterrows())):
        sequence = enh_data["SEQ"].upper().strip()
        encoded_seq1 = encode_seq(sequence, encoding="mixed1")
        encoded_seq2 = encode_seq(sequence, encoding="mixed2")
        encoded_seq3 = encode_seq(sequence, encoding="mixed3")
        if encoded_seq1 != -1:
            mixed1_encoding.append(encoded_seq1)
            mixed2_encoding.append(encoded_seq2)
            mixed3_encoding.append(encoded_seq3)
            valid_idx.append(i)
    X_mixed1 = (pd.DataFrame(mixed1_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    X_mixed2 = (pd.DataFrame(mixed2_encoding).replace({"R": -1, "F": 1})).values
    X_mixed3 = (pd.DataFrame(mixed3_encoding).replace({"R": 0, "F": 1})).values
    return X_mixed1, X_mixed2, X_mixed3, valid_idx

# Wrapper function to encode all three mixed encodings for the OLS library. \
# Currrently supports mixed 1.0, 2.0 and 3.0
def mixed_OLS_encode(OLS_dataset):
    site_dict = loadSiteName2bindingSiteSequence()  # Sitenames to sequence
    aff_dict = loadBindingSiteName2affinities()  # Sitenames to affinities
    mixed1_encoding, mixed2_encoding, mixed3_encoding = [], [], []
    for i, (row_num, enh_data) in enumerate(tqdm.tqdm(OLS_dataset.iterrows())):
        sequence = enh_data.values
        encoded_seq1 = encode_OLS_seq(sequence, encoding="mixed1", sitename_dict=site_dict, affinity_dict=aff_dict)
        encoded_seq2 = encode_OLS_seq(sequence, encoding="mixed2", sitename_dict=site_dict, affinity_dict=aff_dict)
        encoded_seq3 = encode_OLS_seq(sequence, encoding="mixed3", sitename_dict=site_dict, affinity_dict=aff_dict)
        mixed1_encoding.append(encoded_seq1)
        mixed2_encoding.append(encoded_seq2)
        mixed3_encoding.append(encoded_seq3)
    X_mixed1s = (pd.DataFrame(mixed1_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    X_mixed2s = (pd.DataFrame(mixed2_encoding).replace({"R": -1, "F": 1})).values
    X_mixed3s = (pd.DataFrame(mixed3_encoding).replace({"R": 0, "F": 1})).values
    return X_mixed1s, X_mixed2s, X_mixed3s

# Wrapper function to encode all three mixed encodings for the OLS library. \
# Currrently supports mixed 1.0, 2.0 and 3.0
def otx_encode(seqs):
    mixed1_encoding, mixed2_encoding, mixed3_encoding, valid_idx = [], [], [], []
    for i, sequence in tqdm.tqdm(enumerate(seqs)):
        encoded_seq1 = encode_seq(sequence, encoding="mixed1")
        encoded_seq2 = encode_seq(sequence, encoding="mixed2")
        encoded_seq3 = encode_seq(sequence, encoding="mixed3")
        if encoded_seq1 != -1:
            mixed1_encoding.append(encoded_seq1)
            mixed2_encoding.append(encoded_seq2)
            mixed3_encoding.append(encoded_seq3)
            valid_idx.append(i)
    X_mixed1 = (pd.DataFrame(mixed1_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    X_mixed2 = (pd.DataFrame(mixed2_encoding).replace({"R": -1, "F": 1})).values
    X_mixed3 = (pd.DataFrame(mixed3_encoding).replace({"R": 0, "F": 1})).values
    return X_mixed1, X_mixed2, X_mixed3, valid_idx


### One-hot encodings

# vocabularies:
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

def encodeSequence(seq_vec, vocab, neutral_vocab, maxlen=None,
                   seq_align="start", pad_value="N", encode_type="one_hot"):
    """Convert a list of genetic sequences into one-hot-encoded array.
    # Arguments
       seq_vec: list of strings (genetic sequences)
       vocab: list of chars: List of "words" to use as the vocabulary. Can be strings of length>0,
            but all need to have the same length. For DNA, this is: ["A", "C", "G", "T"].
       neutral_vocab: list of chars: Values used to pad the sequence or represent unknown-values. For DNA, this is: ["N"].
       maxlen: int or None,
            Should we trim (subset) the resulting sequence. If None don't trim.
            Note that trims wrt the align parameter.
            It should be smaller than the longest sequence.
       seq_align: character; 'end' or 'start'
            To which end should we align sequences?
       encode_type: "one_hot" or "token". "token" represents each vocab element as a positive integer from 1 to len(vocab) + 1.
                  neutral_vocab is represented with 0.
    # Returns
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

def oheDNA(seq, vocab=DNA, neutral_vocab="N"):
    return token2one_hot(tokenize(seq, vocab, neutral_vocab), len(vocab))

def decodeOHE(arr, vocab=DNA, neutral_vocab="N"):
    tokens = arr.argmax(axis=1)
    indexToLetter = _get_index_dict(vocab)
    return ''.join([indexToLetter[x] for x in tokens])

def encodeDNA(seq_vec, maxlen=None, seq_align="start"):
    """Convert the DNA sequence into 1-hot-encoding numpy array
    # Arguments
        seq_vec: list of chars
            List of sequences that can have different lengths
        maxlen: int or None,
            Should we trim (subset) the resulting sequence. If None don't trim.
            Note that trims wrt the align parameter.
            It should be smaller than the longest sequence.
        seq_align: character; 'end' or 'start'
            To which end should we align sequences?
    # Returns
        3D numpy array of shape (len(seq_vec), trim_seq_len(or maximal sequence length if None), 4)
    # Example
        ```python
            >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
            >>> X_seq = encodeDNA(sequence_vec, seq_align="end", maxlen=8)
            >>> X_seq.shape
            (2, 8, 4)
            >>> print(X_seq)
            [[[0 0 0 1]
              [1 0 0 0]
              [0 1 0 0]
              [0 0 0 1]
              [0 1 0 0]
              [1 0 0 0]
              [0 0 1 0]
              [1 0 0 0]]
             [[0 0 0 0]
              [0 0 0 0]
              [0 0 0 1]
              [0 1 0 0]
              [0 0 0 1]
              [0 0 0 1]
              [0 0 0 1]
              [1 0 0 0]]]
        ```
    """
    return encodeSequence(seq_vec,
                          vocab=DNA,
                          neutral_vocab="N",
                          maxlen=maxlen,
                          seq_align=seq_align,
                          pad_value="N",
                          encode_type="one_hot")

def decodeDNA(arr, vocab=DNA):
    """Convert a one-hot encoded array back to string
    """
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)
    return [''.join([indexToLetter[x] for x in row]) for row in tokens]


### Numerical encodings

def ascii_encode(seq, pad=0):
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(encode_seq, pad_width=(0, pad), mode="constant", constant_values=36)
    return encode_seq

def ascii_decode(seq):
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")
