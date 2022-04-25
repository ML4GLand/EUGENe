import numpy as np
np.random.seed(42)

# Definitions
DEFAULT_NUC_ORDER = {y: x for x, y in enumerate(["A", "T", "C", "G"])}
NUCLEOTIDES = sorted([x for x in DEFAULT_NUC_ORDER.keys()])
COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def random_base():
    """
    Generate a random base.
    :return: Random base.
    """
    return np.random.choice(NUCLEOTIDES)

def random_seq(seq_len):
    """
    Generate a random sequence of length seq_len.
    :args
    seq_len (int): length of sequence to return
    :return: Random sequence.
    """
    return "".join([np.random.choice(NUCLEOTIDES) for i in range(seq_len)])

def random_seqs(seq_num, seq_len):
    """
    Generate seq_num random sequences of length seq_len
    :args
    seq_num (int): number of sequences to return
    seq_len (int): length of sequence to return
    :return: numpy array of random sequences.
    """
    return np.array([random_seq(seq_len) for i in range(seq_num)])

def random_seqs_to_file(file, ext="csv", **kwargs):
    """
    Generate a random sequence of length seq_len and save to file
    :args
    seq_len (int): length of sequence to return
    :return: Random sequence.
    """
    pass

def reverse_complement(seq):
    return "".join(COMPLEMENT.get(base, base) for base in reversed(seq))

def seq2Fasta(seqs, IDs, name="seqs"):
    file = open("{}.fa".format(name), "w")
    for i in range(len(seqs)):
        file.write(">" + IDs[i] + "\n" + seqs[i] + "\n")
    file.close()

def ohe(sequence, one_hot_axis=1):
    zeros_array = np.zeros((len(sequence),4), dtype=np.int8)
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)

    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            continue
            #raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1
    return zeros_array

def ascii_encode(seq, pad=0):
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(encode_seq, pad_width=(0, pad), mode="constant", constant_values=36)
    return encode_seq

def ascii_decode(seq):
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")
