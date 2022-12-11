from .._settings import settings
from ..dataload import SeqData
from ..models.base import BaseModel
import eugene.preprocess as pp
import torch
import numpy as np

def count_kmers(sequence : str, k : int, data = None) -> dict:
    """
    Counts k-mers in a given sequence.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence expressed as a string.
    k : int
        k value for k-mers (e.g. k=3 generates 3-mers).

    Returns
    -------
    kmers : dict
        k-mers and their counts expressed in a dictionary.
    """
    assert len(sequence) >= k, "Length of sequence must be greater than that of k."
    data = {} if data is None else data
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i: i + k]
        try:
            data[kmer] += 1
        except KeyError:
            data[kmer] = 1
    return data

def count_kmers_sdata(sdata : SeqData, k : int, frequency : bool = False) -> dict:
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

def _find_distance(seq1, seq2) -> int:
    edits = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            edits += 1
    return edits

def edit_distance(seq1 : str, seq2 : str, dual : bool = False) -> int:
    """
    Calculates the nucleotide edit distance between two sequences.

    Parameters
    ----------
    seq1 : str
        First nucleotide sequence expressed as a string.
    seq2 : str
        Second ucleotide sequence expressed as a string.
    dual : bool
        Whether to calculate the forwards and backwards edit distance, and return the lesser.
        Defaults to False.

    Returns
    -------
    edits : int
        Amount of edits between sequences.
    """
    assert len(seq1) == len(seq2), "Both sequences must be of same length."
    f_edits = _find_distance(seq1, seq2)
    b_edits = _find_distance(seq1, seq2[::-1]) if dual else len(seq1)
    return min(f_edits, b_edits)

def edit_distance_sdata(sdata1 : SeqData, sdata2 : SeqData, dual : bool = False, average : bool = False) -> list:
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
    assert len(sdata1.seqs) == len(sdata2.seqs), "Both SeqData objects must be of the same length."
    distances = []
    for i in range(len(sdata1.seqs)):
        distances.append(edit_distance(sdata1.seqs[i], sdata2.seqs[i], dual))
    if average:
        return sum(distances) / len(distances)
    return distances

def latent_interpolation(latent_dim : int, samples : int, num_seqs : int = 1, model : BaseModel = None, normal : bool = False, inclusive : bool = True) -> list:
    """
    Linearly interpolates between two random latent points. Useful for visualizing generative models.

    Parameters
    ----------
    latent_dim : int
        Latent dimension of random latent space points.
    samples : int
        Number of samples to make between the two latent points. Higher numbers represent more sequences and should show smoother results.
    num_seqs : int
        Number of sequence channels to interpolate.
        Defaults to 1.
    model : BaseModel
        If provided, interpolated values will be passed through the given model and be returned in place of latent points.
        Default is None.
    normal : bool
        Whether to randomly sample points from a standard normal distibution rather than from 0 to 1.
        Defaults to False.
    inclusive : bool
        Whether to returm random latent points along with their interpolated samples.
        Defaults to True.

    Returns
    -------
    z_list : list
        List of latent space points represented as tensors.
    gen_seqs : list
        List of tokenized sequences expressed as strings.
        Returns in place of z_list when a model is provided.
    """
    if normal:
        z1 = torch.normal(0, 1, (num_seqs, latent_dim))
        z2 = torch.normal(0, 1, (num_seqs, latent_dim))
    else:
        z1 = torch.rand(num_seqs, latent_dim)
        z2 = torch.rand(num_seqs, latent_dim)

    z_list = []
    for n in range(samples):
        weight = (n + 1)/(samples + 1)
        z_interp = torch.lerp(z1, z2, weight)
        z_list.append(z_interp)
    if inclusive:
        z_list.insert(0, z1)
        z_list.append(z2)

    if model is None:
        return z_list

    gen_seqs = []
    for z in z_list:
        gen_seq = seqs_from_tensor(model(z))
        gen_seqs.append(gen_seq)
    return gen_seqs

def seqs_from_tensor(tensor : torch.tensor, num_seqs : int = 1) -> np.ndarray:
    """
    Decodes sequences represented by tensors into their string values.

    Parameters
    ----------
    tensor : torch.tensor
        Tensor to be decoded.
    num_seqs : int
        Number of sequences to decode.
        Default is 1.

    Returns
    -------
    seqs : np.ndarray
        Numpy array of decoded sequences.
    """
    tokens = np.argmax(tensor.detach().numpy(), axis=1).reshape(num_seqs, -1)
    seqs = np.array([pp.decode_seq(pp._utils._token2one_hot(token)) for token in tokens])
    return seqs

def generate_seqs_from_model(model : BaseModel, num_seqs : int = 1, normal : bool = False):
    """
    Generates random sequences from a generative model.

    Parameters
    ----------
    model : BaseModel
        Generative model used for sequence generation.
    num_seqs : int
        Number of sequences to decode.
        Default is 1.
    normal : bool
        Whether to sample from a normal distribution instead of a uniform distribution.
        Default is false.

    Returns
    -------
    seqs : np.ndarray
        Numpy array of decoded sequences.
    """
    if normal:
        z = torch.Tensor(np.random.normal(0, 1, (num_seqs, model.latent_dim)))
    else: 
        z = torch.rand(num_seqs, model.latent_dim)
    fake = model(z)
    return seqs_from_tensor(fake, num_seqs)
