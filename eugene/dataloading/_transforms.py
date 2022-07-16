# PyTorch
import torch

# EUGENE
from ..preprocessing._otx_preprocess import randomizeLinkers
from ..preprocessing._seq_preprocess import ohe_DNA_seq

# Suite of sequence transforms that can be composed using torchvision
class ReverseComplement(object):
    """Reverse complement an input sequence"""

    def __init__(self, ohe_encoded=False, **kwargs):
        self.ohe = ohe_encoded

    def __call__(self, sample):
        seq = sample[1]
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        rev_seq = "".join(complement.get(base, base) for base in reversed(seq))
        sample[2] = rev_seq
        return sample


class Augment(object):
    """Perform augmentation of the MPRA dataset using a set of predfined parameters"""

    def __init__(self, randomize_linker_p=0.1, modify_features_p=0, enhancer=None, **kwargs):
        self.randomize_linker_p = randomize_linker_p
        self.modify_features_p = modify_features_p
        self.enhancer = enhancer

    def __call__(self, sample):
        sequence = sample[1]
        seq_len = len(sequence)
        if torch.rand(1).item() < self.randomize_linker_p:
            tsfm_sequence = randomizeLinkers(sequence, enhancer=self.enhancer)
            if len(tsfm_sequence) == seq_len:
                sequence = tsfm_sequence
        sample[1] = sequence
        return sample


class OneHotEncode(object):
    """OneHotEncode the input sequence"""
    def __init__(self, **kwargs):
        pass

    def __call__(self, sample):
        sequence = sample[1]
        ohe_seq =  ohe_DNA_seq(sequence)
        sample[1] = ohe_seq
        if len(sample[2]) != 1:
            rev_seq = sample[2]
            ohe_rev_seq =  ohe_DNA_seq(rev_seq)
            sample[2] = ohe_rev_seq
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, transpose=False, **kwargs):
        self.T = transpose

    def __call__(self, sample):
        if self.T:
            sample[1] = sample[1].transpose((1, 0))
        if len(sample[2]) != 1 and self.T:
                sample[2] = sample[2].transpose((1, 0))
        return torch.from_numpy(sample[0]).float(), torch.from_numpy(sample[1]).float(), torch.from_numpy(sample[2]).float(), torch.tensor(sample[3], dtype=torch.float)
