import pandas as pd
import numpy as np
import torch
from seq_utils import randomizeLinkers, ohe


class ReverseComplement(object):
    """Reverse complement an input sequence"""
    
    def __init__(self, ohe_encoded=False):
        self.ohe = ohe_encoded
        
    def __call__(self, sample):
        sequence, target = sample["sequence"], sample["target"]
        if self.ohe:
            complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
            reverse_complement = "".join(complement.get(base, base) for base in reversed(sequence))
        else:
            complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
            reverse_complement = "".join(complement.get(base, base) for base in reversed(sequence))
        #print(reverse_complement)
        return {'sequence': sequence, 'reverse_complement': reverse_complement, 'target': target}
            

class Augment(object):
    """Perform augmentation of the MPRA dataset using a set of predfined parameters"""
    
    def __init__(self, randomize_linker_p=0.1, modify_features_p=0, enhancer=None):
        self.randomize_linker_p = randomize_linker_p
        self.modify_features_p = modify_features_p
        self.enhancer = enhancer
    
    def __call__(self, sample):
        sequence, target = sample["sequence"], sample["target"]
        seq_len = len(sequence)
        if torch.rand(1).item() < self.randomize_linker_p:
            tsfm_sequence = randomizeLinkers(sequence, enhancer=self.enhancer)
            if len(tsfm_sequence) == seq_len:
                sequence = tsfm_sequence
        #print(sequence)
        return {'sequence': sequence, 'target': target}
            
        
class OneHotEncode(object):
    """OneHotEncode the input sequence"""

    def __call__(self, sample, ohe_axis=1):
        sequence, target = sample["sequence"], sample["target"]
        ohe_sequence = ohe(sequence, one_hot_axis=ohe_axis)
        if "reverse_complement" in sample:
            reverse_complement = sample["reverse_complement"]
            ohe_reverse_complement = ohe(reverse_complement, one_hot_axis=ohe_axis)
            #print(ohe_sequence, ohe_reverse_complement)
            return {'sequence': ohe_sequence,
                    'reverse_complement': ohe_reverse_complement,
                    'target': target}
        else:
            return {'sequence': ohe_sequence, 'target': target}

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, transpose=False):
        self.T = transpose
    
    def __call__(self, sample):
        sequence, target = sample["sequence"], sample["target"]    
        if self.T:
            sequence = sequence.transpose((1, 0))      
        if "reverse_complement" in sample:
            #print("here")
            reverse_complement = sample["reverse_complement"]
            #print(reverse_complement)
            if self.T:
                reverse_complement = reverse_complement.transpose((1, 0))
            #print(sequence)
            return {'sequence': torch.from_numpy(sequence).float(),
                    'reverse_complement': torch.from_numpy(reverse_complement).float(),
                    'target': torch.tensor(target, dtype=torch.float)}
        else:
            return {'sequence': torch.from_numpy(sequence).float(),
                    'target': torch.tensor(target, dtype=torch.float)}