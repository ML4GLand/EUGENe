import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MPRADataset(Dataset):
    """MPRA Dataset definition"""
    
    def __init__(self, seqs, targets, rev_comps=None, transform=None):
        """
        Args:
            seqs (iterable): list of sequences to serve as input into models
            targets (iterable): aligned list of targets for each sequence
            rev_comps (iterable, optional): Optional reverse complements of seqs
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.seqs = seqs
        self.rev_comps = rev_comps
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.seqs[idx]
        target = self.targets[idx]
        sample = {"sequence": seq, "target": target}
        if self.rev_comps != None:
            rev_comp = self.rev_comps[idx]
            sample["reverse_complement": rev_comp]
        if self.transform:
            sample = self.transform(sample)
        return sample
        
        

