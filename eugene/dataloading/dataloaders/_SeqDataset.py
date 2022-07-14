import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ...preprocessing._seq_preprocess import ascii_encode


class SeqDataset(Dataset):
    """Sequence based PyTorch dataset definition"""

    def __init__(self, seqs, names=None, targets=None, rev_seqs=None, transform=None):
        """
        Args:
            seqs (iterable): list of sequences to serve as input into models
            names (iterable, optional): list of identifiers for sequences
            targets (iterable): aligned list of targets for each sequence
            rev_seqs (iterable, optional): Optional reverse complements of seqs
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.names = names
        self.seqs = seqs
        self.rev_seqs = rev_seqs
        self.targets = targets
        self.transform = transform
        self._init_dataset()

    def _init_dataset(self):
        """Perform any initialization steps on the dataset, currently converts names into ascii if provided
        """

        if self.names is not None:
            self.name_lengths = np.array([len(i) for i in self.names])
            if np.any(self.name_lengths != self.name_lengths[0]):
                self.longest_name = np.max(self.name_lengths)
                self.ascii_names = np.zeros((len(self.names), self.longest_name))
                for i, name in enumerate(self.names):
                    #np_name = np.array([ord(letter) for letter in name], dtype=int)
                    #self.ascii_names[i] = np.pad(np_name, pad_width=(0, self.longest_name-len(np_name)), mode="constant", constant_values=36)
                    pad_len = self.longest_name-len(name)
                    self.ascii_names[i] = ascii_encode(name, pad_len)
            else:
                #self.ascii_names = np.array([np.array([ord(letter) for letter in name], dtype=int) for name in self.names])
                self.ascii_names = np.array([ascii_encode(name) for name in self.names])
        else:
            self.ascii_names = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        """Get an item from the dataset and return as tuple. Perform any transforms passed in

        Args:
            idx (int): dataset index to grab

        Returns:
            tuple: Returns a quadruple of tensors: identifiers, sequences, reverse complement
                   sequences, targets. If any are not provided tensor([-1.]) is returned for that sequence
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.seqs[idx]

        if self.ascii_names is not None:
            name = self.ascii_names[idx]
        else:
            name = np.array([-1.0])

        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = np.array([-1.0])

        if self.rev_seqs is not None:
            rev_seq = self.rev_seqs[idx]
        else:
            rev_seq = np.array([-1.0])

        sample = np.array([name, seq, rev_seq, target], dtype=object)

        if self.transform:
            sample = self.transform(sample)
        return sample


    def to_dataloader(self, batch_size=32, pin_memory=True, shuffle=False, num_workers=0):
        """Convert the dataset to a PyTorch DataLoader"""
        return DataLoader(self, batch_size=batch_size, pin_memory=True, shuffle=shuffle, num_workers=num_workers)
