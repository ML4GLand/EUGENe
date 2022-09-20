import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ...preprocess import ascii_encode
from ..._settings import settings


class SeqDataset(Dataset):
    """
    PyTorch dataset definition for sequences.

    Parameters
    ----------
    seqs : iterable
        List of sequences to serve as input into models.
    names : iterable
        List of identifiers for sequences.
    targets : iterable
        List of targets for sequences.
    rev_seqs : iterable, optional
        Optional reverse complements of sequences.
    transforms : callable, optional
        Optional transform to be applied on a sample.
    """

    def __init__(self, seqs, names=None, targets=None, rev_seqs=None, transform=None):
        self.names = names
        self.seqs = seqs
        self.rev_seqs = rev_seqs
        self.targets = targets
        self.transform = transform
        self._init_dataset()

    def _init_dataset(self):
        """Perform any initialization steps on the dataset.
        Currently converts names into ascii if provided

        Returns:
            None
        """

        if self.names is not None:
            self.name_lengths = np.array([len(i) for i in self.names])
            if np.any(self.name_lengths != self.name_lengths[0]):
                self.longest_name = np.max(self.name_lengths)
                self.ascii_names = np.zeros((len(self.names), self.longest_name))
                for i, name in enumerate(self.names):
                    pad_len = self.longest_name - len(name)
                    self.ascii_names[i] = ascii_encode(name, pad_len)
            else:
                self.ascii_names = np.array([ascii_encode(name) for name in self.names])
        else:
            self.ascii_names = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        """Get an item from the dataset and return as tuple. Perform any transforms passed in

        Parameters:
        ----------
        idx (int):
            dataset index to grab

        Returns:
        tuple:
            Returns a quadruple of tensors: identifiers, sequences, reverse complement
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

    def to_dataloader(
        self, batch_size=None, pin_memory=True, shuffle=True, num_workers=0, **kwargs
    ):
        """Convert the dataset to a PyTorch DataLoader

        Parameters:
        ----------
        batch_size (int, optional):
            batch size for dataloader
        pin_memory (bool, optional):
            whether to pin memory for dataloader
        shuffle (bool, optional):
            whether to shuffle the dataset
        num_workers (int, optional):
            number of workers for dataloader
        **kwargs:
            additional arguments to pass to DataLoader
        """
        batch_size = batch_size if batch_size is not None else settings.batch_size
        return DataLoader(
            self,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
