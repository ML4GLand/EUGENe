import torch
from ..._settings import settings
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """PyTorch dataset class definition for sequences.

    Attributes
    ----------
    seqs : iterable
        List of sequences to serve as input into models.
    targets : iterable
        List of targets for sequences.
    transforms : callable, optional
        Optional transform to be applied on a sample.
    """

    def __init__(
        self, 
        seqs, 
        targets=None, 
        transforms=None
    ):
        self.seqs = seqs
        self.targets = targets
        self.transforms = transforms

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
            Returns a 
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        if isinstance(self.seqs[idx], str):
            sample["seq"] = self.seqs[idx]
        else:
            sample["seq"] = torch.tensor(self.seqs[idx], dtype=torch.float32)

        if self.targets is not None:
            sample["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            sample["target"] = torch.tensor(-1.0, dtype=torch.float32)
        
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def to_dataloader(
        self, 
        batch_size=None, 
        pin_memory=True, 
        shuffle=True, 
        num_workers=0, 
        **kwargs
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
