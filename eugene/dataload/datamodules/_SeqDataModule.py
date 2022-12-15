import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from .._io import read
from ._SeqDataset import SeqDataset
from .._transforms import ReverseComplement, Augment, OneHotEncode, ToTensor


class SeqDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_file: str,
        batch_size: int = 32,
        num_workers: int = 0,
        transform=None,
        test=False,
        shuffle=True,
        split=0.9,
        seed=13,
        save_names=None,
        read_kwargs={},
    ):
        """Sequence PyTorch Lightning DataModule definition

        Parameters:
        seq_file (str):
            file path to load
        batch_size (int, optional):
            Defaults to 32.
        num_workers (int, optional):
            Defaults to 0.
        transform (_type_, optional):
            seq_transforms to perform. Defaults to None.
        split (float, optional):
            train/validation split. Defaults to 0.9.
        read_kwargs (dict, optional):
            Optional keyword arugments to pass to load function. Defaults to {}.
        """
        super().__init__()
        self.seq_file = seq_file
        self.batch_size = batch_size
        if transform is None:
            self.transform = transforms.Compose(
                [
                    ReverseComplement(ohe_encoded=False),
                    OneHotEncode(),
                    ToTensor(transpose=True),
                ]
            )
        elif isinstance(transform, list):
            transform_classes = []
            for t in transform:
                kwargs = t.get("init_args", {})
                class_module, class_name = t["class_path"].rsplit(".", 1)
                module = __import__(class_module, fromlist=[class_name])
                args_class = getattr(module, class_name)
                transform_classes.append(args_class(**kwargs))
            self.transform = transforms.Compose(transform_classes)
        else:
            self.transform = transform
        self.read_kwargs = read_kwargs
        self.num_workers = num_workers
        self.test = test
        self.shuffle = shuffle
        self.split = split
        self.seed = seed
        self.save_names = save_names

    def setup(self, stage: str = None) -> None:
        """Setup the data module.

        Parameters:
        ----------
        stage (str, optional):
            Defaults to None.
        """
        names, seqs, rev_seqs, targets = read(self.seq_file, **self.read_kwargs)
        dataset = SeqDataset(seqs, names, targets, rev_seqs, transform=self.transform)
        if self.test:
            self.test = dataset
        else:
            dataset_len = len(dataset)
            train_len = int(dataset_len * self.split)
            val_len = dataset_len - train_len
            self.train, self.val = random_split(
                dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(self.seed),
            )
            if self.save_names is not None:
                if not os.path.exists(self.save_names):
                    os.makedirs(self.save_names)
                np.savetxt(
                    os.path.join(self.save_names, "train.txt"),
                    self.train.dataset.names[self.train.indices],
                    fmt="%s",
                )
                np.savetxt(
                    os.path.join(self.save_names, "val.txt"),
                    self.val.dataset.names[self.val.indices],
                    fmt="%s",
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
