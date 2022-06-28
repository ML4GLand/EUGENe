"""
Tests to make sure the example dataloading routines work.
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path
from torchvision import transforms

HERE = Path(__file__).parent

def test_SeqDataset():
    names, seqs, rev_seqs, targets = eu.datasets.load(f"{HERE}/_data/test_1000seqs_66/test_ohe_seqs.npy", names_file=f"{HERE}/_data/test_1000seqs_66/test_ids.npy", target_file=f"{HERE}/_data/test_1000seqs_66/test_labels.npy", rev_seq_file=f"{HERE}/_data/test_1000seqs_66/test_rev_ohe_seqs.npy")
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))
    dataset = eu.dataloading.dataloaders.SeqDataset(names=names, seqs=seqs, targets=targets, rev_seqs=rev_seqs, transform=None)
    assert(len(dataset[0]) == 4)


def test_SeqDataModule():
    data_transform = transforms.Compose([eu.dataloading.ToTensor(transpose=True)])
    datamodule = eu.dataloading.dataloaders.SeqDataModule(seq_file=f"{HERE}/_data/test_1000seqs_66/test_ohe_seqs.npy", batch_size=32, transform=data_transform)
    datamodule.setup()
    dataset = datamodule.train_dataloader().dataset
    assert(len(dataset[0]) == 4)
