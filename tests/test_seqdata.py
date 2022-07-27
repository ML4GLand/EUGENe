"""
Tests to make sure the example SeqData object works
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path
import pyranges as pr

HERE = Path(__file__).parent


def test_init():
    names, seqs, rev_seqs, targets = eu.dl.read_numpy(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.npy",
        names_file=f"{HERE}/../eugene/datasets/random1000/random1000_ids.npy",
        target_file=f"{HERE}/../eugene/datasets/random1000/random1000_labels.npy",
        rev_seq_file=f"{HERE}/../eugene/datasets/random1000/random1000_rev_seqs.npy",
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)
    sdata = eu.dl.SeqData(
        names=names,
        seqs=seqs,
        seqs_annot=targets,
        rev_seqs=rev_seqs,
        pos_annot=f"{HERE}/../eugene/datasets/random1000/random1000_pos_annot.bed",
    )


@pytest.fixture
def sdata():
    """
    sdata
    """
    names, seqs, rev_seqs, targets = eu.dl.read_numpy(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.npy",
        names_file=f"{HERE}/../eugene/datasets/random1000/random1000_ids.npy",
        target_file=f"{HERE}/../eugene/datasets/random1000/random1000_labels.npy",
        rev_seq_file=f"{HERE}/../eugene/datasets/random1000/random1000_rev_seqs.npy",
        return_numpy=True,
    )
    sdata = eu.dl.SeqData(names=names, seqs=seqs, seqs_annot=targets, rev_seqs=rev_seqs)
    return sdata


def test_print(sdata):
    print(sdata)
    sdata.pos_annot = pr.read_bed(
        f"{HERE}/../eugene/datasets/random1000/random1000_pos_annot.bed"
    )
    print(sdata)


def test_write_h5sd(sdata):
    sdata.write_h5sd(f"{HERE}/../eugene/datasets/random1000/random1000_seqs.h5sd")


def test_to_dataset(sdata):
    sdataset = sdata.to_dataset(
        target=0,
        seq_transforms=["augment", "one_hot_encode"],
        transform_kwargs={"enhancer": "Core-otx-a"},
    )
    assert sdataset[0]

    # Instantiate a DataLoader
    import torch
    from torch.utils.data import DataLoader

    test_dataloader = DataLoader(sdataset, batch_size=32, shuffle=True, num_workers=0)
    # Check the DataLoader
    for i_batch, sample_batched in enumerate(test_dataloader):
        assert sample_batched[0].size() == torch.Size([32, 6])
        assert sample_batched[1].size() == torch.Size([32, 66, 4])
        assert sample_batched[2].size() == torch.Size([32, 66, 4])
        assert sample_batched[3].size() == torch.Size([32])
        if i_batch == 3:
            break
