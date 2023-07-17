"""
Tests to make sure the dataload module isn't busted
"""

import os
import torch
import numpy as np
import eugene as eu
import pytest
from pathlib import Path
from eugene.dataload import SeqData, SeqDataset

HERE = Path(__file__).parent


def check_random1000_load(sdata, has_target=False):
    assert isinstance(sdata, SeqData)
    assert sdata.n_obs == 1000
    assert sdata.names[-1] == "seq999"
    if has_target:
        assert sdata.seqs_annot.iloc[:, -1][0] is not np.nan


def test_read():
    dataset_dir = f"{HERE}/../eugene/datasets/random1000"
    sdata = eu.dl.read(os.path.join(dataset_dir, "random1000_seqs.tsv"))
    check_random1000_load(sdata)


def test_read_csv():
    dataset_dir = f"{HERE}/../eugene/datasets/random1000"
    sdata = eu.dl.read_csv(
        filename=os.path.join(dataset_dir, "random1000_seqs.tsv"),
        seq_col="seq",
        name_col="name",
        target_col="activity_0",
        rev_comp=False,
        sep="\t",
        low_memory=False,
        return_numpy=False,
        return_dataframe=False,
        col_names=None,
        auto_name=False,
        compression="infer",
    )
    check_random1000_load(sdata, has_target=True)


def test_read_fasta():
    dataset_dir = f"{HERE}/../eugene/datasets/random1000"
    sdata = eu.dl.read_fasta(
        seq_file=os.path.join(dataset_dir, "random1000_seqs.fa"),
        target_file=os.path.join(dataset_dir, "random1000_activities.npy"),
        rev_comp=False,
        is_target_text=False,
        return_numpy=False,
    )
    check_random1000_load(sdata, has_target=True)


def test_read_numpy():
    dataset_dir = f"{HERE}/../eugene/datasets/random1000"
    sdata = eu.dl.read_numpy(
        seq_file=os.path.join(dataset_dir, "random1000_seqs.npy"),
        names_file=os.path.join(dataset_dir, "random1000_ids.npy"),
        target_file=os.path.join(dataset_dir, "random1000_activities.npy"),
        rev_seq_file=os.path.join(dataset_dir, "random1000_rev_seqs.npy"),
        is_names_text=False,
        is_target_text=False,
        delim="\n",
        ohe=False,
        return_numpy=False,
    )
    check_random1000_load(sdata, has_target=True)


@pytest.fixture
def sdata():
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.reverse_complement_seqs_sdata(sdata)
    return sdata


def test_SeqData_to_dataset(sdata):
    sdataset = sdata.to_dataset(target_keys="activity_0")
    assert isinstance(sdataset, SeqDataset)
    transforms = sdataset.transform
    assert transforms.transforms.pop()


def test_SeqDataset_get_item(sdata):
    sdataset = sdata.to_dataset(target_keys="activity_0")
    dataset_item = sdataset[0]
    assert np.all([isinstance(itm, torch.Tensor) for itm in dataset_item])
    assert dataset_item[1].shape == (4, 100)
    assert dataset_item[2].shape == (4, 100)


def test_SeqData_write_h5sd(sdata):
    sdata.write_h5sd(f"{HERE}/../eugene/datasets/random1000/random1000.h5sd")
