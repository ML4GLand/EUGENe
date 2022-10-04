"""
Tests to make sure the example SeqData object works
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path

HERE = Path(__file__).parent


@pytest.fixture
def sdata():
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    return sdata


def test_reverse_complement_data(sdata):
    eu.pp.reverse_complement_data(sdata)
    assert sdata.rev_seqs is not None
    assert len(sdata.seqs) == len(sdata.rev_seqs)

def test_one_hot_encode_data(sdata):
    eu.pp.one_hot_encode_data(sdata)
    assert sdata.ohe_seqs is not None
    assert len(sdata.ohe_seqs) == len(sdata.seqs)

def test_train_test_split(sdata):
    eu.pp.train_test_split_data(sdata)
    assert sdata["train"] is not None

def test_clamp_percentiles(sdata):
    eu.pp.train_test_split_data(sdata)
    eu.pp.clamp_percentiles(sdata, "target", 0.8, "train", store_clamp_nums=True)
    assert sdata
    assert sdata.uns["clamp_nums"] is not None

def test_scale_target(sdata):
    eu.pp.train_test_split_data(sdata)
    eu.pp.scale_targets(sdata, "target", "train", store_scaler=True)
    assert sdata
    assert sdata.uns["scaler"] is not None

def test_prepare_data(sdata):
    eu.pp.prepare_data(sdata)
    assert sdata.seqs is not None
    assert sdata.rev_seqs is not None
    assert sdata.ohe_seqs is not None
    assert sdata.ohe_rev_seqs is not None
    assert len(sdata.seqs) == len(sdata.rev_seqs) == len(sdata.ohe_seqs) == len(sdata.ohe_rev_seqs)
    assert sdata["train"] is not None
