"""
Tests to make sure preprocess functions works
"""

import eugene as eu
import pytest
from pathlib import Path
from eugene.dataload import SeqData
HERE = Path(__file__).parent


@pytest.fixture
def sdata():
    sdata = eu.datasets.random1000()
    return sdata


def test_sanitize_seqs_sdata(sdata):
    sdata_copy = eu.pp.sanitize_seqs_sdata(sdata, copy=True)
    assert isinstance(sdata_copy, SeqData)


def test_ohe_seqs_sdata(sdata):
    eu.pp.ohe_seqs_sdata(sdata)
    assert sdata.ohe_seqs is not None
    assert len(sdata.ohe_seqs) == len(sdata.seqs)


def test_reverse_complement_seqs_sdata(sdata):
    eu.pp.reverse_complement_seqs_sdata(sdata, rc_seqs=False, copy=False)


def test_clean_nan_targets_sdata(sdata):
    eu.pp.clean_nan_targets_sdata(sdata, target_keys="activity_0", copy=True)


def test_clamp_targets_sdata(sdata):
    eu.pp.train_test_split_sdata(sdata)
    eu.pp.clamp_targets_sdata(sdata, "activity_0", 0.8, "train_val", store_clamp_nums=True)
    assert sdata.uns["clamp_nums"] is not None


def test_scale_targets_sdata(sdata):
    eu.pp.train_test_split_sdata(sdata)
    eu.pp.scale_targets_sdata(sdata, "activity_0", "train_val", store_scaler=True)
    assert sdata
    assert sdata.uns["scaler"] is not None


def test_binarize_targets_sdata(sdata):
    eu.pp.binarize_targets_sdata(sdata, target_keys="activity_0", upper_threshold=0, suffix=True, copy=False)


def test_train_test_split_sdata(sdata):
    eu.pp.train_test_split_sdata(sdata)
    assert sdata["train_val"] is not None


def test_prepare_seqs_sdata(sdata):
    eu.pp.prepare_seqs_sdata(sdata)
    assert sdata.seqs is not None
    assert sdata.ohe_seqs is not None
    assert sdata.ohe_rev_seqs is not None
    assert len(sdata.seqs) == len(sdata.ohe_seqs) == len(sdata.ohe_rev_seqs)
    assert sdata["train_val"] is not None