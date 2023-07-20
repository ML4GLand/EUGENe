"""
Tests to make sure preprocess functions work
"""

import eugene as eu
import pytest
from pathlib import Path

HERE = Path(__file__).parent


@pytest.fixture
def sdata():
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    return sdata


@pytest.fixture
def ohe_seqs(sdata):
    return sdata.ohe_seqs


@pytest.fixture
def targets(sdata):
    return sdata.seqs_annot["activity_0"].values


def test_split_train_test(ohe_seqs, targets):
    train_seqs, test_seqs, train_targets, test_targets = eu.pp.split_train_test(
        ohe_seqs, targets
    )
    assert len(train_seqs) == len(train_targets)
    assert len(test_seqs) == len(test_targets)
    assert len(train_seqs) + len(test_seqs) == len(ohe_seqs)
    assert len(train_targets) + len(test_targets) == len(targets)


def test_standardize_features(ohe_seqs, targets):
    train_seqs, test_seqs, train_targets, test_targets = eu.pp.split_train_test(
        ohe_seqs, targets
    )
    standardized_train, standardized_test = eu.pp.standardize_features(
        train_seqs, test_seqs
    )
    assert standardized_train.shape == train_seqs.shape
    assert standardized_test.shape == test_seqs.shape


def test_binarize_values(targets):
    binarized_targets = eu.pp.binarize_values(targets)
    assert binarized_targets.shape == targets.shape
