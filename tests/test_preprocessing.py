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
    names, seqs, rev_seqs, targets = eu.dl.read_numpy(f"{HERE}/_data/test_1000seqs_66/test_seqs.npy", return_numpy=True)
    sdata = eu.dl.SeqData(seqs=seqs)
    return sdata


def test_reverse_complement_data(sdata):
    eu.pp.reverse_complement_data(sdata)
    assert(sdata.rev_seqs is not None)


def test_one_hot_encode_data(sdata):
    eu.pp.one_hot_encode_data(sdata)
    assert(sdata.ohe_seqs is not None)
