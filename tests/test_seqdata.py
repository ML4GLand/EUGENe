"""
Tests to make sure the example SeqData object works
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path

HERE = Path(__file__).parent

def test_init():
    names, seqs, rev_seqs, targets = eu.dl.read_numpy(f"{HERE}/_data/test_1000seqs_66/test_seqs.npy", names_file=f"{HERE}/_data/test_1000seqs_66/test_ids.npy", target_file=f"{HERE}/_data/test_1000seqs_66/test_labels.npy", rev_seq_file=f"{HERE}/_data/test_1000seqs_66/test_rev_seqs.npy", return_numpy=True)
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))
    sdata = eu.dl.SeqData(names=names, seqs=seqs, seqs_annot=targets, rev_seqs=rev_seqs)


@pytest.fixture
def sdata():
    """
    sdata
    """
    names, seqs, rev_seqs, targets = eu.dl.read_numpy(f"{HERE}/_data/test_1000seqs_66/test_seqs.npy", names_file=f"{HERE}/_data/test_1000seqs_66/test_ids.npy", target_file=f"{HERE}/_data/test_1000seqs_66/test_labels.npy", rev_seq_file=f"{HERE}/_data/test_1000seqs_66/test_rev_seqs.npy", return_numpy=True)
    sdata = eu.dl.SeqData(names=names, seqs=seqs, seqs_annot=targets, rev_seqs=rev_seqs)
    return sdata


def test_print(sdata):
    print(sdata)


def test_write_h5sd(sdata):
    sdata.write_h5sd(f"{HERE}/_data/test_1000seqs_66/test_seqs.h5sd")
