"""
Tests to make sure the example datasets load and the basic I/O functions work.
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path

HERE = Path(__file__).parent

@pytest.fixture(scope="module")
def tmp_dataset_dir(tmpdir_factory):
    new_dir = Path(tmpdir_factory.mktemp("eugene_data"))
    old_dir = eu.settings.datasetdir
    eu.settings.datasetdir = new_dir  # Set up
    yield eu.settings.datasetdir
    eu.settings.datasetdir = old_dir  # Tear down


def test_random1000(tmp_dataset_dir):
    data = eu.datasets.random1000()
    assert len(data) == 4
    assert len(data[0]) == 1000
    assert len(data[1][0]) == 66


def test_load_csv():
    # Maximum functionality
    names, seqs, rev_seqs, targets = eu.datasets.load_csv(f"{HERE}/_data/test_1000seqs_66/test_seqs.tsv", seq_col="SEQ")
    assert(seqs is not None)
    assert(names == None)
    assert(rev_seqs == None)
    assert(targets == None)

    # Maximum functionality
    names, seqs, rev_seqs, targets = eu.datasets.load_csv(f"{HERE}/_data/test_1000seqs_66/test_seqs.tsv", seq_col="SEQ", name_col="NAME", target_col="ACTIVITY", rev_comp=True)
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))


def test_load_fasta():
    names, seqs, rev_seqs, targets = eu.datasets.load_fasta(f"{HERE}/_data/test_1000seqs_66/test_seqs.fa")
    assert(seqs is not None)
    assert(names is not None)
    assert(rev_seqs == None)
    assert(targets == None)

    names, seqs, rev_seqs, targets = eu.datasets.load_fasta(f"{HERE}/_data/test_1000seqs_66/test_seqs.fa", target_file=f"{HERE}/_data/test_1000seqs_66/test_labels.npy", rev_comp=True)
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))


def test_load_numpy():
    names, seqs, rev_seqs, targets = eu.datasets.load_numpy(f"{HERE}/_data/test_1000seqs_66/test_seqs.npy")
    assert(seqs is not None)
    assert(names == None)
    assert(rev_seqs == None)
    assert(targets == None)

    names, seqs, rev_seqs, targets = eu.datasets.load_numpy(f"{HERE}/_data/test_1000seqs_66/test_ohe_seqs.npy", names_file=f"{HERE}/_data/test_1000seqs_66/test_ids.npy", target_file=f"{HERE}/_data/test_1000seqs_66/test_labels.npy", rev_seq_file=f"{HERE}/_data/test_1000seqs_66/test_rev_ohe_seqs.npy")
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))


def test_load():
    names, seqs, rev_seqs, targets = eu.datasets.load(f"{HERE}/_data/test_1000seqs_66/test_seqs.tsv", seq_col="SEQ", name_col="NAME", target_col="ACTIVITY", rev_comp=True)
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))

    names, seqs, rev_seqs, targets = eu.datasets.load(f"{HERE}/_data/test_1000seqs_66/test_ohe_seqs.npy", names_file=f"{HERE}/_data/test_1000seqs_66/test_ids.npy", target_file=f"{HERE}/_data/test_1000seqs_66/test_labels.npy", rev_seq_file=f"{HERE}/_data/test_1000seqs_66/test_rev_ohe_seqs.npy")
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))

    names, seqs, rev_seqs, targets = eu.datasets.load(f"{HERE}/_data/test_1000seqs_66/test_seqs.fa", target_file=f"{HERE}/_data/test_1000seqs_66/test_labels.npy", rev_comp=True)
    assert(len(names) == len(seqs) == len(rev_seqs) == len(targets))
