"""
Tests to make sure preprocess functions works
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path

HERE = Path(__file__).parent


@pytest.fixture
def seq():
    seq = eu.utils.random_seq(seq_len=10)
    return seq


@pytest.fixture
def seqs():
    seqs = eu.utils.random_seqs(seq_num=10, seq_len=10)
    return seqs


@pytest.fixture
def jagged_seqs():
    jagged_seqs = [eu.utils.random_seq(seq_len=10), eu.utils.random_seq(seq_len=5)]
    return jagged_seqs


@pytest.fixture
def bad_seqs():
    bad_seqs = ["AggaAATC ", " GGTAa"]
    return bad_seqs


def test_sanitize_seqs(bad_seqs):
    assert eu.pp.sanitize_seq(bad_seqs[0]) == "AGGAAATC"
    assert np.all(eu.pp.sanitize_seqs(bad_seqs) == np.array(["AGGAAATC", "GGTAA"]))


def test_ascii_seqs(seq, seqs):
    encoded_seq = eu.pp.ascii_encode_seq(seq)
    assert eu.pp.ascii_decode_seq(encoded_seq) == seq
    encoded_seqs = eu.pp.ascii_encode_seqs(seqs)
    assert np.all(eu.pp.ascii_decode_seqs(encoded_seqs) == seqs)


def test_reverse_complement_seqs(seq, seqs, jagged_seqs):
    assert seq == eu.pp.reverse_complement_seq(eu.pp.reverse_complement_seq(seq))
    assert np.all(
        seqs == eu.pp.reverse_complement_seqs(eu.pp.reverse_complement_seqs(seqs))
    )
    assert np.all(
        jagged_seqs
        == eu.pp.reverse_complement_seqs(eu.pp.reverse_complement_seqs(jagged_seqs))
    )


def test_ohe_seqs(seq, seqs, jagged_seqs):
    ohe = eu.pp.ohe_seq(seq)
    decoded_seq = eu.pp.decode_seq(ohe)
    assert seq == decoded_seq
    ohes = eu.pp.ohe_seqs(seqs)
    decoded_seqs = eu.pp.decode_seqs(ohes)
    assert np.all(seqs == decoded_seqs)
    rc_ohes = eu.pp.reverse_complement_seqs(ohes)
    decode_rc_ohes = eu.pp.reverse_complement_seqs(eu.pp.decode_seqs(rc_ohes))
    assert np.all(seqs == decode_rc_ohes)
    jagged_ohe_seqs = eu.pp.ohe_seqs(jagged_seqs)
    jagged_decoded_seqs = eu.pp.decode_seqs(jagged_ohe_seqs)


def test_dinuc_shuffle_seqs(seq, seqs):
    dnt_shuf_seq = eu.pp.dinuc_shuffle_seq(seq, num_shufs=10)
    assert np.all(seq != dnt_shuf_seq)
    assert len(dnt_shuf_seq) == 10
    dnt_shuf_seqs = eu.pp.dinuc_shuffle_seqs(seqs, num_shufs=10)
    assert dnt_shuf_seqs.shape == (10, 10)


def test_perturb_seqs(seq, seqs):
    ohe = eu.pp.ohe_seq(seq)
    perturbed_seq = eu.pp.perturb_seq(ohe)
    assert perturbed_seq.shape == (30, 4, 10)
    decoded_perturb = eu.pp.decode_seqs(perturbed_seq)
    assert len(decoded_perturb) == 30
    ohes = eu.pp.ohe_seqs(seqs)
    perturbed_seqs = eu.pp.perturb_seqs(ohes)
    assert perturbed_seqs.shape == (10, 30, 4, 10)
    decoded_perturbed_seq = eu.pp.decode_seqs(perturbed_seqs[0])
    assert len(decoded_perturbed_seq) == 30
