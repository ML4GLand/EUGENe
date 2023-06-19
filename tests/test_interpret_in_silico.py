"""
Tests to make sure the in silioc interpretation methods work
"""

import pytest
import numpy as np
import eugene as eu
from pathlib import Path
from eugene.interpret._filters import (
    _get_first_conv_layer,
    _get_activations_from_layer,
    _get_filter_activators,
    _get_pfms,
)

HERE = Path(__file__).parent


@pytest.fixture
def sdata():
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.reverse_complement_seqs_sdata(sdata)
    return sdata[:5]


@pytest.fixture
def seq(sdata):
    return sdata.seqs[0]


@pytest.fixture
def seqs(sdata):
    return sdata.seqs


@pytest.fixture
def ohe_seq(sdata):
    return sdata.ohe_seqs[0]


@pytest.fixture
def ohe_seqs(sdata):
    return sdata.ohe_seqs


@pytest.fixture
def model():
    model = eu.models.DeepBind(input_len=100, output_dim=1)
    return model


def test_in_silico_best_k_muts(model, seq, ohe_seq):
    mut_ohe_seq, delta, delta_ind = eu.interpret.best_k_muts(model, ohe_seq, k=1)
    mut_seq = eu.pp.decode_seq(mut_ohe_seq.squeeze(axis=0))
    assert eu.pp._utils._hamming_distance(seq, mut_seq) == 1


def test_in_silico_best_mut_seqs(model, seqs, ohe_seqs):
    mut_ohe_seqs, deltas, delta_inds = eu.interpret.best_mut_seqs(
        model, ohe_seqs, batch_size=32
    )
    for i in range(len(mut_ohe_seqs)):
        mut_seq = eu.pp.decode_seq(mut_ohe_seqs[i])
        assert eu.pp._utils._hamming_distance(seqs[i], mut_seq) == 1


def test_evolution(model, seq, ohe_seq):
    evolved_ohe_seq, deltas, delta_pos = eu.interpret.evolution(
        model, ohe_seq, force_different=True
    )
    evolved_seq = eu.pp.decode_seq(evolved_ohe_seq)
    assert eu.pp._utils._hamming_distance(seq, evolved_seq) == 10


def test_evolve_seqs_sdata(model, sdata):
    evolved_seqs = eu.interpret.evolve_seqs_sdata(
        model, sdata, rounds=5, force_different=True, return_seqs=True
    )
    assert "original_score" in sdata.seqs_annot
    assert "evolved_5_score" in sdata.seqs_annot
    # assert len(evolved_seqs) == 5


@pytest.fixture
def pfm():
    meme = eu.dl.motif.MinimalMEME(path=f"{HERE}/_data/CPEs.meme")
    motif = meme.motifs["TATA"]
    return motif.pfm


@pytest.fixture
def consensus():
    meme = eu.dl.motif.MinimalMEME(path=f"{HERE}/_data/CPEs.meme")
    motif = meme.motifs["TATA"]
    return motif.consensus


def test_feature_implant_seq(seq, consensus, ohe_seq, pfm):
    implanted_seq = eu.pp.feature_implant_seq(seq, consensus, 2, encoding="str")
    assert implanted_seq[2:18] == consensus
    implanted_seq = eu.pp.feature_implant_seq(ohe_seq, pfm, 2, encoding="onehot")
    assert np.all(implanted_seq.transpose()[2:18] == pfm)


def test_feature_implant_across_seq(seq, consensus, ohe_seq, pfm):
    implanted_seqs = eu.pp.feature_implant_across_seq(seq, consensus, encoding="str")
    assert len(implanted_seqs) == 85
    assert implanted_seqs[0][0:16] == consensus
    implanted_seqs = eu.pp.feature_implant_across_seq(ohe_seq, pfm, encoding="onehot")
    assert len(implanted_seqs) == 85
    assert np.all(implanted_seqs[0].transpose()[0:16] == pfm)
    implanted_seqs = eu.pp.feature_implant_across_seq(
        ohe_seq, pfm, encoding="onehot", onehot=True
    )
    assert len(implanted_seqs) == 85
    assert np.all(np.unique(implanted_seqs) == np.array([0.0, 1.0]))


def test_feature_implant_seq_sdata(model, sdata, consensus):
    scores = eu.interpret.feature_implant_seq_sdata(
        model,
        sdata,
        seq_id=sdata.names[0],
        feature=consensus,
        feature_name="name",
        encoding="str",
        onehot=False,
        device="cpu",
        store=False,
    )
    assert len(scores) == 85


def test_feature_implant_seqs_sdata(model, sdata, consensus):
    eu.interpret.feature_implant_seqs_sdata(
        model,
        sdata,
        feature=consensus,
        seqsm_key=f"name_slide",
        encoding="str",
        onehot=False,
        device="cpu",
    )
    assert sdata.seqsm["name_slide"].shape == (5, 85)
