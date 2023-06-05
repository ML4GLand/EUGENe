"""
Tests to make sure the filter visualization interpretation methods work
"""

import torch
from torch.utils.data import DataLoader
import eugene as eu
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from eugene.interpret._filters import _get_first_conv_layer, _get_activations_from_layer, _get_filter_activators, _get_pfms


@pytest.fixture
def sdataloader():
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.reverse_complement_seqs_sdata(sdata)
    sdataset = sdata.to_dataset(target_keys="activity_0")
    sdataloader = DataLoader(sdataset, batch_size=32, num_workers=0)
    return sdataloader

@pytest.fixture
def model():
    """
    model
    """
    model = eu.models.DeepBind(input_len=100, output_dim=2)
    return model


@pytest.fixture
def sdata():
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    return sdata


def test_get_first_conv_layer(model):
    first_layer = _get_first_conv_layer(model)
    assert isinstance(first_layer, torch.nn.Conv1d)


def test_get_first_conv_layer(model):
    first_layer = _get_first_conv_layer(model)
    assert isinstance(first_layer, torch.nn.Conv1d)


def test_get_activations_from_layer(model, sdataloader):
    first_layer = _get_first_conv_layer(model)
    np_activators, np_sequences = _get_activations_from_layer(first_layer, sdataloader, vocab="DNA")
    assert np_activators.shape == (1000, 16, 85)
    assert len(np_sequences) == 1000
    isinstance(np_sequences[0], str)


def test_get_filter_activators(model, sdataloader):
    first_layer = _get_first_conv_layer(model)
    np_activators, np_sequences = _get_activations_from_layer(first_layer, sdataloader, vocab="DNA")
    activator_seqs = _get_filter_activators(np_activators, np_sequences, first_layer.kernel_size[0], method="Alipanahi15", threshold=0.75)
    assert len(activator_seqs) == 16
    assert isinstance(activator_seqs[0], list)
    assert isinstance(activator_seqs[0][0], str)
    activator_seqs = _get_filter_activators(np_activators, np_sequences, first_layer.kernel_size[0], method="Minnoye20", num_seqlets=10)
    assert len(activator_seqs) == 16
    assert isinstance(activator_seqs[0], list)
    assert isinstance(activator_seqs[0][0], str)
    assert len(activator_seqs[0]) == 10


def test_get_pfm(model, sdataloader):
    first_layer = _get_first_conv_layer(model)
    np_activators, np_sequences = _get_activations_from_layer(first_layer, sdataloader, vocab="DNA")
    activator_seqs = _get_filter_activators(np_activators, np_sequences, first_layer.kernel_size[0], method="Alipanahi15", threshold=0.75)
    pfms = _get_pfms(activator_seqs, first_layer.kernel_size[0], vocab="DNA")
    assert len(pfms) == 16
    assert isinstance(pfms[0], pd.DataFrame)
    assert pfms[0].shape == (16, 4)


def test_generate_pfms_sdata(model, sdata):
    eu.interpret.generate_pfms_sdata(model, sdata)
    assert "pfms" in sdata.uns
    assert len(sdata.uns["pfms"]) == 16
    assert isinstance(sdata.uns["pfms"][0], pd.DataFrame)
    assert sdata.uns["pfms"][0].shape == (16, 4)
    