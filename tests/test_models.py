"""
Tests to make sure basic model functionality works 
"""

import os
import torch
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import eugene as eu

SEQ_LEN = 100
OUT_DIMS = 2
MODEL = "hybrid"
STRAND = "ss"
TASK = "regression"
LOSS_FXN = "mse"
CNN_KWARGS = dict(channels=[4, 16, 32], conv_kernels=[15, 5], pool_kernels=[1, 1])
RNN_KWARGS = dict(output_dim=32, bidirectional=True, batch_first=True)
FCN_KWARGS = dict(hidden_dims=[50])
HERE = Path(__file__).parent


@pytest.fixture
def sdata():
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.reverse_complement_seqs_sdata(sdata)
    return sdata


def check_model(test_model, sdata, transpose=False):
    x = torch.randn(10, 4, SEQ_LEN)
    x_rev = torch.randn(10, 4, SEQ_LEN)
    eu.models.init_weights(test_model)
    if transpose:
        x = x.transpose(1, 2)
        x_rev = x_rev.transpose(1, 2)
        transform_kwargs = {"transpose": True}
    else:
        x = x
        x_rev = x_rev
        transform_kwargs = {"transpose": False}
    output = test_model(x, x_rev)
    assert output.shape == (10, 2)
    eu.evaluate.predictions(
        test_model,
        sdata,
        target_keys=["activity_0", "activity_1"],
        transform_kwargs=transform_kwargs,
        store_only=True,
    )
    assert "activity_0_predictions" in sdata.seqs_annot.columns


def test_FCN(sdata):
    model = eu.models.FCN(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr=None,
        loss_fxn=LOSS_FXN,
        fc_kwargs=FCN_KWARGS,
    )
    check_model(model, sdata)


def test_CNN(sdata):
    model = eu.models.CNN(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr=None,
        loss_fxn=LOSS_FXN,
        fc_kwargs=FCN_KWARGS,
        conv_kwargs=CNN_KWARGS,
    )
    check_model(model, sdata)


def test_RNN(sdata):
    model = eu.models.RNN(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr=None,
        loss_fxn=LOSS_FXN,
        rnn_kwargs=RNN_KWARGS,
    )
    check_model(model, sdata, transpose=True)


def test_Hybrid(sdata):
    model = eu.models.Hybrid(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr=None,
        loss_fxn=LOSS_FXN,
        fc_kwargs=FCN_KWARGS,
        conv_kwargs=CNN_KWARGS,
        rnn_kwargs=RNN_KWARGS,
    )
    check_model(model, sdata)


def test_DeepBind(sdata):
    model = eu.models.DeepBind(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr="max",
        loss_fxn=LOSS_FXN,
    )
    check_model(model, sdata)


def test_DeepSEA(sdata):
    model = eu.models.DeepBind(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr=None,
        loss_fxn=LOSS_FXN,
    )
    check_model(model, sdata)


def test_Jores21CNN(sdata):
    model = eu.models.Jores21CNN(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr=None,
        loss_fxn=LOSS_FXN,
    )
    check_model(model, sdata)


def test_Kopp21CNN(sdata):
    model = eu.models.Kopp21CNN(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr="max",
        loss_fxn=LOSS_FXN,
    )
    check_model(model, sdata, transpose=False)


def test_TutorialCNN(sdata):
    model = eu.models.TutorialCNN(
        input_len=SEQ_LEN,
        output_dim=OUT_DIMS,
        strand=STRAND,
        task=TASK,
        aggr="avg",
        loss_fxn=LOSS_FXN,
    )
    check_model(model, sdata, transpose=False)


def test_load_config(sdata):
    model_config = f"{HERE}/_configs/ssHybrid.yaml"
    model = eu.models.load_config("Hybrid", model_config)
    check_model(model, sdata)
