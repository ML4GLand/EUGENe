"""
Tests to make sure the feature attribution interpretation methods work
"""

import torch
import eugene as eu
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

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


@pytest.fixture
def model():
    """
    model
    """
    model = eu.models.DeepBind(input_len=100, output_dim=2)
    return model


@pytest.fixture
def inputs():
    x = torch.randn(10, 4, 100)
    x_rev = torch.randn(10, 4, 100)
    return (x, x_rev)


def check_attribution_method(model, sdata, inputs, method, **kwargs):
    """
    Check that an attribution method works
    """
    explains = eu.interpret.nn_explain(
        model=model, inputs=inputs, target=0, saliency_type=method, **kwargs
    )
    assert explains.shape == (10, 4, 100)
    eu.interpret.feature_attribution_sdata(
        sdata=sdata, model=model, target=0, method=method, **kwargs
    )
    assert sdata.uns[f"{method}_imps"].shape == (1000, 4, 100)


def test_NaiveISM(model, sdata, inputs):
    check_attribution_method(model, sdata, inputs, "NaiveISM")


def test_InputXGradient(model, sdata, inputs):
    check_attribution_method(model, sdata, inputs, "InputXGradient")


def test_DeepLIFT(model, sdata, inputs):
    check_attribution_method(model, sdata, inputs, "DeepLift", reference="zero")
    check_attribution_method(model, sdata, inputs, "DeepLift", reference="shuffle")
    check_attribution_method(model, sdata, inputs, "DeepLift", reference="gc")


def test_GradientSHAP(model, sdata, inputs):
    check_attribution_method(model, sdata, inputs, "GradientSHAP", reference="zero")
    check_attribution_method(model, sdata, inputs, "GradientSHAP", reference="shuffle")
    check_attribution_method(model, sdata, inputs, "GradientSHAP", reference="gc")


def test_pca(model, sdata):
    """
    Test PCA
    """
    from sklearn.decomposition import PCA

    eu.interpret.feature_attribution_sdata(
        sdata=sdata, model=model, target=0, method="InputXGradient"
    )
    eu.interpret.pca(sdata, uns_key="InputXGradient_imps")
    assert sdata.seqsm["InputXGradient_imps_pca"].shape == (1000, 30)
    assert isinstance(sdata.uns["InputXGradient_imps_pca"], PCA)
    eu.pl.skree(sdata, uns_key="InputXGradient_imps_pca")


def test_aggregate_importances_sdata(model, sdata):
    eu.interpret.feature_attribution_sdata(
        sdata=sdata, model=model, target=0, method="InputXGradient"
    )
    eu.dataload.motif.jaspar_annots_sdata(sdata, motif_names=["GATA1"])
    eu.interpret.aggregate_importances_sdata(sdata, uns_key="InputXGradient_imps")
    assert "InputXGradient_imps_agg_scores" in sdata.pos_annot.df.columns
