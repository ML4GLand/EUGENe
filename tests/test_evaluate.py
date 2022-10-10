"""
Tests to make sure evaluate module functionality works as expected.
"""

import pytest
import eugene as eu
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
eu.settings.logging_dir = f"{HERE}/_output"


@pytest.fixture
def sdata():
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.reverse_complement_seqs_sdata(sdata)
    eu.pp.train_test_split_sdata(sdata)
    return sdata[:128]


@pytest.fixture
def model():
    """
    model
    """
    model = eu.models.DeepBind(input_len=66, output_dim=1)
    return model


def test_predictions(sdata, model):
    eu.evaluate.predictions(
        model,
        sdata,
        target_keys="activity_0",
        out_dir=f"{HERE}/_output",
        file_label="test",
    )
    saved_t = pd.read_csv(
        f"{HERE}/_output/ssDeepBind_regression/test_predictions.tsv", index_col=0, sep="\t"
    )
    assert np.allclose(
        saved_t["predictions_0"].values,
        sdata.seqs_annot.loc[saved_t.index]["activity_0_predictions"].values,
    )


def test_train_val_predictions(sdata, model):
    eu.evaluate.train_val_predictions(
        model, sdata=sdata, target_keys="activity_0", train_key="train_val", out_dir=f"{HERE}/_output/"
    )
    saved_t = pd.read_csv(f"{HERE}/_output/ssDeepBind_regression/train_predictions.tsv", index_col=0, sep="\t")
    assert np.allclose(
        saved_t["predictions_0"].values,
        sdata.seqs_annot.loc[saved_t.index]["activity_0_predictions"].values,
    )
    saved_v = pd.read_csv(f"{HERE}/_output/ssDeepBind_regression/val_predictions.tsv", index_col=0, sep="\t")
    assert np.allclose(
        saved_v["predictions_0"].values,
        sdata.seqs_annot.loc[saved_v.index]["activity_0_predictions"].values,
    )
