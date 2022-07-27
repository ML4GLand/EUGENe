"""
Tests to make sure predict module functionality works as expected.
"""

import eugene as eu
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

HERE = Path(__file__).parent


eu.settings.batch_size = 128


@pytest.fixture
def sdata():
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    eu.pp.prepare_data(sdata)
    return sdata


@pytest.fixture
def model():
    """
    model
    """
    model = eu.models.DeepBind(input_len=66, output_dim=1)
    return model


def test_predictions(sdata, model):
    eu.predict.predictions(
        model,
        sdata=sdata,
        target="TARGETS",
        out_dir=f"{HERE}/_out/",
        file_label="random1000",
    )
    saved_t = pd.read_csv(
        f"{HERE}/_out/random1000_predictions.tsv", index_col=0, sep="\t"
    )
    assert np.allclose(
        saved_t["PREDICTIONS_0"].values,
        sdata.seqs_annot.loc[saved_t.index]["TARGETS_predictions"].values,
    )


def test_train_val_predictions(sdata, model):
    eu.predict.train_val_predictions(
        model, sdata=sdata, target="TARGETS", train_key="TRAIN", out_dir=f"{HERE}/_out/"
    )
    saved_t = pd.read_csv(f"{HERE}/_out/train_predictions.tsv", index_col=0, sep="\t")
    assert np.allclose(
        saved_t["PREDICTIONS_0"].values,
        sdata.seqs_annot.loc[saved_t.index]["TARGETS_PREDICTIONS"].values,
    )
    saved_v = pd.read_csv(f"{HERE}/_out/val_predictions.tsv", index_col=0, sep="\t")
    assert np.allclose(
        saved_v["PREDICTIONS_0"].values,
        sdata.seqs_annot.loc[saved_v.index]["TARGETS_PREDICTIONS"].values,
    )
