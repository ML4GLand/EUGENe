"""
Tests to make sure the example SeqData object works
"""

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
    return sdata


@pytest.fixture
def model():
    """
    model
    """
    model = eu.models.DeepBind(input_len=66, output_dim=1)
    return model


def test_predict(sdata, model):
    eu.predict.predictions(model, sdata=sdata, out_dir=f"{HERE}/_out/test_", save_preds="random1000_PREDS")
    saved_t = pd.read_csv(f"{HERE}/_out/test_predictions.tsv", index_col=0, sep="\t")
    assert(np.allclose(saved_t["PREDICTIONS"].values, sdata.seqs_annot.loc[saved_t.index]["random1000_PREDS"].values))
