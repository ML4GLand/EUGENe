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
    eu.pp.prepare_data(sdata)
    return sdata


@pytest.fixture
def model():
    """
    model
    """
    model = eu.models.DeepBind(input_len=66, output_dim=1)
    return model


def test_fit(sdata, model):
    eu.train.fit(
        model,
        sdata=sdata,
        target="target",
        train_key="train",
        epochs=5,
        log_dir=f"{HERE}/_logs",
    )
