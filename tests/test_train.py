"""
Tests to make sure the example SeqData object works
"""

import eugene as eu
import numpy as np
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
    model = eu.models.DeepBind(input_len=66)
    return model


def test_fit(sdata, model):

    eu.pp.train_test_split_data(sdata)
    eu.train.fit(model, sdata=sdata, epochs=5, log_dir=f"{HERE}/_logs")
