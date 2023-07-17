"""
Tests to make sure the example SeqData object works
"""

import os
import eugene as eu
import pytest
from pathlib import Path

HERE = Path(__file__).parent


@pytest.fixture
def sdata():
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.reverse_complement_seqs_sdata(sdata)
    eu.pp.train_test_split_sdata(sdata)
    return sdata


def test_fit_single_task(sdata):
    model = eu.models.DeepBind(input_len=66, output_dim=1)
    eu.settings.logging_dir = f"{HERE}/_logs"
    eu.train.fit(
        model,
        sdata,
        target_keys="activity_0",
        epochs=1,
        name="test_fit",
        version="singletask",
    )
    assert os.path.exists(f"{eu.settings.logging_dir}/test_fit/singletask/checkpoints/")


def test_fit_multitask(sdata):
    model = eu.models.DeepBind(input_len=66, output_dim=10)
    eu.settings.logging_dir = f"{HERE}/_logs"
    eu.train.fit(
        model,
        sdata,
        target_keys=[f"activity_{i}" for i in range(10)],
        epochs=1,
        name="test_fit",
        version="multitask",
    )
    assert os.path.exists(f"{eu.settings.logging_dir}/test_fit/multitask/checkpoints/")
