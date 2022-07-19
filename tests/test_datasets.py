"""
Tests to make sure the example datasets load and the basic I/O functions work.
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path

HERE = Path(__file__).parent
eu.settings.datasetdir = "/_data/datasets/"

@pytest.fixture(scope="module")
def tmp_dataset_dir(tmpdir_factory):
    new_dir = Path(tmpdir_factory.mktemp("eugene_data"))
    old_dir = eu.settings.datasetdir
    eu.settings.datasetdir = new_dir  # Set up
    yield eu.settings.datasetdir
    eu.settings.datasetdir = old_dir  # Tear down


def test_random1000(tmp_dataset_dir):
    data = eu.datasets.random1000()
    assert(data)


def test_farley2015(tmp_dataset_dir):
    data = eu.datasets.farley15()
    assert(data)
