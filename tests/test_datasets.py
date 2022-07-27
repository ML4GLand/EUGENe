"""
Tests to make sure the example datasets load and the basic I/O functions work.
"""

import eugene as eu
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def tmp_dataset_dir(tmpdir_factory):
    new_dir = Path(tmpdir_factory.mktemp("eugene_data"))
    old_dir = eu.settings.dataset_dir
    eu.settings. dataset_dir= new_dir  # Set up
    yield eu.settings.dataset_dir
    eu.settings. dataset_dir= old_dir  # Tear down


def test_random1000(tmp_dataset_dir):
    data = eu.datasets.random1000()
    assert(data)


def test_farley15(tmp_dataset_dir):
    data = eu.datasets.farley15()
    assert(data)


def test_deBoer20(tmp_dataset_dir):
    data = eu.datasets.deBoer20(0)
    assert(data)


def test_jores21(tmp_dataset_dir):
    data = eu.datasets.jores21()
    assert(data)


def test_deAlmeida22(tmp_dataset_dir):
    data = eu.datasets.deAlmeida22()
    assert(data)
