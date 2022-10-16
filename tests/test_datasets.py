"""
Tests to make sure the example datasets load
"""

import os
import numpy as np
import pandas as pd
import eugene as eu
from pathlib import Path
HERE = Path(__file__).parent
eu.settings.dataset_dir = f"{HERE}/_data/datasets"

def test_get_dataset_info():
    dataset_info = eu.datasets.get_dataset_info()
    assert dataset_info.index.name == "dataset_name"
    assert "description" in dataset_info
    

def test_random1000():
    sdata = eu.datasets.random1000()
    assert sdata.n_obs == 1000
    assert len(sdata.seqs) == 1000
    assert len(sdata.seqs[0]) == 100
    assert isinstance(sdata.seqs_annot, pd.DataFrame)
    assert sdata.seqs_annot.shape == (1000, 20) 


def test_ray13():
    sdata = eu.datasets.ray13()
    assert sdata.n_obs == 241357
    assert "RBD_v3" in sdata.names[0]
    assert "Probe_Set" in sdata.seqs_annot.columns
    assert sdata.seqs_annot.shape == (241357, 245)
    sdata_path = eu.datasets.ray13(return_sdata=False)[0]
    assert os.path.exists(sdata_path) 


def test_deBoer20():
    sdata = eu.datasets.deBoer20(0)
    assert sdata.n_obs == 9982
    assert sdata.names[-1] == "seq9981" 
    sdata_path = eu.datasets.deBoer20(0, return_sdata=False)[0]
    assert os.path.exists(sdata_path)


def test_jores21():
    sdata = eu.datasets.jores21(dataset="leaf", add_metadata=False)
    assert sdata.n_obs == 72158
    assert sdata.names[-1] == "seq72157"
    assert np.all(sdata.seqs_annot.columns == ['set', 'sp', 'gene', 'enrichment'])
    assert sdata.seqs_annot.shape == (72158, 4)
    sdata_path = eu.datasets.jores21(dataset="leaf", return_sdata=False)[0]
    assert os.path.exists(sdata_path)
    #TODO: Add in metadata reading
    sdata = eu.datasets.jores21(dataset="proto", add_metadata=False)
    assert sdata.n_obs == 75808
    assert sdata.names[-1] == "seq75807"
    assert np.all(sdata.seqs_annot.columns == ['set', 'sp', 'gene', 'enrichment'])
    assert sdata.seqs_annot.shape == (75808, 4)
    sdata_path = eu.datasets.jores21(dataset="proto", return_sdata=False)[0]
    assert os.path.exists(sdata_path)


def test_deAlmeida22():
    sdata = eu.datasets.deAlmeida22("train")
    assert sdata.n_obs == 402296
    assert sdata.names[0] == "chr2L_5587_5835_+_positive_peaks"
    assert sdata.seqs_annot.shape == (402296, 6)
    sdata_path = eu.datasets.deAlmeida22("train", return_sdata=False)[0]
    assert os.path.exists(sdata_path)
    sdata = eu.datasets.deAlmeida22("val")
    assert sdata.n_obs == 40570
    assert sdata.names[0] == "chr2R_4429_4677_+_positive_peaks"
    assert sdata.seqs_annot.shape == (40570, 6)
    sdata_path = eu.datasets.deAlmeida22("val", return_sdata=False)[0]
    assert os.path.exists(sdata_path)
    sdata = eu.datasets.deAlmeida22("test")
    assert sdata.n_obs == 41186
    assert sdata.names[0] == "chr2R_10574736_10574984_+_positive_peaks"
    assert sdata.seqs_annot.shape == (41186, 6)
    sdata_path = eu.datasets.deAlmeida22("test", return_sdata=False)[0]
    assert os.path.exists(sdata_path)
