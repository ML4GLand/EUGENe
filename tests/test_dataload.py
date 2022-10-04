"""
Tests to make sure the example dataload module isn't busted
"""

import eugene as eu
from pathlib import Path
from torchvision import transforms
import os

HERE = Path(__file__).parent

eu.logging.dataset_dir = "../../../eugene/datasets/random1000"


def check_random1000_load(sdata, has_target=False):
    assert isinstance(sdata, SeqData)
    assert sdata.n_obs == 1000
    assert sdata.names[-1] == "seq999"
    if has_target:
        assert sdata.seqs_annot.iloc[:, -1][0] is not np.nan
