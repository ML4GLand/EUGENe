"""
Tests to make sure the dataload module isn't busted
"""

import os
import pytest
import numpy as np
import eugene as eu
from pathlib import Path
HERE = Path(__file__).parent


@pytest.fixture
def sdata():
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.reverse_complement_seqs_sdata(sdata)
    return sdata


def test_Motif():
    minimal_meme = eu.dl.motif.MinimalMEME(f"{HERE}/_data/CPEs.meme")
    motifs = minimal_meme.motifs
    example_motif = motifs["TATA"]
    assert isinstance(example_motif, eu.dl.motif.Motif)


def test_minimalMEME():
    minimal_meme = eu.dl.motif.MinimalMEME(f"{HERE}/_data/CPEs.meme")
    assert isinstance(minimal_meme, eu.dl.motif.MinimalMEME)


def test_pwm_to_meme():
    minimal_meme = eu.dl.motif.MinimalMEME(f"{HERE}/_data/CPEs.meme")
    motifs = minimal_meme.motifs
    example_motif = motifs["TATA"]
    pfm = np.expand_dims(example_motif.pfm, axis=0)
    eu.dl.motif.pwm_to_meme(pfm, output_file_path=f"{HERE}/_data/TATA.meme")
    assert os.path.exists(f"{HERE}/_data/TATA.meme")


def test_filters_to_meme_sdata(sdata):
    from pymemesuite.common import Motif
    model = eu.models.DeepBind(input_len=66, output_dim=1)
    eu.interpret.generate_pfms_sdata(model, sdata)
    eu.dl.motif.filters_to_meme_sdata(
        sdata,
        filter_ids=list(sdata.uns["pfms"].keys()),
        output_dir=f"{HERE}/_data/",
        file_name=f"model_filters.meme"
    )
    assert os.path.exists(f"{HERE}/_data/model_filters.meme")
    loaded_meme = eu.dl.motif.load_meme(f"{HERE}/_data/model_filters.meme")
    assert np.all([isinstance(item, Motif) for item in loaded_meme[0]])


def test_get_jaspar_motifs():
    from Bio.motifs.jaspar import Motif
    single_test = eu.dl.motif.get_jaspar_motifs(motif_accs=['MA0095.2'])
    assert isinstance(single_test[0], Motif)
    multi_test = eu.dl.motif.get_jaspar_motifs(motif_names=['CTCF', 'GATA1'])
    assert np.all([isinstance(test, Motif) for test in multi_test])


def test_save_motifs_as_meme():
    from pymemesuite.common import Motif
    motifs = eu.dl.motif.get_jaspar_motifs(motif_names=['CTCF', "GATA1"])
    eu.dl.motif.save_motifs_as_meme(motifs, f"{HERE}/_data/jaspar.meme")
    assert os.path.exists(f"{HERE}/_data/jaspar.meme")
    loaded_meme = eu.dl.motif.load_meme(f"{HERE}/_data/jaspar.meme")
    assert np.all([isinstance(item, Motif) for item in loaded_meme[0]])



def test_fimo_motifs(sdata):
    loaded_meme = eu.dl.motif.load_meme(f"{HERE}/_data/jaspar.meme")
    annots = eu.dl.motif.fimo_motifs(sdata, *loaded_meme)
    assert isinstance(annots, list)


def test_score_seqs(sdata):
    from pyranges import PyRanges
    score_df = eu.dl.motif.score_seqs(
        sdata=sdata, 
        motif_accs=['MA0095.2'], 
        filename=f"{HERE}/_data/jaspar.meme"
    )
    assert isinstance(score_df, PyRanges)
    assert np.unique(score_df.df["Name"])[0] == "YY1"


def test_jaspar_annots_sdata(sdata):
    from pyranges import PyRanges
    eu.dl.motif.jaspar_annots_sdata(
        sdata,
        motif_accs=['MA0048.1', 'MA0049.1']
    )
    assert isinstance(sdata.pos_annot, PyRanges)
