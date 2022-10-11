"""
Tests to make sure the dataload module isn't busted
"""

import os
import numpy as np
import eugene as eu
from pathlib import Path
from eugene.dataload import SeqData


eu.settings.dataset_dir = f"{HERE}/../eugene/datasets/janggu_resources"
ref_file = "sample_genome.fa"
roi_file = "sample.bed"
bed_file = "scored_sample.bed"
bam_file = "sample2.bam"
bw_file = "sample.bw"  


def check_janggu_load(sdata, has_target=False):
    assert isinstance(sdata, SeqData)
    assert sdata.n_obs == 100
    assert "chr2" in sdata.names[-1]
    if has_target:
        assert sdata.seqs_annot.iloc[:, -1][0] is not np.nan

"""
def test_read_bed():
    sdata = eu.dl.read_bed(
        bed_file=os.path.join(eu.settings.dataset_dir, bed_file),
        roi_file=os.path.join(eu.settings.dataset_dir, roi_file),
        ref_file=os.path.join(eu.settings.dataset_dir, ref_file),
        binsize=200, 
        collapser="max",
        dnaflank=50,
        add_seqs=True,
        return_janggu=False
    )
    check_janggu_load(sdata, has_target=True)


def test_read_bed_janggu():
    dna, cov = eu.dl.read_bed(
        bed_file=os.path.join(eu.settings.dataset_dir, bed_file),
        roi_file=os.path.join(eu.settings.dataset_dir, roi_file),
        ref_file=os.path.join(eu.settings.dataset_dir, ref_file),
        binsize=200, 
        collapser="max",
        dnaflank=50,
        return_janggu=True
    )
    assert isinstance(dna, Bioseq)
    assert isinstance(cov, Cover)
"""

def test_read_bam():
    sdata = eu.dl.read_bam(
        bam_file=os.path.join(eu.settings.dataset_dir, bam_file),
        roi_file=os.path.join(eu.settings.dataset_dir, roi_file),
        ref_file=os.path.join(eu.settings.dataset_dir, ref_file), 
        binsize=200, 
        resolution=25
    )
    check_janggu_load(sdata, has_target=True)


def test_read_bigwig():
    sdata = eu.dl.read_bigwig(
        bigwig_file=os.path.join(eu.settings.dataset_dir, bw_file),
        roi_file=os.path.join(eu.settings.dataset_dir, roi_file),
        ref_file=os.path.join(eu.settings.dataset_dir, ref_file), 
        dnaflank=50,
        binsize=200,
        resolution=None,
        collapser="max"
    )
    check_janggu_load(sdata, has_target=True)