#import numpy as np
#import pandas as pd
import sys
sys.path.append("/Users/adamklie/Desktop/research/lab/dev/EUGENE/eugene")
import load_data    # The code to test
from MPRADataset import MPRADataset
import unittest   # The test framework
import logging

class Test_load_data(unittest.TestCase):
    def test_load_csv(self):
        names, seqs, rev_seqs, targets = load_data.load_csv("test_seqs.tsv", target_col="ACTIVITY")
        self.assertTrue(len(names) == len(seqs) == len(targets))
        self.assertTrue(rev_seqs == None)
        names, seqs, rev_seqs, targets = load_data.load_csv("test_seqs.tsv", target_col="ACTIVITY", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))

    def test_load_fasta(self):
        names, seqs, rev_seqs, targets = load_data.load_fasta("test_seqs.fa", "test_labels.npy")
        self.assertTrue(len(names) == len(seqs) == len(targets))
        self.assertTrue(rev_seqs == None)
        names, seqs, rev_seqs, targets = load_data.load_fasta("test_seqs.fa", "test_labels.npy", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets)) 

    def test_load_numpy(self):
        names, seqs, rev_seqs, targets = load_data.load_numpy("test_ids.npy", "test_seqs.npy", "test_labels.npy")
        self.assertTrue(len(names) == len(seqs) == len(targets))
        self.assertTrue(rev_seqs == None)
        names, seqs, rev_seqs, targets = load_data.load_numpy("test_ids.npy", "test_ohe_seqs.npy", "test_labels.npy", rev_seq_file="test_rev_ohe_seqs.npy")
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))
        
    def test_load(self):
        names, seqs, rev_seqs, targets = load_data.load("test_seqs.tsv", target_col="ACTIVITY", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))
        names, seqs, rev_seqs, targets = load_data.load("test_ids.npy", "test_ohe_seqs.npy", "test_labels.npy", rev_seq_file="test_rev_ohe_seqs.npy")
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))
        names, seqs, rev_seqs, targets = load_data.load("test_seqs.fa", "test_labels.npy", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))


class Test_MPRADataset(unittest.TestCase):
    def test_init(self):
        names, seqs, rev_seqs, targets = load_data.load("test_ids.npy", "test_ohe_seqs.npy", "test_labels.npy", rev_seq_file="test_rev_ohe_seqs.npy")
        dataset = MPRADataset(names=names, seqs=seqs, targets=targets, rev_seqs=rev_seqs, transform=None)
        names, seqs, rev_seqs, targets = dataset.names, dataset.seqs, dataset.rev_seqs, dataset.targets
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets)) 


if __name__ == '__main__':
    unittest.main()