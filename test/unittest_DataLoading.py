from matplotlib import transforms
from eugene import load_data
from eugene import seq_transforms
from torchvision import transforms
from eugene.MPRADataset import MPRADataset
from eugene.MPRADataModule import MPRADataModule
import unittest   # The test framework

class Test_load_data(unittest.TestCase):
    def test_load_csv(self):
        # Minimum functonality
        names, seqs, rev_seqs, targets = load_data.load_csv("test_seqs.tsv", seq_col="SEQ")
        self.assertTrue(seqs is not None)
        self.assertTrue(names == None)
        self.assertTrue(rev_seqs == None)
        self.assertTrue(targets == None)

        # Maximum functionality
        names, seqs, rev_seqs, targets = load_data.load_csv("test_seqs.tsv", seq_col="SEQ", name_col="NAME", target_col="ACTIVITY", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))

    def test_load_fasta(self):
        names, seqs, rev_seqs, targets = load_data.load_fasta("test_seqs.fa")
        self.assertTrue(seqs is not None)
        self.assertTrue(names is not None)
        self.assertTrue(rev_seqs == None)
        self.assertTrue(targets == None)

        names, seqs, rev_seqs, targets = load_data.load_fasta("test_seqs.fa", target_file="test_labels.npy", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))

    def test_load_numpy(self):
        names, seqs, rev_seqs, targets = load_data.load_numpy("test_seqs.npy")
        self.assertTrue(seqs is not None)
        self.assertTrue(names == None)
        self.assertTrue(rev_seqs == None)
        self.assertTrue(targets == None)

        names, seqs, rev_seqs, targets = load_data.load_numpy("test_ohe_seqs.npy", names_file="test_ids.npy", target_file="test_labels.npy", rev_seq_file="test_rev_ohe_seqs.npy")
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))

    def test_load(self):
        names, seqs, rev_seqs, targets = load_data.load("test_seqs.tsv", seq_col="SEQ", name_col="NAME", target_col="ACTIVITY", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))

        names, seqs, rev_seqs, targets = load_data.load("test_ohe_seqs.npy", names_file="test_ids.npy", target_file="test_labels.npy", rev_seq_file="test_rev_ohe_seqs.npy")
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))

        names, seqs, rev_seqs, targets = load_data.load("test_seqs.fa", target_file="test_labels.npy", rev_comp=True)
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))


class Test_MPRADataset(unittest.TestCase):
    def test_init(self):
        names, seqs, rev_seqs, targets = load_data.load("test_ohe_seqs.npy", names_file="test_ids.npy", target_file="test_labels.npy", rev_seq_file="test_rev_ohe_seqs.npy")
        self.assertTrue(len(names) == len(seqs) == len(rev_seqs) == len(targets))
        dataset = MPRADataset(names=names, seqs=seqs, targets=targets, rev_seqs=rev_seqs, transform=None)
        self.assertTrue(len(dataset[0]) == 4)


class Test_MPRADataModule(unittest.TestCase):
    def test_init(self):
        data_transform = transforms.Compose([seq_transforms.ToTensor(transpose=True)])
        datamodule = MPRADataModule(seq_file="test_ohe_seqs.npy", batch_size=32, transform=data_transform)
        datamodule.setup()
        dataset = datamodule.train_dataloader().dataset
        self.assertTrue(len(dataset[0]) == 4)

if __name__ == '__main__':
    unittest.main()
