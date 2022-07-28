"""
Tests to make sure the example dataloading routines work.
"""

import eugene as eu
from pathlib import Path
from torchvision import transforms

HERE = Path(__file__).parent

# TODO: Add test_read_and_concat_dataframes() funtions
# TODO: More comprehesive testing of reading
# TODO: Add write tests


def test_read_csv():
    names, seqs, rev_seqs, targets = eu.dl.read_csv(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.tsv",
        seq_col="SEQ",
        return_numpy=True,
        auto_name=False,
    )
    assert seqs is not None
    assert names is None
    assert rev_seqs is None
    assert targets is None

    names, seqs, rev_seqs, targets = eu.dl.read_csv(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.tsv",
        seq_col="SEQ",
        name_col="NAME",
        target_col="ACTIVITY",
        rev_comp=True,
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)


def test_read_fasta():
    names, seqs, rev_seqs, targets = eu.dl.read_fasta(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.fa", return_numpy=True
    )
    assert seqs is not None
    assert names is not None
    assert rev_seqs is None
    assert targets is None

    names, seqs, rev_seqs, targets = eu.dl.read_fasta(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.fa",
        target_file=f"{HERE}/../eugene/datasets/random1000/random1000_labels.npy",
        rev_comp=True,
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)


def test_read_numpy():
    names, seqs, rev_seqs, targets = eu.dl.read_numpy(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.npy", return_numpy=True
    )
    assert seqs is not None
    assert names is None
    assert rev_seqs is None
    assert targets is None

    names, seqs, rev_seqs, targets = eu.dl.read_numpy(
        f"{HERE}/../eugene/datasets/random1000/random1000_ohe_seqs.npy",
        names_file=f"{HERE}/../eugene/datasets/random1000/random1000_ids.npy",
        target_file=f"{HERE}/../eugene/datasets/random1000/random1000_labels.npy",
        rev_seq_file=f"{HERE}/../eugene/datasets/random1000/random1000_rev_ohe_seqs.npy",
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)


def test_read():
    names, seqs, rev_seqs, targets = eu.dl.read(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.tsv",
        seq_col="SEQ",
        name_col="NAME",
        target_col="ACTIVITY",
        rev_comp=True,
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)

    names, seqs, rev_seqs, targets = eu.dl.read(
        f"{HERE}/../eugene/datasets/random1000/random1000_ohe_seqs.npy",
        names_file=f"{HERE}/../eugene/datasets/random1000/random1000_ids.npy",
        target_file=f"{HERE}/../eugene/datasets/random1000/random1000_labels.npy",
        rev_seq_file=f"{HERE}/../eugene/datasets/random1000/random1000_rev_ohe_seqs.npy",
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)

    names, seqs, rev_seqs, targets = eu.dl.read(
        f"{HERE}/../eugene/datasets/random1000/random1000_seqs.fa",
        target_file=f"{HERE}/../eugene/datasets/random1000/random1000_labels.npy",
        rev_comp=True,
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)


def test_SeqDataset():
    names, seqs, rev_seqs, targets = eu.dl.read(
        f"{HERE}/../eugene/datasets/random1000/random1000_ohe_seqs.npy",
        names_file=f"{HERE}/../eugene/datasets/random1000/random1000_ids.npy",
        target_file=f"{HERE}/../eugene/datasets/random1000/random1000_labels.npy",
        rev_seq_file=f"{HERE}/../eugene/datasets/random1000/random1000_rev_ohe_seqs.npy",
        return_numpy=True,
    )
    assert len(names) == len(seqs) == len(rev_seqs) == len(targets)
    dataset = eu.dl.SeqDataset(
        names=names, seqs=seqs, targets=targets, rev_seqs=rev_seqs, transform=None
    )
    assert len(dataset[0]) == 4


def test_SeqDataModule():
    data_transform = transforms.Compose([eu.dl.ToTensor(transpose=True)])
    datamodule = eu.dl.SeqDataModule(
        seq_file=f"{HERE}/../eugene/datasets/random1000/random1000_ohe_seqs.npy",
        batch_size=32,
        transform=data_transform,
        read_kwargs={"return_numpy": True},
    )
    datamodule.setup()
    dataset = datamodule.train_dataloader().dataset
    assert len(dataset[0]) == 4
