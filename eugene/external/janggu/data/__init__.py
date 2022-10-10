"""Janggu datasets for deep learning in genomics."""
import os as _os
import sys as _sys
try:
    _bin_dir = _os.path.dirname(_sys.executable)
    _os.environ["PATH"] += _os.pathsep + _bin_dir
    from pybedtools import paths as _paths
    _paths._set_bedtools_path(_bin_dir)
except ImportError:
    raise ImportError(
        "Please install janggu dependencies `pip install eugene[janggu]`"
    )

print("Janggu datasets for deep learning in genomics.")

from copy import copy

from .coverage import Cover  # noqa
from .data import Dataset  # noqa
from .data import JangguSequence  # noqa
from .dna import Bioseq  # noqa
from .dna import VariantStreamer  # noqa
from .genomic_indexer import GenomicIndexer  # noqa
from .genomic_indexer import check_gindexer_compatibility  # noqa
from .genomicarray import GenomicArray  # noqa
from .genomicarray import LogTransform  # noqa
from .genomicarray import PercentileTrimming  # noqa
from .genomicarray import RegionLengthNormalization  # noqa
from .genomicarray import ZScore  # noqa
from .genomicarray import ZScoreLog  # noqa
from .genomicarray import create_genomic_array  # noqa
from .genomicarray import normalize_garray_tpm  # noqa
from .nparr import Array  # noqa
from .nparr import NanToNumConverter  # noqa
from .nparr import RandomOrientation  # noqa
from .nparr import RandomShift  # noqa
from .nparr import RandomSignalScale  # noqa
from .nparr import ReduceDim  # noqa
from .nparr import SqueezeDim  # noqa
from .nparr import Transpose  # noqa
from .visualization import HeatTrack  # noqa
from .visualization import LineTrack  # noqa
from .visualization import SeqTrack  # noqa
from .visualization import Track  # noqa
from .visualization import plotGenomeTrack  # noqa


def view(dataset, use_regions):
    """Creates a new view on the dataset.

    It may be used to utilize the same dataset
    for reading out a training, validation and test
    set without creating an additional memory overhead.
    When using this method, consider using the `store_whole_genome=True`
    option with the datasets.

    Parameters
    ----------
    dataset : Cover or Bioseq object
        Original Dataset containing a union of training and test set.
    use_regions: str
        BED file name defining the regions to use for the new view.
    """
    if not hasattr(dataset, 'gindexer'):
        raise ValueError("Unknown dataset type: {}".format(type(dataset)))

    gind = GenomicIndexer.create_from_file(use_regions,
                                           dataset.gindexer.binsize,
                                           dataset.gindexer.stepsize,
                                           dataset.gindexer.flank,
                                           zero_padding=dataset.gindexer.zero_padding,
                                           collapse=dataset.gindexer.collapse,
                                           random_state=dataset.gindexer.random_state)

    check_gindexer_compatibility(gind, dataset.garray.resolution,
                                 dataset.garray._full_genome_stored)
    subdata = copy(dataset)
    subdata.gindexer = gind

    return subdata


def subset(dataset, include_regions=None, exclude_regions=None):
    """Create a new subset of the dataset.

    A Cover or Bioseq dataset will be filtered
    with respect to the selected chromosomes.

    Parameters
    ----------
    dataset : Cover or Bioseq object
        Original Dataset containing a union of training and test set.
    include_regions: None, str or list(str)
        List of chromosome names for regions to keep.
    exclude_regions: None, str or list(str)
        List of chromosome names for regions to remove.
    """
    if include_regions is None and exclude_regions is None:
        raise ValueError("No filter specified.")
    if not hasattr(dataset, 'gindexer'):
        raise ValueError("Unknown dataset type: {}".format(type(dataset)))

    gind = dataset.gindexer
    subdata = copy(dataset)
    subdata.gindexer = gind.filter_by_region(include_regions, exclude_regions)

    return subdata


def split_train_test_(dataset, holdout_chroms):
    """Splits dataset into training and test set.

    A Cover or Bioseq dataset will be split into
    training and test set according to a list of
    heldout_chroms. That is the training datset
    exludes the heldout_chroms and the test set
    only includes the heldout_chroms.

    Parameters
    ----------
    dataset : Cover or Bioseq object
        Original Dataset containing a union of training and test set.
    holdout_chroms: list(str)
        List of chromosome names which will be used as validation chromosomes.
    """
    if not hasattr(dataset, 'gindexer'):
        raise ValueError("Unknown dataset type: {}".format(type(dataset)))

    traindata = subset(dataset, exclude_regions=holdout_chroms)
    testdata = subset(dataset, include_regions=holdout_chroms)

    return traindata, testdata


def split_train_test(datasets, holdout_chroms):
    """Splits dataset into training and test set.

    A Cover or Bioseq dataset will be split into
    training and test set according to a list of
    heldout_chroms. That is the training datset
    exludes the heldout_chroms and the test set
    only includes the heldout_chroms.

    Parameters
    ----------
    dataset : Cover or Bioseq object, list of Datasets or tuple(inputs, outputs)
        Original Dataset containing a union of training and test set.
    holdout_chroms: list(str)
        List of chromosome names which will be used as validation chromosomes.
    """

    if isinstance(datasets, tuple) and not hasattr(datasets, 'gindexer'):
        inputs = split_train_test(datasets[0], holdout_chroms)
        outputs = split_train_test(datasets[1], holdout_chroms)
        train = (inputs[0], outputs[0])
        test = (inputs[1], outputs[1])
    elif isinstance(datasets, list):
        train = []
        test = []
        for data in datasets:
            traindata, testdata = split_train_test_(data, holdout_chroms)
            test.append(testdata)
            train.append(traindata)
    elif isinstance(datasets, dict):
        train = []
        test = []
        for key in datasets:
            traindata, testdata = split_train_test_(datasets[key], holdout_chroms)
            test.append(testdata)
            train.append(traindata)
    else:
        train, test = split_train_test_(datasets, holdout_chroms)

    return train, test
