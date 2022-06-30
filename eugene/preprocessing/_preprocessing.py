import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..utils._decorators import track
from ..dataloading import SeqData
from ._encoding import encodeDNA
from ._utils import reverse_complement_seqs


@track
def reverse_complement_data(sdata: SeqData, copy=False) -> SeqData:
    """Reverse complement sequences.
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    Returns
    -------
    SeqData
        SeqData object with reverse complement sequences.
    """
    sdata = sdata.copy() if copy else sdata
    sdata.rev_seqs = reverse_complement_seqs(sdata.seqs)
    return sdata if copy else None


@track
def one_hot_encode_data(sdata: SeqData, copy=False) -> SeqData:
    """One-hot encode sequences.
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    Returns
    -------
    SeqData
        SeqData object with one-hot encoded sequences.
    """
    sdata = sdata.copy() if copy else sdata
    if sdata.seqs is not None:
        sdata.ohe_seqs = encodeDNA(sdata.seqs)
    if sdata.rev_seqs is not None:
        sdata.ohe_rev_seqs = encodeDNA(sdata.rev_seqs)
    return sdata if copy else None
