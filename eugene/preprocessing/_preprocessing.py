import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..utils._decorators import track
from ..dataloading import SeqData
from ._encoding import encodeDNA

@track
def one_hot_encode(sdata: SeqData, copy=False, **kwargs) -> SeqData:
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
    sdata.ohe_seqs = encodeDNA(sdata.seqs)
    return sdata if copy else None
