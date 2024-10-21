import torch
from tqdm.auto import tqdm
from seqexplainer import evolution
from .._settings import settings
import xarray as xr
import numpy as np

from typing import Optional, List, Dict, Any


def evolve_seqs_sdata(
    model: torch.nn.Module,
    sdata: xr.Dataset,
    rounds: int,
    seq_var: str = "ohe_seq",
    axis_order=("_sequence", "_ohe", "length"),
    add_seqs=True,
    return_seqs: bool = False,
    device: Optional[str] = None,
    batch_size: int = 128,
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """In silico evolve a set of sequences that are stored in a SeqData object using a PyTorch nn.Module.

    This function is a wrapper around the `evolution` function from the `seqexplainer`
    package. It takes a SeqData object containing sequences and evolves them in silico
    using the specified model. The evolved sequences are stored in the SeqData object
    as a new variable. The function returns the evolved sequences if `return_seqs` is
    set to True.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch nn.Module to evolve the sequences with
    sdata : xr.Dataset
        The SeqData object containing the sequences to evolve
    rounds : int
        The number of rounds of evolution to perform
    seq_var : str, optional
        The name of the sequence variable in the SeqData object, by default "ohe_seq"
    axis_order : tuple, optional
        The axis order of the sequence expected by the model. This is used to transpose
        the sequence data to the correct order before passing it to the model. The keys
        should be the names of the axes in the SeqData object. By default ("_sequence", "_ohe", "length")
    add_seqs : bool, optional
        Whether to add the evolved sequences to the SeqData object, by default True
    return_seqs : bool, optional
        Whether to return the evolved sequences, by default False
    device : str, optional
        The device to use for scoring the sequences, by default "cpu"
    batch_size : int, optional
        The batch size to use for scoring the sequences, by default 128.
    copy : bool, optional
        Whether to copy and return the copy of the SeqData object, by default False.

    Returns
    -------
    sdata   
        The SeqData object with the evolved sequences added to it if `copy` is False. If `return_seqs` is True, the
        function returns the evolved sequences as a torch.Tensor.
    """

    sdata = sdata.copy() if copy else sdata

    # Set device
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device

    # Grab seqs
    ohe_seqs = sdata[seq_var].transpose(*axis_order).to_numpy()
    evolved_seqs = np.zeros(ohe_seqs.shape)
    deltas = np.zeros((sdata.dims["_sequence"], rounds))

    # Evolve seqs
    for i, ohe_seq in tqdm(
        enumerate(ohe_seqs), total=len(ohe_seqs), desc="Evolving seqs"
    ):
        evolved_seq, delta, _ = evolution(model, ohe_seq, rounds=rounds, device=device)
        evolved_seqs[i] = evolved_seq
        deltas[i, :] = deltas[i, :] + delta

    # Get original scores
    orig_seqs = torch.tensor(ohe_seqs, dtype=torch.float32).to(device)
    original_scores = (
        model.predict(orig_seqs, batch_size=batch_size, verbose=False)
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )

    # Put evolved scores into sdata
    sdata["original_score"] = xr.DataArray(original_scores, dims="_sequence")
    sdata["evolved_1_score"] = xr.DataArray(
        original_scores + deltas[:, 0], dims="_sequence"
    )
    for i in range(2, rounds + 1):
        sdata[f"evolved_{i}_score"] = xr.DataArray(
            sdata[f"evolved_{i-1}_score"] + deltas[:, i - 1], dims="_sequence"
        )
    if return_seqs:
        evolved_seqs = torch.tensor(evolved_seqs, dtype=torch.float32)
        return evolved_seqs
    if add_seqs:
        sdata["evolved_seqs"] = xr.DataArray(
            evolved_seqs, dims=("_sequence", "_ohe", "length")
        )
    return sdata if copy else None
