import torch
from tqdm.auto import tqdm
from seqexplainer import evolution
from .._settings import settings
import xarray as xr
import numpy as np


def evolve_seqs_sdata(
    model: torch.nn.Module,
    sdata,
    rounds: int,
    seq_key: str = "ohe_seq",
    axis_order=("_sequence", "_ohe", "length"),
    add_seqs=True,
    return_seqs: bool = False,
    device: str = "cpu",
    batch_size: int = 128,
    copy: bool = False,
    **kwargs,
):
    """
    In silico evolve a set of sequences that are stored in a SeqData object.

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    sdata: SeqData
        The SeqData object containing the sequences to evolve
    rounds: int
        The number of rounds of evolution to perform
    return_seqs: bool, optional
        Whether to return the evolved sequences
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.
    copy: bool, optional
        Whether to copy the SeqData object before mutating it
    kwargs: dict, optional
        Additional arguments to pass to the evolution function

    Returns
    -------
    sdata: SeqData
        The SeqData object containing the evolved sequences
    """

    sdata = sdata.copy() if copy else sdata

    # Set device
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device

    # Grab seqs
    ohe_seqs = sdata[seq_key].transpose(*axis_order).to_numpy()
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
