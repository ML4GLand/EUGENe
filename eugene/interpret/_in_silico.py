import torch

@track
def evolve_seqs_sdata(
    model: torch.nn.Module, 
    sdata, 
    rounds: int, 
    return_seqs: bool = False, 
    device: str = "cpu", 
    copy: bool = False, 
    **kwargs
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
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    evolved_seqs = np.zeros(sdata.ohe_seqs.shape)
    deltas = np.zeros((sdata.n_obs, rounds))
    for i, ohe_seq in tqdm(enumerate(sdata.ohe_seqs), total=len(sdata.ohe_seqs), desc="Evolving seqs"):
        evolved_seq, delta, _ = evolution(model, ohe_seq, rounds=rounds, device=device)
        evolved_seqs[i] = evolved_seq
        deltas[i, :] = deltas[i, :] + delta
    orig_seqs = torch.Tensor(sdata.ohe_seqs).to(device)
    original_scores = model(orig_seqs).detach().cpu().numpy().squeeze()
    sdata["original_score"] = original_scores
    sdata["evolved_1_score"] = original_scores + deltas[:, 0]
    for i in range(2, rounds + 1):
        sdata.seqs_annot[f"evolved_{i}_score"] = (
            sdata.seqs_annot[f"evolved_{i-1}_score"] + deltas[:, i - 1]
        )
    print(return_seqs)
    if return_seqs:
        evolved_seqs = torch.Tensor(evolved_seqs).to(device)
        return evolved_seqs
    return sdata if copy else None


def feature_implant_seq_sdata(
    model: torch.nn.Module,
    sdata,
    seq_id: str,
    feature: np.ndarray,
    feature_name: str = "feature",
    encoding: str = "str",
    onehot: bool = False,
    store: bool = True,
    device: str = "cpu",
):
    """
    Score a set of sequences with a feature inserted at every position of each sequence in sdata.
    For double stranded models, the feature is inserted in both strands, with the reverse complement
    of the feature in the reverse strand

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    sdata: SeqData
        The SeqData object containing the sequences to score
    seq_id: str
        The id of the sequence to score
    feature: np.ndarray
        The feature to insert into the sequences
    feature_name: str, optional
        The name of the feature
    encoding: str, optional
        The encoding of the feature. One of 'str', 'ohe', 'int'
    onehot: bool, optional
        Whether the feature is one-hot encoded
    store: bool, optional
        Whether to store the scores in the SeqData object
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        The scores of the sequences with the feature inserted if store is False
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    seq_idx = np.where(sdata.seqs_annot.index == seq_id)[0][0]
    if encoding == "str":
        seq = sdata.seqs[seq_idx]
        implanted_seqs = feature_implant_across_seq(seq, feature, encoding=encoding)
        implanted_seqs = ohe_seqs(implanted_seqs, vocab="DNA", verbose=False)
        X = torch.from_numpy(implanted_seqs).float()
    elif encoding == "onehot":
        seq = sdata.ohe_seqs[seq_idx]
        implanted_seqs = feature_implant_across_seq(
            seq, 
            feature, 
            encoding=encoding, 
            onehot=onehot
        )
        X = torch.from_numpy(implanted_seqs).float()
    else:
        raise ValueError("Encoding not recognized.")
    if model.strand == "ss":
        X = X.to(device)
        X_rev = X
    else:
        X = X.to(device)
        X_rev = torch.flip(X, [1, 2]).to(device)
    preds = model(X, X_rev).cpu().detach().numpy().squeeze()
    if store:
        sdata.seqsm[f"{seq_id}_{feature_name}_slide"] = preds
    return preds


def feature_implant_seqs_sdata(
    model: torch.nn.Module,
    sdata,
    feature: np.ndarray,
    seqsm_key: str = None, 
    device: str = "cpu",
    **kwargs
):
    """
    Score a set of sequences with a feature inserted at every position of each sequence in sdata

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    sdata: SeqData
        The SeqData object containing the sequences to score
    feature: np.ndarray
        The feature to insert into the sequences
    seqsm_key: str, optional
        The key to store the scores in the SeqData object
    kwargs: dict, optional
        Additional arguments to pass to the feature_implant_seq_sdata function
    
    Returns
    -------
    np.ndarray
        The scores of the sequences with the feature inserted if seqsm_key is None
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    predictions = []
    for i, seq_id in tqdm(
        enumerate(sdata.seqs_annot.index),
        desc="Implanting feature in all seqs of sdata",
        total=len(sdata.seqs_annot),
    ):
        predictions.append(
            feature_implant_seq_sdata(
                model, 
                sdata, 
                seq_id, 
                feature, 
                store=False, 
                **kwargs
            )
        )
    if seqsm_key is not None:
        sdata.seqsm[seqsm_key] = np.array(predictions)
    else:
        return np.array(predictions)

import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt