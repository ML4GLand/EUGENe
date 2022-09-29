import torch
import numpy as np
from tqdm.auto import tqdm
from ._utils import _k_largest_index_argsort, _naive_ism
from ..preprocess import ohe_seqs, feature_implant_across_seq
from ..utils import track
from .. import settings


def best_k_muts(
    model: torch.nn.Module, 
    X: np.ndarray, 
    k: int = 1, 
    device: str = None
) -> np.ndarray:
    """
    Find and return the k highest scoring sequence from referenece sequence X.

    Using ISM, this function calculates all the scores of all possible mutations
    of the reference sequence using a trained model. It then returns the k highest
    scoring sequences, along with delta scores from the reference sequence and the indeces
    along the lengths of the sequences where the mutations can be found

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded sequence to calculate find mutations for.
    k: int, optional
        The number of mutated seqences to return
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    mut_X: np.ndarray
        The k highest scoring one-hot-encoded sequences
    maxs: np.ndarray
        The k highest delta scores corresponding to the k highest scoring sequences
    locs: np.ndarray
        The indeces along the length of the sequences where the mutations can be found
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    X = np.expand_dims(X, axis=0) if X.ndim == 2 else X
    X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    X = torch.Tensor(X).float().numpy()
    X_ism = _naive_ism(model, X, device=device, batch_size=1)
    X_ism = X_ism.squeeze(axis=0)
    inds = _k_largest_index_argsort(X_ism, k)
    locs = inds[:, 1]
    maxs = np.max(X_ism, axis=0)[locs]
    mut_Xs = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        mut_X = X.copy().squeeze(axis=0)
        mut_X[:, inds[i][1]] = np.zeros(mut_X.shape[0])
        mut_X[:, inds[i][1]][inds[i][0]] = 1
        mut_Xs[i] = mut_X
    return mut_Xs, maxs, locs


def best_mut_seqs(
    model: torch.nn.Module,
    X: np.ndarray, 
    batch_size: int = None, 
    device: str = None
) -> np.ndarray:
    """Find and return the highest scoring sequence for each sequence from a set reference sequences X. 
    
    X should contain one-hot-encoded sequences
    and should be of shape (n, 4, l). n is the number of sequences, 4 is the number of
    nucleotides, and l is the length of the sequence.

    Using ISM, this function calculates all the scores of all possible mutations
    of each reference sequence using a trained model. It then returns the highest
    scoring sequence, along with delta scores from the reference sequence and the indeces
    along the lengths of the sequences where the mutations can be found.

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded sequences to calculate find mutations for.
    batch_size: int, optional
        The number of sequences to score at once.
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    mut_X: np.ndarray
        The highest scoring one-hot-encoded sequences
    maxs: np.ndarray
        The highest delta scores corresponding to the highest scoring sequences
    locs: np.ndarray
        The indeces along the length of the sequences where the mutations can be found

    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = settings.batch_size if batch_size is None else batch_size
    model.eval().to(device)
    X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    X = torch.Tensor(X).float().numpy()
    X_ism = _naive_ism(model, X, device=device, batch_size=batch_size)
    maxs, inds, mut_X = [], [], X.copy()
    for i in range(len(mut_X)):
        maxs.append(np.max(X_ism[i]))
        ind = np.unravel_index(X_ism[i].argmax(), X_ism[i].shape)
        inds.append(ind[1])
        mut_X[i][:, ind[1]] = np.zeros(mut_X.shape[1])
        mut_X[i][:, ind[1]][ind[0]] = 1
    return mut_X, np.array(maxs), np.array(inds)


def evolution(
    model: torch.nn.Module,
    X: np.ndarray,
    rounds: int = 10,
    k: int = 10,
    force_different: bool = True,
    batch_size: int = None,
    device: str = "cpu",
) -> np.ndarray:
    """Perform rounds rounds of in-silico evolution on a single sequence X.

    Using ISM, this function calculates all the scores of all possible mutations
    on a starting sequence X. It then mutates the sequence and repeats the process
    for rounds rounds. In the end, it returns new "evolved" sequence after rounds mutations
    and the delta scores from the reference sequence and the indeces along the lengths of the sequences
    with which the mutations occured

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded reference sequence to start the evolution with
    rounds: int, optional
        The number of rounds of evolution to perform
    force_different: bool, optional
        Whether to force the mutations to occur at different locations in the reference sequence
    k: int, optional
        The number of mutated sequences to consider at each round. This is in case
        the same position scores highest multiple times.
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = settings.batch_size if batch_size is None else batch_size
    model.eval().to(device)
    curr_X = X.copy()
    mutated_positions, mutated_scores = [], []
    for r in range(rounds):
        mut_X, score, positions = best_k_muts(model, curr_X, k=k, device=device)
        if force_different:
            for i, p in enumerate(positions):
                if p not in mutated_positions:
                    curr_X = mut_X[i]
                    mutated_positions.append(p)
                    mutated_scores.append(score[i])
                    break
        else:
            curr_X = mut_X[0]
            mutated_positions.append(positions[0])
            mutated_scores.append(score[0])
    return curr_X, mutated_scores, mutated_positions


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
