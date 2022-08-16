import torch
import numpy as np
from tqdm.auto import tqdm
from yuzu.naive_ism import naive_ism
from ._utils import k_largest_index_argsort
from .. import settings
#from ..preprocessing import decode_DNA_seq

def best_k_muts(
    model,
    X: np.ndarray,
    k: str = 1,
    device: str =None
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
    X: numpy.ndarray
        The one-hot encoded sequence to calculate find mutations for.
    k: int, optional
        The number of mutated seqences to return
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.
    
    Returns
    -------
    mut_X: numpy.ndarray
        The k highest scoring one-hot-encoded sequences
    maxs: numpy.ndarray
        The k highest delta scores corresponding to the k highest scoring sequences
    locs: numpy.ndarray
        The indeces along the length of the sequences where the mutations can be found
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    X = np.expand_dims(X, axis=0) if X.ndim == 2 else X
    X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    X = torch.Tensor(X).float().numpy()
    X_ism = naive_ism(model, X, device=device, batch_size=1)
    X_ism = X_ism.squeeze(axis=0)
    inds = k_largest_index_argsort(X_ism, k)
    locs = inds[:, 1]
    maxs = np.max(X_ism, axis=0)[locs]
    #print(inds, locs, maxs)
    #_max, ind = np.max(X_ism), np.unravel_index(X_ism.argmax(), X_ism.shape)
    mut_Xs = np.zeros((k, X.shape[2], X.shape[1]))
    #print(mut_Xs.shape)
    for i in range(k):
        mut_X = X.copy().transpose(0, 2, 1).squeeze(axis=0)
        mut_X[inds[i][1]] = np.zeros(mut_X.shape[1])
        mut_X[inds[i][1]][inds[i][0]] = 1
        #print(mut_X.shape)
        mut_Xs[i] = mut_X
    return mut_Xs, maxs, locs


def best_mut_seqs(
    model,
    X: np.ndarray,
    batch_size: int = None,
    device: str = None
) -> np.ndarray:
    """
    Find and return the highest scoring sequence for each sequence 
    from a set reference sequences X. X should contain one-hot-encoded sequences
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
    X: numpy.ndarray
        The one-hot encoded sequences to calculate find mutations for.
    batch_size: int, optional
        The number of sequences to score at once.
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.
    
    Returns
    -------
    mut_X: numpy.ndarray
        The highest scoring one-hot-encoded sequences
    maxs: numpy.ndarray
        The highest delta scores corresponding to the highest scoring sequences
    locs: numpy.ndarray
        The indeces along the length of the sequences where the mutations can be found

    """

    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = settings.batch_size if batch_size is None else batch_size
    X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    X = torch.Tensor(X).float().numpy()
    #print(X.shape)
    X_ism = naive_ism(model, X, device="cpu", batch_size=batch_size)
    maxs, inds, mut_X = [], [], X.copy().transpose(0, 2, 1)
    #print(mut_X.shape)
    for i in range(len(mut_X)):
        maxs.append(np.max(X_ism[i]))
        ind = np.unravel_index(X_ism[i].argmax(), X_ism[i].shape)
        #print(ind)
        inds.append(ind[1])
        mut_X[i][ind[1]] = np.zeros(mut_X.shape[2])
        mut_X[i][ind[1]][ind[0]] = 1
    return mut_X, np.array(maxs), np.array(inds)


def evolution(
    model,
    X: np.ndarray,
    rounds: int = 10,
    force_different: bool = True,
    k: int = 10,
    batch_size: int = None,
    device: str = None
) -> np.ndarray:
    """
    Perform rounds rounds of in-silico evolution on a single sequence X.

    Using ISM, this function calculates all the scores of all possible mutations
    on a starting sequence X. It then mutates the sequence and repeats the process
    for rounds rounds. In the end, it returns new "evolved" sequence after rounds mutations
    and the delta scores from the reference sequence and the indeces along the lengths of the sequences
    with which the mutations occured

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: numpy.ndarray
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
    curr_X = X.copy()
    mutated_positions, mutated_scores = [], []
    for r in range(rounds):
        #print(f"Round {r}")
        mut_X, score, positions = best_k_muts(model, curr_X, k=10, device=device)
        #print(mut_X.shape, positions)
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
        #print(decode_DNA_seq(curr_X))
    return curr_X, mutated_scores, mutated_positions


def evolve_sdata(
    model,
    sdata,
    rounds,
    return_seqs=False,
    **kwargs
):
    model.eval()
    evolved_seqs = np.zeros(sdata.ohe_seqs.shape)
    for i, ohe_seq in tqdm(enumerate(sdata.ohe_seqs), total=len(sdata.ohe_seqs), desc="Evolving seqs"):
        evolved_seq = evolution(
            model,
            ohe_seq,
            rounds=rounds
        )[0]
        evolved_seqs[i] = evolved_seq
    
    original_scores = model(torch.Tensor(sdata.ohe_seqs.transpose(0, 2, 1))).detach().numpy()
    evolved_scores = model(torch.Tensor(evolved_seqs.transpose(0, 2, 1))).detach().numpy()
    sdata["original_scores"] = original_scores
    sdata[f"evolved_{rounds}_scores"] = evolved_scores
    if return_seqs:
        return evolved_seqs
    