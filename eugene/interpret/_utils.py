import torch
import numpy as np
from yuzu.utils import perturbations


def _k_largest_index_argsort(
    a: np.ndarray, 
    k: int = 1
) -> np.ndarray:
    """Returns the indeces of the k largest values of a numpy array. 
    
    If a is multi-dimensional, the indeces are returned as an array k x d array where d is 
    the dimension of a. The kth row represents the kth largest value of the overall array.
    The dth column returned repesents the index of the dth dimension of the kth largest value.
    So entry [i, j] in the return array represents the index of the jth dimension of the ith
    largets value in the overall array.

    a = array([[38, 14, 81, 50],
               [17, 65, 60, 24],
               [64, 73, 25, 95]])

    k_largest_index_argsort(a, k=2)
    array([[2, 3],  # first largest value is at [2,3] of the array (95)
           [0, 2]])  # second largest value is at [0,2] of the array (81)


    Parameters
    ----------
    a : numpy array
        The array to get the k largest values from.
    k : int
        The number of largest values to get.
    
    Returns
    -------
    numpy array
        The indexes of the k largest values of a.

    Note
    ----
    This is from:
    https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-arra
    """
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


@torch.inference_mode()
def _naive_ism(
    model: torch.nn.Module, 
    X_0: np.ndarray, 
    type: str = "delta", 
    batch_size: int = 128, 
    device: str = "cpu"):
    """
    In-silico mutagenesis saliency scores.
    
    This function will perform in-silico mutagenesis in a naive manner, i.e.,
    where each input sequence has a single mutation in it and the entirety
    of the sequence is run through the given model. It returns the ISM score,
    which is a vector of the L1 difference between the reference sequence
    and the perturbed sequences with one value for each output of the model.
    
    Parameters
    ----------
    model: torch.nn.Module
        The model to use.
    X_0: numpy.ndarray
        The one-hot encoded sequence to calculate saliency for.
    type: str, optional
        The type of ISM to perform. Can be either "delta" "l1", or "l2".
    batch_size: int, optional
        The size of the batches.
    device: str, optional
        Whether to use a 'cpu' or 'gpu'.
    
    Returns
    -------
    X_ism: numpy.ndarray
        The saliency score for each perturbation.

    Note
    ----
    This was modified from the Yuzu package
    """
    n_seqs, n_choices, seq_len = X_0.shape
    X_idxs = X_0.argmax(axis=1)

    X = perturbations(X_0)
    X_0_np = X_0
    X_0 = torch.from_numpy(X_0)

    if device[:4] != str(next(model.parameters()).device):
        model = model.to(device)

    if device[:4] != X_0.device:
        X_0 = X_0.to(device)

    model = model.eval()
    reference = model(X_0).unsqueeze(1)
    starts = np.arange(0, X.shape[1], batch_size)
    isms = []
    for i in range(n_seqs):
        X = perturbations(np.expand_dims(X_0_np[i], 0))
        y = []

        for start in starts:
            X_ = X[0, start : start + batch_size]
            if device[:4] == "cuda":
                X_ = X_.to(device)

            y_ = model(X_)
            y.append(y_)
            del X_

        y = torch.cat(y)
        if type == "delta":
            ism = (y - reference[i]).sum(axis=-1)
            if len(ism.shape) == 2:
                ism = ism.sum(axis=-1)
        elif type == "l1":
            ism = (y - reference[i]).abs().sum(axis=-1)
            if len(ism.shape) == 2:
                ism = ism.sum(axis=-1)
        elif type == "l2":
            ism = torch.square(y - reference[i]).sum(axis=-1)
            if len(ism.shape) == 2:
                ism = ism.sum(axis=-1)
            ism = torch.sqrt(ism)

        isms.append(ism)

        if device[:4] == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    isms = torch.stack(isms)
    isms = isms.reshape(n_seqs, seq_len, n_choices - 1)

    j_idxs = torch.arange(n_seqs * seq_len)
    X_ism = torch.zeros(n_seqs * seq_len, n_choices, device=device)
    for i in range(1, n_choices):
        i_idxs = (X_idxs.flatten() + i) % n_choices
        X_ism[j_idxs, i_idxs] = isms[:, :, i - 1].flatten()

    X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)

    if device[:4] == "cuda":
        X_ism = X_ism.cpu()

    X_ism = X_ism.numpy()
    return X_ism