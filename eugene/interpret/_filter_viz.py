import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
from ..preprocess._utils import _get_vocab
from ._utils import _k_largest_index_argsort
from ..utils import track
from .._settings import settings


def _get_first_conv_layer(model, device="cpu"):
    """
    Get the first convolutional layer in a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the first convolutional layer from.
    device : str, optional
        The device to move the model to, by default "cpu"

    Returns
    -------
    torch.nn.Module
        The first convolutional layer in the model.
    """
    if model.__class__.__name__ == "Jores21CNN":
        layer_shape = model.biconv.kernels[0].shape
        kernels = model.biconv.kernels[0]
        biases = model.biconv.biases[0]
        layer = nn.Conv1d(
            in_channels=layer_shape[1],
            out_channels=layer_shape[0],
            kernel_size=layer_shape[2],
            padding="same",
        )
        layer.weight = nn.Parameter(kernels)
        layer.bias = nn.Parameter(biases)
        return layer.to(device)
    elif model.__class__.__name__ == "Kopp21CNN":
        return model.conv
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            first_layer = model.convnet.module[0]
            return first_layer.to(device)
    print("No Conv1d layer found, returning None")
    return None


def _get_activations_from_layer(layer, sdataloader, device="cpu", vocab="DNA"):
    from ..preprocess import decode_seqs

    activations = []
    sequences = []
    dataset_len = len(sdataloader.dataset)
    batch_size = sdataloader.batch_size
    for i_batch, batch in tqdm(
        enumerate(sdataloader),
        total=int(dataset_len / batch_size),
        desc="Getting maximial activating seqlets",
    ):
        ID, x, x_rev_comp, y = batch
        sequences.append(
            decode_seqs(x.detach().cpu().numpy(), vocab=vocab, verbose=False)
        )
        x = x.to(device)
        layer = layer.to(device)
        activations.append(F.relu(layer(x)).detach().cpu().numpy())
        np_act = np.concatenate(activations)
        np_seq = np.concatenate(sequences)
    return np_act, np_seq


def _get_filter_activators(
    activations,
    sequences,
    kernel_size,
    num_filters=None,
    method="Alipahani15",
    threshold=0.5,
    num_seqlets=100,
):
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    if method == "Alipahani15":
        assert (
            threshold is not None
        ), "Threshold must be specified for Alipanahi15 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            max_val = np.max(single_filter)
            activators = []
            for i in range(len(single_filter)):
                starts = np.where(single_filter[i] >= max_val * threshold)[0]
                for start in starts:
                    activators.append(sequences[i][start : start + kernel_size])
            filter_activators.append(activators)
    elif method == "Minnoye20":
        assert (
            num_seqlets is not None
        ), "num_seqlets must be specified for Minnoye20 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            inds = _k_largest_index_argsort(single_filter, num_seqlets)
            filter_activators.append(
                [
                    seq[inds[i][1] : inds[i][1] + kernel_size]
                    for i, seq in enumerate(sequences[inds[:, 0]])
                ]
            )
    return filter_activators


def _get_pfms(
    filter_activators,
    kernel_size,
    vocab="DNA",
):
    filter_pfms = {}
    vocab = _get_vocab(vocab)
    for i, activators in tqdm(
        enumerate(filter_activators),
        total=len(filter_activators),
        desc="Getting PFMs from filters",
    ):
        pfm = {
            vocab[0]: np.zeros(kernel_size),
            vocab[1]: np.zeros(kernel_size),
            vocab[2]: np.zeros(kernel_size),
            vocab[3]: np.zeros(kernel_size),
            "N": np.zeros(kernel_size),
        }
        for seq in activators:
            for j, nt in enumerate(seq):
                pfm[nt][j] += 1
        filter_pfm = pd.DataFrame(pfm)
        filter_pfm = filter_pfm.drop("N", axis=1)
        filter_pfms[i] = filter_pfm
        filter_pfms[i] = filter_pfms[i].div(filter_pfms[i].sum(axis=1), axis=0)
    return filter_pfms


@track
def generate_pfms_sdata(
    model,
    sdata,
    method="Alipahani15",
    vocab="DNA",
    num_filters=None,
    threshold=0.5,
    num_seqlets=100,
    batch_size=None,
    num_workers=None,
    device="cpu",
    transform_kwargs={},
    key_name="pfms",
    copy=False,
    **kwargs,
):
    sdata = sdata.copy() if copy else sdata
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    sdataset = sdata.to_dataset(target_keys=None, transform_kwargs=transform_kwargs)
    sdataloader = DataLoader(
        sdataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    first_layer = _get_first_conv_layer(model, device=device)
    activations, sequences = _get_activations_from_layer(
        first_layer, sdataloader, device=device, vocab=vocab
    )
    filter_activators = _get_filter_activators(
        activations,
        sequences,
        first_layer.kernel_size[0],
        num_filters=num_filters,
        method=method,
        threshold=threshold,
        num_seqlets=num_seqlets,
    )
    filter_pfms = _get_pfms(filter_activators, first_layer.kernel_size[0], vocab=vocab)
    sdata.uns[key_name] = filter_pfms
    return sdata if copy else None
