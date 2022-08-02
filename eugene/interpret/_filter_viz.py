import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
from ..utils import track
from .._settings import settings


def _get_activation(name):
    activation = {}

    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def _get_first_conv_layer_params(model):
    if model.__class__.__name__ == "Jores21CNN":
        return model.biconv.kernels[0].cpu()
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            pwms = next(layer.parameters()).cpu()
            return pwms
    print("No Conv1d layer found, returning None")
    return None


def _get_first_conv_layer(model):
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
        return layer.cpu()
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            first_layer = model.convnet.module[0]
            return first_layer
    print("No Conv1d layer found, returning None")
    return None


def _get_activations_from_layer(layer, sdataloader):
    from ..preprocessing import decode_DNA_seqs

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
        sequences.append(decode_DNA_seqs(x.transpose(2, 1).detach().cpu().numpy()))
        activations.append(F.relu(layer(x)).detach().cpu().numpy())
        np_act = np.concatenate(activations)
        np_seq = np.concatenate(sequences)
    return np_act, np_seq


def _get_filter_activators(activations, sequences, layer):
    kernel_size = layer.kernel_size[0]
    filter_activators = []
    for filt in range(activations.shape[1]):
        single_filter = activations[:, filt, :]
        max_val = np.max(single_filter)
        activators = []
        for i in range(len(single_filter)):
            starts = np.where(single_filter[i] > max_val / 2)[0]
            for start in starts:
                activators.append(sequences[i][start : start + kernel_size])
        filter_activators.append(activators)
    return filter_activators


def _get_pfms(filter_activators, kernel_size):
    filter_pfms = {}
    for i, activators in tqdm(
        enumerate(filter_activators),
        total=len(filter_activators),
        desc="Getting PFMs from filters",
    ):
        pfm = {
            "A": np.zeros(kernel_size),
            "C": np.zeros(kernel_size),
            "G": np.zeros(kernel_size),
            "T": np.zeros(kernel_size),
        }
        for seq in activators:
            for j, nt in enumerate(seq):
                pfm[nt][j] += 1
        filter_pfm = pd.DataFrame(pfm)
        filter_pfms[i] = filter_pfm
    return filter_pfms


@track
def generate_pfms(
    model, sdata, batch_size=None, num_workers=None, key_name="pfms", copy=False
):
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    sdata = sdata.copy() if copy else sdata
    sdataset = sdata.to_dataset(
        target=None, seq_transforms=None, transform_kwargs={"transpose": True}
    )
    sdataloader = DataLoader(sdataset, batch_size=batch_size, num_workers=num_workers)
    first_layer = _get_first_conv_layer(model)
    activations, sequences = _get_activations_from_layer(first_layer, sdataloader)
    filter_activators = _get_filter_activators(activations, sequences, first_layer)
    filter_pfms = _get_pfms(filter_activators, first_layer.kernel_size[0])
    sdata.uns[key_name] = filter_pfms
    return sdata if copy else None


# Adapted from gopher
def meme_generate(W, output_file="meme.txt", prefix="filter"):
    """generate a meme file for a set of filters, W ∈ (N,L,A)"""

    # background frequency
    nt_freqs = [1.0 / 4 for i in range(4)]

    # open file for writing
    f = open(output_file, "w")

    # print intro material
    f.write("MEME version 4\n")
    f.write("\n")
    f.write("ALPHABET= ACGT\n")
    f.write("\n")
    f.write("Background letter frequencies:\n")
    f.write("A %.4f C %.4f G %.4f T %.4f \n" % tuple(nt_freqs))
    f.write("\n")

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write("MOTIF %s%d \n" % (prefix, j))
        f.write("letter-probability matrix: alength= 4 w= %d nsites= %d \n" % (L, L))
        for i in range(L):
            f.write("%.4f %.4f %.4f %.4f \n" % tuple(pwm[i, :]))
        f.write("\n")

    f.close()
