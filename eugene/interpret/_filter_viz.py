import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
from ..utils import track
from .._settings import settings
from ..utils._decorators import nostdout


def _get_activation(name):
    activation = {}

    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def _get_first_conv_layer_params(model):
    if model.__class__.__name__ == "Jores21CNN":
        return model.biconv.kernels[0].cpu()
    elif model.__class__.__name__ == "Kopp21CNN":
        return model.conv.cpu()
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            pwms = next(layer.parameters()).cpu()
            return pwms
    print("No Conv1d layer found, returning None")
    return None


def _get_first_conv_layer(model, device="cpu"):
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


def _get_activations_from_layer(layer, sdataloader, device, alphabet="DNA"):
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
        with nostdout():
            sequences.append(
                decode_DNA_seqs(
                    x.transpose(2, 1).detach().cpu().numpy(),
                    vocab=alphabet,
                    verbose=False,
                )
            )
        # print(x.shape)
        x = x.to(device)
        layer = layer.to(device)
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


def _get_pfms(filter_activators, kernel_size, alphabet):
    filter_pfms = {}
    DNA = ["A", "C", "G", "T"]
    RNA = ["A", "C", "G", "U"]
    if alphabet == "DNA":
        bases = DNA
    elif alphabet == "RNA":
        bases = RNA
    else:
        raise ValueError("Alphabet must be either 'DNA' or 'RNA'.")

    for i, activators in tqdm(
        enumerate(filter_activators),
        total=len(filter_activators),
        desc="Getting PFMs from filters",
    ):
        pfm = {
            bases[0]: np.zeros(kernel_size),
            bases[1]: np.zeros(kernel_size),
            bases[2]: np.zeros(kernel_size),
            bases[3]: np.zeros(kernel_size),
        }
        for seq in activators:
            for j, nt in enumerate(seq):
                pfm[nt][j] += 1
        filter_pfm = pd.DataFrame(pfm)
        filter_pfms[i] = filter_pfm
    return filter_pfms


@track
def generate_pfms(
    model,
    sdata,
    batch_size=None,
    num_workers=None,
    key_name="pfms",
    alphabet="DNA",
    copy=False,
    device=None,
):
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    sdata = sdata.copy() if copy else sdata
    sdataset = sdata.to_dataset(
        target=None, seq_transforms=None, transform_kwargs={"transpose": True}
    )
    sdataloader = DataLoader(sdataset, batch_size=batch_size, num_workers=num_workers)
    first_layer = _get_first_conv_layer(model, device=device)
    activations, sequences = _get_activations_from_layer(
        first_layer, sdataloader, device=device, alphabet=alphabet
    )
    filter_activators = _get_filter_activators(activations, sequences, first_layer)
    filter_pfms = _get_pfms(
        filter_activators, first_layer.kernel_size[0], alphabet=alphabet
    )
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


def pwm_to_meme(pwm, output_file_path):
    """
    Function to convert pwm array to meme file
    :param pwm: numpy.array, pwm matrices, shape (U, 4, filter_size), where U - number of units
    :param output_file_path: string, the name of the output meme file
    """

    n_filters = pwm.shape[0]
    filter_size = pwm.shape[2]
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4\n\n")
    meme_file.write("ALPHABET= ACGT\n\n")
    meme_file.write("strands: + -\n\n")
    meme_file.write("Background letter frequencies\n")
    meme_file.write("A 0.25 C 0.25 G 0.25 T 0.25\n")

    print("Saved PWM File as : {}".format(output_file_path))

    for i in range(0, n_filters):
        if np.sum(pwm[i, :, :]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF filter%s\n" % i)
            meme_file.write(
                "letter-probability matrix: alength= 4 w= %d \n"
                % np.count_nonzero(np.sum(pwm[i, :, :], axis=0))
            )

        for j in range(0, filter_size):
            if np.sum(pwm[i, :, j]) > 0:
                meme_file.write(
                    str(pwm[i, 0, j])
                    + "\t"
                    + str(pwm[i, 1, j])
                    + "\t"
                    + str(pwm[i, 2, j])
                    + "\t"
                    + str(pwm[i, 3, j])
                    + "\n"
                )

    meme_file.close()
