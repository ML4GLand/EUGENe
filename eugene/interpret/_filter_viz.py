import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from ..utils import track


def _get_activation(name):
    activation = {}
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def _get_first_conv_layer_params(model):
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            pwms = next(layer.parameters()).cpu()
            return pwms
    print("No Conv1d layer found, returning None")
    return None


def _get_first_conv_layer(model):
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            first_layer = model.convnet.module[0]
            return first_layer
    print("No Conv1d layer found, returning None")
    return None


def _get_activations_from_layer(layer, sdataloader):
    from ..preprocessing import decodeDNA
    activations = []
    sequences = []
    for i_batch, batch in tqdm(enumerate(sdataloader)):
        ID, x, x_rev_comp, y = batch
        sequences.append(decodeDNA(x.transpose(2,1).detach().cpu().numpy()))
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
            starts = np.where(single_filter[i] > max_val/2)[0]
            for start in starts:
                activators.append(sequences[i][start:start+kernel_size])
        filter_activators.append(activators)
    return filter_activators


def _get_pfms(filter_activators, kernel_size):
    filter_pfms = {}
    for i, activators in tqdm(enumerate(filter_activators)):
        pfm = {"A": np.zeros(kernel_size), "C": np.zeros(kernel_size), "G": np.zeros(kernel_size), "T": np.zeros(kernel_size)}
        for seq in activators:
            for j, nt in enumerate(seq):
                pfm[nt][j]+=1
        filter_pfm = pd.DataFrame(pfm)
        filter_pfms[i] = filter_pfm
    return filter_pfms


@track
def generate_pfms(model, sdata, copy=False):
    sdata = sdata.copy() if copy else sdata
    sdataset = sdata.to_dataset(label="TARGETS", seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
    sdataloader = DataLoader(sdataset, batch_size=32, num_workers=0)
    first_layer = _get_first_conv_layer(model)
    activations, sequences = _get_activations_from_layer(first_layer, sdataloader)
    filter_activators = _get_filter_activators(activations, sequences, first_layer)
    filter_pfms = _get_pfms(filter_activators, first_layer.kernel_size[0])
    sdata.uns["pfms"] = filter_pfms
    return sdata if copy else None
