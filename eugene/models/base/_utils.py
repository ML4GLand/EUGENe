import torch
import torch.nn as nn
import numpy as np


def BuildFullyConnected(layers, activation="relu", dropout_rate=0.0, batchnorm=False):
    """
    Parameters
    ----------
    layers : int
    activation : str
    dropout_rate : float
    batchnorm: boolean
    """
    net = []
    for i in range(1, len(layers) - 1):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU(inplace=False))
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout_rate != 0.0:
            net.append(nn.Dropout(dropout_rate))
        if batchnorm:
            net.append(nn.BatchNorm1d(layers[i]))
    net.append(nn.Linear(layers[-2], layers[-1]))
    return nn.Sequential(*net)


def GetFlattenDim(network, seq_len):
    output_len = seq_len
    for module in network:
        name = module.__class__.__name__
        if name == "Conv1d":
            output_len = output_len - module.kernel_size[0] + 1
        elif name == "MaxPool1d":
            output_len = np.ceil((output_len - module.kernel_size + 1) / module.stride)
    return int(output_len)
