import numpy as np
import torch.nn as nn


def GetFlattenDim(network, seq_len):
    """
    Get the dimension of the flattened output of a convolutional network with only Conv1d and Maxpool1d layers.

    Parameters
    ----------
    network : nn.Module
        network to get flattened dimension of
    seq_len : int
        length of the sequence to flatten

    Returns
    -------
    int
        flattened dimension of the network
    """
    output_len = seq_len
    for module in network:
        name = module.__class__.__name__
        if name == "Conv1d":
            if module.padding == "valid":
                output_len = output_len - module.kernel_size[0] + 1
            elif module.padding == "same":
                output_len = output_len
            else:
                assert isinstance(module.padding[0], int)
                output_len = output_len - module.kernel_size[0] + 1 + (2 * module.padding[0])
        elif name == "MaxPool1d":
            output_len = np.ceil((output_len - module.kernel_size + 1) / module.stride)
    return int(output_len)
