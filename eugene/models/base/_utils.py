import torchinfo
import numpy as np
import torch.nn as nn


def get_conv1dblock_output_len(modules, input_len):
    """
    Get the dimension of the flattened output of a convolutional modules with only Conv1d and Maxpool1d layers.
    This will be deprecated in the future.

    Parameters
    ----------
    modules : nn.Module
        modules to get flattened dimension of
    input_len : int
        length of the sequence to flatten

    Returns
    -------
    int
        flattened dimension of themodules
    """
    output_len = input_len
    for module in modules:
        name = module.__class__.__name__
        if name == "Conv1d":
            if module.padding == "valid":
                output_len = output_len - module.kernel_size[0] + 1
            elif module.padding == "same":
                output_len = output_len
            else:
                assert isinstance(module.padding[0], int)
                output_len = (
                    output_len - module.kernel_size[0] + 1 + (2 * module.padding[0])
                )
        elif name == "MaxPool1d" or name == "AvgPool1d":
            if isinstance(module.kernel_size, tuple):
                module.kernel_size = module.kernel_size[0]
            if isinstance(module.stride, tuple):
                module.stride = module.stride[0]
            output_len = np.ceil((output_len - module.kernel_size + 1) / module.stride)
    return int(output_len)


def get_output_size(modules, input_size):
    if isinstance(input_size, int):
        input_size = (input_size,)
    summary = torchinfo.summary(
        modules, input_size=(1, *input_size), verbose=0, device="cpu"
    )
    out_size = summary.summary_list[-1].output_size[1:]
    return out_size
