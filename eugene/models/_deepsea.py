import torch
import torch.nn as nn
import numpy as np

from ..utils import GetFlattenDim

class DeepSeaModule(nn.Module):
    # Need to modify this to have as many blocks as I want
    def __init__(self, input_len=1000, channels=[320, 480, 960], conv_kernels=8, pool_kernels=4, dropout_rates=[0.2, 0.2, 0.5]):
        """
        Generates a PyTorch module with architecture matching the convnet part of DeepSea. Default parameters are those specified in the DeepSea paper
        Parameters
        ----------
        input_len : int, input sequence length
        channels : list-like or int, channel width for each conv layer. If int each of the three layers will be the same channel width
        conv_kernels : list-like or int, conv kernel size for each conv layer. If int will be the same for all conv layers
        pool_kernels : list-like or int, maxpooling kernel size for the first two conv layers. If int will be the same for all conv layers
        dropout_rates : list-like or float, dropout rates for each conv layer. If int will be the same for all conv layers
        """
        super(DeepSeaModule, self).__init__()

        # If only one conv channel size provided, apply it to all convolutional layers
        if type(channels) == int:
            channels = list(np.full(3, channels))
        else:
            assert len(channels) == 3

        # If only one conv kernel size provided, apply it to all convolutional layers
        if type(conv_kernels) == int:
            conv_kernels = list(np.full_like(channels, conv_kernels))
        else:
            assert len(conv_kernels) == len(channels)

        # If only one dropout rate provided, apply it to all convolutional layers
        if type(dropout_rates) == float:
            dropout_rates = list(np.full_like(channels, dropout_rates))
        else:
            assert len(dropout_rates) == len(channels)

        # If only one maxpool kernel size provided, apply it to the first two conv layers
        if type(pool_kernels) == int:
            pool_kernels = list(np.full(3, pool_kernels))
        else:
            assert len(pool_kernels) == len(channels)-1

        # Build the architecture as a sequential model
        self.module = nn.Sequential(
            nn.Conv1d(4, channels[0], kernel_size=conv_kernels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernels[0], stride=pool_kernels[0]),
            nn.Dropout(p=dropout_rates[0]),
            nn.Conv1d(channels[0], channels[1], kernel_size=conv_kernels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernels[1], stride=pool_kernels[1]),
            nn.Dropout(p=dropout_rates[0]),
            nn.Conv1d(channels[1], channels[2], kernel_size=conv_kernels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rates[0]))

        # Get the dimensions if last conv output was flattened and store this, turn this into a function
        reduce_by = np.array(conv_kernels) - 1. # broadcast this
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (input_len - reduce_by[0]) / pool_kernels[0])
                 - reduce_by[1]) / pool_kernels[1])
            - reduce_by[2])
        #self.flatten_dim =  channels[2] * self.n_channels
        self.flatten_dim = GetFlattenDim(self.module, seq_len=input_len)*channels[-1]

    def forward(self, x):
        return self.module(x)
