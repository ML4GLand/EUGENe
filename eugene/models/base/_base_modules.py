from typing import Union
import torch.nn as nn
from ._utils import GetFlattenDim, BuildFullyConnected


class BasicFullyConnectedModule(nn.Module):
    """Instantiate a PyTorch module with a basic fully connected module.
    (i.e. input, output, layers, activation, dropout, and batchnorm)

    Parameters
    ----------
    input_dim : int
        The input dimension of the model.
    output_dim : int
        The output dimension of the model.
    hidden_dims : list-like
        The hidden dimensions of the model.
    kwargs : dict
        keyword arguments for BuildFullyConnected function (e.g. dropout_rate)

    Returns
    -------
    nn.Module
        The instantiated fully connected module.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims=[], 
        **kwargs
    ):
        super(BasicFullyConnectedModule, self).__init__()
        dlayers = [input_dim] + hidden_dims + [output_dim]
        self.module = BuildFullyConnected(dlayers, **kwargs)

    def forward(self, x):
        return self.module(x)


# Convolutional modules
class BasicConv1D(nn.Module):
    """Generates a PyTorch module with a basic 1D convnet architecture
    (i.e. convolution, maxpooling, dropout, batchnorm)

    Parameters
    ----------
    input_len : int
        input sequence length
    channels : list-like or int
        channel width for each conv layer.
    conv_kernels : list-like or int
        conv kernel size for each conv layer.
    pool_kernels : list-like or int
        maxpooling kernel size for the first two conv layers.
    dropout_rates : list-like or float
        dropout rates for each conv layer.
    batchnorm : boolean
        implementation of batch normalization on output.
    omit_final_pool : boolean
        omit final max pooling after activation.

    Returns
    -------
    nn.Module
        The instantiated 1D convolutional module.
    """
    def __init__(
        self,
        input_len: int,
        channels: list,
        conv_kernels: list,
        pool_kernels: list,
        activation: str = "relu",
        pool_strides: list = None,
        dropout_rates: float = 0.0,
        dilations: list = None,
        padding: Union[str, list] = "valid",
        batchnorm: bool = False,
        omit_final_pool: bool =False
    ):
        super(BasicConv1D, self).__init__()
        pool_strides = pool_kernels if pool_strides is None else pool_strides
        if dropout_rates != 0.0:
            if type(dropout_rates) == float:
                dropout_rates = [dropout_rates] * len(channels)
            else:
                assert len(dropout_rates) == (len(channels) - 1)
        dilations = [1] * (len(channels) - 1) if dilations is None else dilations
        if isinstance(padding, str):
            padding = [padding] * (len(channels) - 1)
        else:
            assert len(padding) == (len(channels) - 1)
        net = []
        for i in range(1, len(channels)):
            net.append(
                nn.Conv1d(
                    channels[i - 1], 
                    channels[i], 
                    kernel_size=conv_kernels[i - 1], 
                    dilation=dilations[i - 1],
                    padding=padding[i - 1]
                )
            )
            if activation == "relu":
                net.append(nn.ReLU(inplace=False))
            elif activation == "sigmoid":
                net.append(nn.Sigmoid())
            if i == len(channels) - 1:
                if not omit_final_pool:
                    net.append(nn.MaxPool1d(kernel_size=pool_kernels[i - 1], stride=pool_strides[i - 1]))
            else:
                net.append(nn.MaxPool1d(kernel_size=pool_kernels[i - 1], stride=pool_strides[i - 1]))
            if dropout_rates != 0.0:
                net.append(nn.Dropout(dropout_rates[i - 1]))
            if batchnorm:
                net.append(nn.BatchNorm1d(channels[i]))
        self.module = nn.Sequential(*net)
        self.out_channels = channels[-1]
        self.flatten_dim = (GetFlattenDim(self.module, seq_len=input_len) * self.out_channels)

    def forward(self, x):
        return self.module(x)


# Recurrent modules
class BasicRecurrent(nn.Module):
    """Instantiate a PyTorch module with a basic 1D recurrent architecture

    Parameters
    ----------
    input_dim : int
        input dimension of the model.
    output_dim : int
        output dimension of the model.
    unit_type : str
        type of recurrent unit.
    bidirectional : boolean
        whether to use bidirectional recurrent unit.
    kwargs : dict
        keyword arguments for nn.LSTM or nn.RNN.

    Returns
    -------
    nn.Module
        The instantiated recurrent module.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        unit_type: str ="lstm", 
        bidirectional: bool = False, 
        **kwargs
    ):
        super(BasicRecurrent, self).__init__()
        if unit_type == "lstm":
            self.module = nn.LSTM(
                input_size=input_dim,
                hidden_size=output_dim,
                bidirectional=bidirectional,
                **kwargs
            )
        elif unit_type == "rnn":
            self.module = nn.RNN(
                input_size=input_dim,
                hidden_size=output_dim,
                bidirectional=bidirectional,
                **kwargs
            )
        elif unit_type == "gru":
            self.module = nn.RNN(
                input_size=input_dim,
                hidden_size=output_dim,
                bidirectional=bidirectional,
                **kwargs
            )
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.out_dim = output_dim * 2
        else:
            self.out_dim = output_dim

    def forward(self, x):
        return self.module(x)
