import torch
import torch.nn as nn

from ._utils import GetFlattenDim, BuildFullyConnected

# Fully connected modules
class BasicFullyConnectedModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], **kwargs):
        """
        Parameters
        ----------
        input_dim : int
        output_dim : int
        hidden_dims : list-like
        kwargs : dict, keyword arguments for BuildFullyConnected function (e.g. dropout_rate)
        """
        super(BasicFullyConnectedModule, self).__init__()

        dlayers = [input_dim] + hidden_dims + [output_dim]
        self.module = BuildFullyConnected(dlayers, **kwargs)

    def forward(self, x):
        return self.module(x)


# Convolutional modules
class BasicConv1D(nn.Module):
    def __init__(self, input_len, channels, conv_kernels, pool_kernels, activation="relu", pool_strides=None, dropout_rates=0.0, batchnorm=False, omit_final_pool=False):
        """
        Generates a PyTorch module with a basic 1D convnet architecture (i.e. convolution, maxpooling, dropout, batchnorm)
        input_len : int, input sequence length
        channels : list-like or int, channel width for each conv layer.
        conv_kernels : list-like or int, conv kernel size for each conv layer.
        pool_kernels : list-like or int, maxpooling kernel size for the first two conv layers.
        dropout_rates : list-like or float, dropout rates for each conv layer.
        batchnorm : boolean, implementation of batch normalization on output.
        omit_final_pool : boolean, omit final max pooling after activation.
        """
        super(BasicConv1D, self).__init__()
        if pool_strides == None:
            pool_strides = pool_kernels
        if dropout_rates != 0.0:
            if type(dropout_rates) == float:
                dropout_rates = [dropout_rates] * len(channels)
            else:
                assert len(dropout_rates) == len(channels)
        net = []

        for i in range(1, len(channels)):
            net.append(nn.Conv1d(channels[i-1], channels[i], kernel_size=conv_kernels[i-1]))
            if activation=="relu":
                net.append(nn.ReLU(inplace=True))
            elif activation=="sigmoid":
                net.append(nn.Sigmoid())
            if not (omit_final_pool and (len(channels) - i == 1)): # Only omit max pool on final iteration
                if len(channels) != 2:
                    net.append(nn.MaxPool1d(kernel_size=pool_kernels[i-1], stride=pool_strides[i-1]))
            if len(channels) == 2 and not omit_final_pool:
                net.append(nn.MaxPool1d(kernel_size=pool_kernels[i-1], stride=pool_strides[i-1]))
            if dropout_rates != 0.0:
                net.append(nn.Dropout(dropout_rates[i]))
            if batchnorm:
                net.append(nn.BatchNorm1d(channels[i]))
        self.module = nn.Sequential(*net)
        self.out_channels = channels[-1]
        self.flatten_dim = GetFlattenDim(self.module, seq_len=input_len)*self.out_channels

    def forward(self, x):
        return self.module(x)


# Recurrent modules
class BasicRecurrent(nn.Module):
    def __init__(self, input_dim, output_dim, unit_type="lstm", bidirectional=False, **kwargs):
        super(BasicRecurrent, self).__init__()
        if unit_type == "lstm":
            self.module = nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=bidirectional, **kwargs)
        elif unit_type == "rnn":
            self.module = nn.RNN(input_size=input_dim, hidden_size=output_dim, bidirectional=bidirectional, **kwargs)
        elif unit_type == "gru":
            self.module = nn.RNN(input_size=input_dim, hidden_size=output_dim, bidirectional=bidirectional, **kwargs)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.out_dim = output_dim*2
        else:
            self.out_dim = output_dim

    def forward(self, x):
        return self.module(x)
