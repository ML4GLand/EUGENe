import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, List
from ._utils import get_output_size
from . import _layers as layers


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_channels: int,
        output_channels: int,
        conv_kernel: int,
        conv_type: Union[str, Callable] = "conv1d",
        conv_stride: int = 1,
        conv_padding: Union[str, int] = "valid",
        conv_dilation: int = 1,
        conv_bias: bool = True,
        activation: Union[str, Callable] = "relu",
        pool_type: Union[str, Callable] = "max",
        pool_kernel: int = 1,
        pool_stride: int = None,
        pool_padding: int = 0,
        norm_type: Union[str, Callable] = "batchnorm",
        norm_dim: int = None,
        dropout_rate: float = 0.0,
        order: str = "conv-norm-act-pool-dropout",
    ):
        super(Conv1DBlock, self).__init__()

        # Define the block's attributes
        self.input_len = input_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_dilation = conv_dilation
        self.conv_padding = conv_padding
        self.conv_bias = conv_bias
        self.activation = activation
        self.pool_type = pool_type
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.norm_dim = norm_dim if norm_dim is not None else self.output_channels
        self.order = order

        # Define the conv layer
        self.conv_type = (
            layers.CONVOLUTION_REGISTRY[conv_type]
            if isinstance(conv_type, str)
            else conv_type
        )
        conv = self.conv_type(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=self.conv_kernel,
            stride=self.conv_stride,
            padding=self.conv_padding,
            dilation=self.conv_dilation,
            bias=self.conv_bias,
        )

        # Define the activation
        activation = (
            layers.ACTIVATION_REGISTRY[self.activation](inplace=False)
            if isinstance(self.activation, str)
            else self.activation
        )

        # Define the pooling layer
        if self.pool_type is not None:
            pool = layers.POOLING_REGISTRY[self.pool_type](
                kernel_size=self.pool_kernel,
                stride=self.pool_stride,
                padding=self.pool_padding,
            )
        else:
            pool = pool

        # Define the dropout layer
        if self.dropout_rate is not None and self.dropout_rate != 0:
            dropout = nn.Dropout(self.dropout_rate)
        else:
            dropout = None

        # Define the batchnorm layer
        if self.norm_type is not None:
            norm = layers.NORMALIZER_REGISTRY[self.norm_type](self.norm_dim)
        else:
            norm = norm_type

        # Define the order of the layers
        self.order = self.order.split("-")
        self.layers = nn.Sequential()
        for layer in self.order:
            if layer == "conv":
                self.layers.add_module("conv", conv)
            elif layer == "norm":
                if norm is not None:
                    self.layers.add_module("norm", norm)
            elif layer == "act":
                if self.activation is not None:
                    self.layers.add_module("act", activation)
            elif layer == "pool":
                if pool is not None:
                    self.layers.add_module("pool", pool)
            elif layer == "dropout":
                if dropout is not None:
                    self.layers.add_module("dropout", dropout)
            else:
                raise ValueError("Invalid layer type: {}".format(layer))

        self.output_size = get_output_size(
            self.layers, (self.input_channels, self.input_len)
        )

    def forward(self, x):
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [],
        activations: Union[str, Callable, List[Union[str, Callable]]] = "relu",
        dropout_rates: float = 0.0,
        batchnorm: bool = False,
        batchnorm_first: bool = False,
        biases: bool = True,
    ):
        super(DenseBlock, self).__init__()

        # Define the layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = len(hidden_dims) + 1
        self.hidden_dims = hidden_dims if len(hidden_dims) > 0 else [input_dim]

        # Define the activations
        activations = (
            activations
            if type(activations) == list
            else [activations] * len(hidden_dims)
        )
        self.activations = [
            layers.ACTIVATION_REGISTRY[activation](inplace=False)
            if isinstance(activation, str)
            else activation
            for activation in activations
        ]

        # Define the dropout rates
        if dropout_rates != 0.0 and dropout_rates is not None:
            if type(dropout_rates) == float:
                self.dropout_rates = [dropout_rates] * len(hidden_dims)
            elif type(dropout_rates) == list:
                self.dropout_rates = dropout_rates
        else:
            self.dropout_rates = []

        # Define the batchnorm
        self.batchnorm = batchnorm
        self.batchnorm_first = batchnorm_first

        # Define the biases
        self.biases = biases if isinstance(biases, list) else [biases] * self.num_layers

        # Build the block
        self.layers = nn.Sequential()
        j = 0
        for i in range(self.num_layers - 1):
            # Add the linear layer
            if i == 0:
                self.layers.append(
                    nn.Linear(self.input_dim, self.hidden_dims[0], bias=self.biases[0])
                )
            else:
                self.layers.append(
                    nn.Linear(
                        self.hidden_dims[i - 1],
                        self.hidden_dims[i],
                        bias=self.biases[i],
                    )
                )

            # Add batchnorm if specified
            if batchnorm and batchnorm_first:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i]))

            # Add activation
            if i < len(self.activations):
                if self.activations[i] is not None:
                    self.layers.append(self.activations[i])

            # Add dropout if specified
            if i < len(self.dropout_rates):
                if self.dropout_rates[i] is not None:
                    self.layers.append(nn.Dropout(self.dropout_rates[i]))

            # Add batchnorm if specified
            if batchnorm and not batchnorm_first:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i]))

            # Keep track of the number of layers seen
            j += 1

        # Add the final linear layer
        self.layers.append(
            nn.Linear(self.hidden_dims[-1], self.output_dim, bias=self.biases[-1])
        )

        # If enough droupout rates are specified, add the final dropout
        if len(self.dropout_rates) > j:
            self.layers.append(nn.Dropout(self.dropout_rates[j]))

    def forward(self, x):
        return self.layers(x)


class RecurrentBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        unit_type: str = "lstm",
        bidirectional: bool = False,
        dropout_rates: float = 0.0,
        bias=True,
        batch_first=True,
    ):
        super(RecurrentBlock, self).__init__()

        # Define input parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.unit_type = layers.RECURRENT_REGISTRY[unit_type]
        self.bidirectional = bidirectional
        self.dropout_rates = dropout_rates
        self.bias = bias
        self.batch_first = batch_first

        # Define recurrent layers
        self.layers = self.unit_type(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout_rates,
            bidirectional=self.bidirectional,
        )

        # Define output parameters
        self.out_channels = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )

    def forward(self, x):
        return self.layers(x)


BLOCK_REGISTRY = {
    "dense": DenseBlock,
    "conv1d": Conv1DBlock,
    "recurrent": RecurrentBlock,
}
