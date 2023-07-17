import torch
from torch import nn
import torch.nn.functional as F
from inspect import signature
from typing import Type, Dict, Union, Callable, List, Any
from ._utils import get_output_size, get_conv1dblock_output_len
from . import _layers as layers


class Tower(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        repeats: int,
        input_size: tuple,
        static_block_args: Dict[str, Any] = None,
        dynamic_block_args: Dict[str, Any] = None,
        mults: Dict[str, float] = None,
        name: str = "tower",
    ):
        """A tower of blocks.

        Parameters
        ----------
        block : Type[nn.Module]
        repeats : int
        static_block_args : Dict[str, Any]
            Arguments to initialize blocks that are static across repeats.
        dynamic_block_args : Dict[str, Any]
            Arguments to initialize blocks that change across repeats.
        mults : Dict[str, float]
            Multipliers for dynamic block arguments.
        """
        super().__init__()
        self.input_size = input_size
        self.repeats = repeats
        self.block_name = block.__name__.lower()
        self.name = name

        blocks = nn.ModuleList()
        if static_block_args is None:
            static_block_args = {}
        if dynamic_block_args is None:
            dynamic_block_args = {}
        if mults is None:
            mults = {}

        for arg, mult in mults.items():
            # replace initial value with geometric progression
            init_val = dynamic_block_args.get(
                arg, signature(block).parameters[arg].default
            )
            dynamic_block_args[arg] = (
                (
                    init_val
                    * torch.logspace(start=0, end=repeats - 1, steps=repeats, base=mult)
                )
                .to(dtype=signature(block).parameters[arg].annotation)
                .tolist()
            )

        self.blocks = nn.Sequential()
        for i in range(repeats):
            args = {arg: vals[i] for arg, vals in dynamic_block_args.items()}
            args.update(static_block_args)
            self.blocks.add_module(f"{self.block_name}_{i}", block(**args))

        self.output_size = get_output_size(self.blocks, self.input_size)

    def forward(self, x):
        return self.blocks(x)


class Conv1DTower(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_channels: int,
        conv_channels: list,
        conv_kernels: list,
        conv_strides: list = None,
        conv_dilations: list = None,
        conv_padding: Union[str, list] = "valid",
        conv_biases: bool = True,
        activations: Union[str, Callable, List[Union[str, Callable]]] = "relu",
        pool_types: Union[str, Callable, List[Union[str, Callable]]] = "max",
        pool_kernels: list = None,
        pool_strides: list = None,
        pool_dilations: list = None,
        pool_padding: list = None,
        dropout_rates: float = 0.0,
        batchnorm: bool = False,
        batchnorm_first: bool = False,
    ):
        super(Conv1DTower, self).__init__()

        # Define input parameters
        self.input_len = input_len
        self.input_channels = input_channels

        # Define convolutional layers
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.conv_strides = (
            conv_strides if conv_strides is not None else [1] * len(conv_channels)
        )
        self.conv_dilations = (
            conv_dilations if conv_dilations is not None else [1] * len(conv_channels)
        )
        self.conv_padding = (
            conv_padding
            if type(conv_padding) == list
            else [conv_padding] * len(conv_channels)
        )
        self.conv_biases = (
            conv_biases
            if type(conv_biases) == list
            else [conv_biases] * len(conv_channels)
        )
        assert (
            len(self.conv_channels)
            == len(self.conv_kernels)
            == len(self.conv_strides)
            == len(self.conv_dilations)
            == len(self.conv_padding)
            == len(self.conv_biases)
        ), "Convolutional parameters must be of the same length"

        # Define activation layers
        activations = (
            activations
            if type(activations) == list
            else [activations] * len(conv_channels)
        )
        self.activations = [
            layers.ACTIVATION_REGISTRY[activation](inplace=False)
            if isinstance(activation, str)
            else activation
            for activation in activations
        ]

        # Define pooling layers
        pool_types = (
            pool_types
            if type(pool_types) == list
            else [pool_types] * len(conv_channels)
        )
        self.pool_types = [
            layers.POOLING_REGISTRY[pool_type]
            if isinstance(pool_type, str)
            else pool_type
            for pool_type in pool_types
        ]
        self.pool_kernels = (
            pool_kernels if pool_kernels is not None else [1] * len(self.pool_types)
        )
        self.pool_strides = (
            pool_strides if pool_strides is not None else [1] * len(self.pool_types)
        )
        self.pool_padding = (
            pool_padding if pool_padding is not None else [0] * len(self.pool_types)
        )
        self.pool_dilations = (
            pool_dilations if pool_dilations is not None else [1] * len(self.pool_types)
        )
        assert (
            len(self.pool_types)
            == len(self.pool_kernels)
            == len(self.pool_strides)
            == len(self.pool_padding)
            == len(self.pool_dilations)
        ), "Pooling parameters must be of equal length"

        # Define dropout layers
        if dropout_rates != 0.0 and dropout_rates is not None:
            if type(dropout_rates) == float:
                self.dropout_rates = [dropout_rates] * len(conv_channels)
            elif type(dropout_rates) == list:
                self.dropout_rates = dropout_rates
        else:
            self.dropout_rates = []

        # Define batchnorm layers
        self.batchnorm = batchnorm
        self.batchnorm_first = batchnorm_first

        # Build block
        self.layers = nn.Sequential()
        for i in range(len(conv_channels)):
            # Add a convolutional layer
            if i == 0:
                self.layers.append(
                    nn.Conv1d(
                        self.input_channels,
                        self.conv_channels[0],
                        self.conv_kernels[0],
                        stride=self.conv_strides[0],
                        dilation=self.conv_dilations[0],
                        padding=self.conv_padding[0],
                    )
                )
            else:
                self.layers.append(
                    nn.Conv1d(
                        self.conv_channels[i - 1],
                        self.conv_channels[i],
                        self.conv_kernels[i],
                        stride=self.conv_strides[i],
                        dilation=self.conv_dilations[i],
                        padding=self.conv_padding[i],
                    )
                )

            # Add a batchnorm layer if specified
            if batchnorm and batchnorm_first:
                self.layers.append(nn.BatchNorm1d(self.conv_channels[i]))

            # Add an activation layer if specified
            if i < len(self.activations):
                if self.activations[i] is not None:
                    self.layers.append(self.activations[i])

            # Add a pooling layer if specified
            if i < len(self.pool_types):
                if self.pool_types[i] is not None:
                    self.layers.append(
                        self.pool_types[i](
                            kernel_size=self.pool_kernels[i],
                            stride=self.pool_strides[i],
                            padding=self.pool_padding[i],
                        )
                    )

            # Add a dropout layer if specified
            if i < len(self.dropout_rates):
                if self.dropout_rates[i] is not None:
                    self.layers.append(nn.Dropout(self.dropout_rates[i]))

            # Add a batchnorm layer if specified
            if self.batchnorm and not self.batchnorm_first:
                self.layers.append(nn.BatchNorm1d(conv_channels[i]))

        # Define output parameters
        self.out_channels = self.conv_channels[-1]
        self.output_len = get_conv1dblock_output_len(
            self.layers, input_len=self.input_len
        )
        self.flatten_dim = self.output_len * self.out_channels

    def forward(self, x):
        return self.layers(x)


class BiConv1DTower(nn.Module):
    """Generates a PyTorch module with the convolutional architecture described in:
    TODO: Add paper reference
    Will be deprecated in favor of Conv1D blocks wrapped by a Tower in the future

    Parameters
    ----------
    input_len : int
        Length of the input sequence
    filters : int
        Number of filters in the convolutional layers. Applies the same number to all layers
    kernel_size : int
        Size of the kernel in the convolutional layers. Applies the same size to all layers
    layers : int
        Number of convolutional layers
    stride : int
        Stride of the convolutional layers. Applies the same stride to all layers
    dropout_rate : float
        Dropout rate of the convolutional layers. Applies the same rate to all layers

    Returns
    -------
    torch.nn.Module
        TODO: Add description
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        input_size: int = 4,
        layers: int = 1,
        stride: int = 1,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_size = input_size
        if layers < 1:
            raise ValueError("At least one layer needed")
        self.layers = layers
        if (dropout_rate < 0) or (dropout_rate > 1):
            raise ValueError("Dropout rate must be a float between 0 and 1")
        self.dropout_rate = dropout_rate
        self.stride = stride

        self.kernels = nn.ParameterList()
        self.biases = nn.ParameterList()
        kernel = nn.Parameter(torch.zeros(self.filters, self.input_size, kernel_size))
        nn.init.xavier_uniform_(kernel)
        self.kernels.append(kernel)
        bias = nn.Parameter(torch.zeros(filters))
        nn.init.zeros_(bias)
        self.biases.append(bias)
        for layer in range(1, self.layers):
            kernel = nn.Parameter(
                torch.empty((self.filters, self.filters, self.kernel_size))
            )
            nn.init.xavier_uniform_(kernel)
            self.kernels.append(kernel)
            bias = nn.Parameter(torch.empty((self.filters)))
            nn.init.zeros_(bias)
            self.biases.append(bias)

    def forward(self, x):
        x_fwd = F.conv1d(x, self.kernels[0], stride=self.stride, padding="same")
        x_fwd = torch.add(x_fwd.transpose(1, 2), self.biases[0]).transpose(1, 2)
        x_fwd = F.dropout(F.relu(x_fwd), p=self.dropout_rate, training=self.training)
        x_rev = F.conv1d(
            x,
            torch.flip(self.kernels[0], dims=[0, 1]),
            stride=self.stride,
            padding="same",
        )
        x_rev = torch.add(x_rev.transpose(1, 2), self.biases[0]).transpose(1, 2)
        x_rev = F.dropout(F.relu(x_rev), p=self.dropout_rate, training=self.training)
        for layer in range(1, self.layers):
            x_fwd = F.conv1d(
                x_fwd, self.kernels[layer], stride=self.stride, padding="same"
            )
            x_fwd = torch.add(x_fwd.transpose(1, 2), self.biases[layer]).transpose(1, 2)
            x_fwd = F.dropout(F.relu(x_fwd), p=self.dropout_rate, training=self.training)
            x_rev = F.conv1d(
                x_rev,
                torch.flip(self.kernels[layer], dims=[0, 1]),
                stride=self.stride,
                padding="same",
            )
            x_rev = torch.add(x_rev.transpose(1, 2), self.biases[layer]).transpose(1, 2)
            x_rev = F.dropout(F.relu(x_rev), p=self.dropout_rate, training=self.training)
        return torch.add(x_fwd, x_rev)
