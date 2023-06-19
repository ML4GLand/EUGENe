import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import _layers as layers
from ..base import _blocks as blocks
from ..base import _towers as towers


class Jores21CNN(nn.Module):
    """
    Custom convolutional model used in Jores et al. 2021 paper

    PyTorch implementation of the TensorFlow model described here:
    https://github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters

    This model only uses a single strand, but applies convolutions and the reverse complement of the convolutional fitler
    to the same sequence.

    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    strand : str, optional
        Strand of the input. Only ss is supported for this model
    task : str, optional
        Task of the model. Either "regression" or "classification".
    aggr : str, optional
        Aggregation method. Does not apply to this model and will be ignored
    filters : int, optional
        Number of filters in the convolutional layers.
    kernel_size : int, optional
        Kernel size of the convolutional layers.
    layers : int, optional
        Number of convolutional layers.
    stride : int, optional
        Stride of the convolutional layers.
    dropout : float, optional
        Dropout probability.
    hidden_dim : int, optional
        Dimension of the hidden layer.
    """

    def __init__(
        self,
        input_len: int,
        output_dim: int,
        filters: int = 128,
        kernel_size: int = 13,
        layers: int = 2,
        stride: int = 1,
        dropout: float = 0.15,
        hidden_dim: int = 64
    ):
        super(Jores21CNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.layers = layers
        self.stride = stride
        self.dropout = dropout

        # Create the blocks
        self.biconv = towers.BiConv1DTower(
            filters=filters,
            kernel_size=kernel_size,
            layers=layers,
            stride=stride,
            dropout_rate=dropout,
        )
        self.conv = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=input_len * filters, out_features=hidden_dim)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.biconv(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DeepSTARR(nn.Module):
    """DeepSTARR model from de Almeida et al., 2022; see <https://www.nature.com/articles/s41588-022-01048-5>

    Parameters
    """

    def __init__(
        self, input_len: int, output_dim: int, conv_kwargs={}, dense_kwargs={}
    ):
        super(DeepSTARR, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(
            conv_kwargs, dense_kwargs
        )

        # Create the blocks
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.dense_block = blocks.DenseBlock(
            input_dim=self.conv1d_tower.flatten_dim,
            output_dim=output_dim,
            **self.dense_kwargs
        )

    def forward(self, x):
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)
        x = self.dense_block(x)
        return x

    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [246, 60, 60, 120])
        conv_kwargs.setdefault("conv_kernels", [7, 3, 5, 3])
        conv_kwargs.setdefault("conv_strides", [1, 1, 1, 1])
        conv_kwargs.setdefault("conv_padding", "same")
        conv_kwargs.setdefault("pool_kernels", [2, 2, 2, 2])
        conv_kwargs.setdefault("dropout_rates", 0.0)
        conv_kwargs.setdefault("batchnorm", True)
        conv_kwargs.setdefault("batchnorm_first", True)
        dense_kwargs.setdefault("hidden_dims", [256, 256])
        dense_kwargs.setdefault("dropout_rates", 0.4)
        dense_kwargs.setdefault("batchnorm", True)
        dense_kwargs.setdefault("batchnorm_first", True)
        return conv_kwargs, dense_kwargs
