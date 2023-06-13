import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import _layers as layers
from ..base import _blocks as blocks
from ..base import _towers as towers


class DeepSEA(nn.Module):
    """DeepSEA model implementation for EUGENe

    Default parameters are those specified in the DeepSEA paper. We currently do not implement a "ds" or "ts" model
    for DeepSEA.

    Parameters
    ----------
    input_len:
        int, input sequence length
    channels:
        list-like or int, channel width for each conv layer. If int each of the three layers will be the same channel width
    conv_kernels:
        list-like or int, conv kernel size for each conv layer. If int will be the same for all conv layers
    pool_kernels:
        list-like or int, maxpooling kernel size for the first two conv layers. If int will be the same for all conv layers
    dropout_rates:
        list-like or float, dropout rates for each conv layer. If int will be the same for all conv layers
    """

    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict = {},
        dense_kwargs: dict = {},
    ):
        super(DeepSEA, self).__init__()

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
        conv_kwargs.setdefault("conv_channels", [320, 480, 960])
        conv_kwargs.setdefault("conv_kernels", [8, 8, 8])
        conv_kwargs.setdefault("pool_types", ["max", "max", None])
        conv_kwargs.setdefault("pool_kernels", [4, 4, None])
        conv_kwargs.setdefault("activations", "relu")
        conv_kwargs.setdefault("dropout_rates", [0.2, 0.2, 0.5])
        conv_kwargs.setdefault("batchnorm", False)
        dense_kwargs.setdefault("hidden_dims", [925])
        return conv_kwargs, dense_kwargs


class Basset(nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs={},
        dense_kwargs={},
    ):
        super(Basset, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(
            conv_kwargs, dense_kwargs
        )
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
        conv_kwargs.setdefault("conv_channels", [300, 200, 200])
        conv_kwargs.setdefault("conv_kernels", [19, 11, 7])
        conv_kwargs.setdefault("conv_strides", [1, 1, 1])
        conv_kwargs.setdefault("conv_padding", [9, 5, 3])
        conv_kwargs.setdefault("pool_kernels", [3, 4, 4])
        conv_kwargs.setdefault("dropout_rates", 0.0)
        conv_kwargs.setdefault("activations", "relu")
        conv_kwargs.setdefault("batchnorm", True)
        conv_kwargs.setdefault("batchnorm_first", True)
        dense_kwargs.setdefault("hidden_dims", [1000, 164])
        dense_kwargs.setdefault("dropout_rates", 0.0)
        dense_kwargs.setdefault("batchnorm", True)
        dense_kwargs.setdefault("batchnorm_first", True)
        dense_kwargs.setdefault("activations", "relu")
        return conv_kwargs, dense_kwargs


class FactorizedBasset(nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        input_len: int = 1000,
        output_dim=1,
        conv1_kwargs={},
        conv2_kwargs={},
        conv3_kwargs={},
        maxpool_kernels=None,
        dense_kwargs={},
    ):
        super(FactorizedBasset, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        (
            self.conv1_kwargs,
            self.conv2_kwargs,
            self.conv3_kwargs,
            self.maxpool_kernels,
            self.dense_kwargs,
        ) = self.kwarg_handler(
            conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels, dense_kwargs
        )

        # Create the blocks
        self.conv1d_tower1 = towers.Conv1DTower(
            input_len=input_len, input_channels=4, **self.conv1_kwargs
        )
        self.conv1d_tower2 = towers.Conv1DTower(
            input_len=self.conv1d_tower1.output_len,
            input_channels=self.conv1d_tower1.out_channels,
            **self.conv2_kwargs
        )
        self.conv1d_tower3 = towers.Conv1DTower(
            input_len=self.conv1d_tower2.output_len,
            input_channels=self.conv1d_tower2.out_channels,
            **self.conv3_kwargs
        )
        self.dense_block = blocks.DenseBlock(
            input_dim=self.conv1d_tower3.flatten_dim,
            output_dim=output_dim,
            **self.dense_kwargs
        )

    def forward(self, x):
        x = self.conv1d_tower1(x)
        x = self.conv1d_tower2(x)
        x = self.conv1d_tower3(x)
        x = x.view(x.size(0), self.conv1d_tower3.flatten_dim)
        x = self.dense_block(x)
        return x

    def kwarg_handler(
        self, conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels, dense_kwargs
    ):
        """Sets default kwargs FactorizedBasset"""
        conv1_kwargs.setdefault("conv_channels", [48, 64, 100, 150, 300])
        conv1_kwargs.setdefault("conv_kernels", [3, 3, 3, 7, 7])
        conv1_kwargs.setdefault("conv_strides", [1, 1, 1, 1, 1])
        conv1_kwargs.setdefault("conv_padding", [1, 1, 1, 3, 3])
        conv1_kwargs.setdefault("pool_types", [None, None, None, None, "max"])
        conv1_kwargs.setdefault("pool_kernels", [None, None, None, None, 3])
        conv1_kwargs.setdefault("dropout_rates", 0.0)
        conv1_kwargs.setdefault("batchnorm", True)
        conv1_kwargs.setdefault("batchnorm_first", True)
        conv1_kwargs.setdefault("activations", "relu")
        conv2_kwargs.setdefault("conv_channels", [200, 200, 200])
        conv2_kwargs.setdefault("conv_kernels", [7, 3, 3])
        conv2_kwargs.setdefault("conv_strides", [1, 1, 1])
        conv2_kwargs.setdefault("conv_padding", [3, 1, 1])
        conv2_kwargs.setdefault("pool_types", [None, None, "max"])
        conv2_kwargs.setdefault("pool_kernels", [None, None, 4])
        conv2_kwargs.setdefault("dropout_rates", 0.0)
        conv2_kwargs.setdefault("batchnorm", True)
        conv2_kwargs.setdefault("batchnorm_first", True)
        conv2_kwargs.setdefault("activations", "relu")
        conv3_kwargs.setdefault("conv_channels", [200])
        conv3_kwargs.setdefault("conv_kernels", [7])
        conv3_kwargs.setdefault("conv_strides", [1])
        conv3_kwargs.setdefault("conv_padding", [3])
        conv3_kwargs.setdefault("pool_types", ["max"])
        conv3_kwargs.setdefault("pool_kernels", [4])
        conv3_kwargs.setdefault("dropout_rates", 0.0)
        conv3_kwargs.setdefault("batchnorm", True)
        conv3_kwargs.setdefault("batchnorm_first", True)
        conv3_kwargs.setdefault("activations", "relu")
        dense_kwargs.setdefault("hidden_dims", [1000, 164])
        dense_kwargs.setdefault("dropout_rates", 0.0)
        dense_kwargs.setdefault("batchnorm", True)
        dense_kwargs.setdefault("batchnorm_first", True)
        dense_kwargs.setdefault("activations", "relu")
        return conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels, dense_kwargs


class DanQ(nn.Module):
    """DanQ model from Quang and Xie, 2016;

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    task:
        The task of the model.
    dense_kwargs:
        The keyword arguments for the fully connected layer.
    """

    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict = {},
        recurrent_kwargs: dict = {},
        dense_kwargs: dict = {},
    ):
        super(DanQ, self).__init__()

        # Set the attrubutes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs, self.recurrent_kwargs, self.dense_kwargs = self.kwarg_handler(
            conv_kwargs, recurrent_kwargs, dense_kwargs
        )

        # Build the model
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.recurrent_block = blocks.RecurrentBlock(
            input_dim=self.conv1d_tower.out_channels, **self.recurrent_kwargs
        )
        self.dense_block = blocks.DenseBlock(
            input_dim=self.recurrent_block.out_channels,
            output_dim=output_dim,
            **self.dense_kwargs
        )

    def forward(self, x):
        x = self.conv1d_tower(x)
        x = x.transpose(1, 2)
        out, _ = self.recurrent_block(x)
        out = self.dense_block(out[:, -1, :])
        return out

    def kwarg_handler(self, conv_kwargs, recurrent_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [320])
        conv_kwargs.setdefault("conv_kernels", [26])
        conv_kwargs.setdefault("conv_strides", [1])
        conv_kwargs.setdefault("conv_padding", "same")
        conv_kwargs.setdefault("pool_kernels", [13])
        conv_kwargs.setdefault("dropout_rates", 0.2)
        conv_kwargs.setdefault("activations", "relu")
        recurrent_kwargs.setdefault("unit_type", "lstm")
        recurrent_kwargs.setdefault("hidden_dim", 320)
        recurrent_kwargs.setdefault("bidirectional", True)
        recurrent_kwargs.setdefault("batch_first", True)
        dense_kwargs.setdefault("hidden_dims", [925])
        dense_kwargs.setdefault("dropout_rates", 0.5)
        dense_kwargs.setdefault("batchnorm", False)
        return conv_kwargs, recurrent_kwargs, dense_kwargs


class Satori(nn.Module):
    def __init__(
        self,
        input_len,
        output_dim,
        conv_kwargs: dict = {},
        mha_kwargs: dict = {},
        dense_kwargs: dict = {},
    ):
        super(Satori, self).__init__()

        # Set the attrubutes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs, self.mha_kwargs, self.dense_kwargs = self.kwarg_handler(
            conv_kwargs, mha_kwargs, dense_kwargs
        )

        # Build the model
        self.conv_block = blocks.Conv1DBlock(**self.conv_kwargs)
        self.mha_layer = layers.MultiHeadAttention(
            input_dim=self.conv_block.output_size[-1], **self.mha_kwargs
        )
        self.flatten = nn.Flatten()
        self.dense_block = blocks.DenseBlock(
            input_dim=self.conv_block.output_channels * self.conv_block.output_size[-1],
            **self.dense_kwargs
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.mha_layer(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        return x

    def kwarg_handler(self, conv_kwargs, mha_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("output_channels", 320)
        conv_kwargs.setdefault("conv_kernel", 26)
        conv_kwargs.setdefault("conv_padding", "same")
        conv_kwargs.setdefault("norm_type", "batchnorm")
        conv_kwargs.setdefault("activation", "relu")
        conv_kwargs.setdefault("conv_bias", False)
        conv_kwargs.setdefault("pool_type", "max")
        conv_kwargs.setdefault("pool_kernel", 3)
        conv_kwargs.setdefault("pool_padding", 1)
        conv_kwargs.setdefault("dropout_rate", 0.2)
        conv_kwargs.setdefault("order", "conv-norm-act-pool-dropout")
        mha_kwargs.setdefault("head_dim", 64)
        mha_kwargs.setdefault("num_heads", 8)
        mha_kwargs.setdefault("dropout_rate", 0.1)
        dense_kwargs.setdefault("hidden_dims", [])
        dense_kwargs.setdefault("output_dim", self.output_dim)
        return conv_kwargs, mha_kwargs, dense_kwargs
