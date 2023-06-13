import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import _layers as layers
from ..base import _blocks as blocks
from ..base import _towers as towers


class FCN(nn.Module):
    """
    Instantiate a fully connected neural network with the specified layers and parameters.

    By default, this architecture flattens the one-hot encoded sequence and passes
    it through a set of layers that are fully connected. The task defines how the output is
    treated (e.g. sigmoid activation for binary classification). The loss function is
    should be matched to the task (e.g. binary cross entropy ("bce") for binary classification).

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
        input_dims: int = 4,
        dense_kwargs: dict = {},
    ):
        super(FCN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.dense_kwargs = dense_kwargs
        self.flattened_input_dims = input_len * input_dims

        # Create the blocks
        self.dense_block = blocks.DenseBlock(
            input_dim=self.flattened_input_dims, output_dim=output_dim, **dense_kwargs
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dense_block(x)
        return x


class CNN(nn.Module):
    """
    Instantiate a CNN model with a set of convolutional layers and a set of fully
    connected layers.

    By default, this architecture passes the one-hot encoded sequence through a set
    1D convolutions with 4 channels. The task defines how the output is treated (e.g.
    sigmoid activation for binary classification). The loss function is should be matched
    to the task (e.g. binary cross entropy ("bce") for binary classification).

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    task:
        The task of the model.
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the flattened output of the convolutional layers directly to
        the output layer.
    """

    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        dense_kwargs: dict = {},
    ):
        super(CNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs = conv_kwargs

        # Create the blocks
        self.conv1d_tower = towers.Conv1DTower(input_len=input_len, **conv_kwargs)
        self.dense_block = blocks.DenseBlock(
            input_dim=self.conv1d_tower.flatten_dim,
            output_dim=output_dim,
            **dense_kwargs
        )

    def forward(self, x):
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)
        x = self.dense_block(x)
        return x


class RNN(nn.Module):
    """
    Instantiate an RNN model with a set of recurrent layers and a set of fully
    connected layers.

    By default, this model passes the one-hot encoded sequence through recurrent layers
    and then through a set of fully connected layers. The output of the fully connected
    layers is passed to the output layer.

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    task:
        The task of the model.
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the recurrent output of the recurrent layers directly to the
        output layer.
    """

    def __init__(
        self,
        input_len: int,
        output_dim: int,
        recurrent_kwargs: dict,
        input_dims: int = 4,
        dense_kwargs: dict = {},
    ):
        super(RNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.recurrent_kwargs = recurrent_kwargs
        self.dense_kwargs = dense_kwargs

        # Create the blocks
        self.recurrent_block = blocks.RecurrentBlock(
            input_dim=input_dims, **recurrent_kwargs
        )
        self.dense_block = blocks.DenseBlock(
            input_dim=self.recurrent_block.out_channels,
            output_dim=output_dim,
            **dense_kwargs
        )

    def forward(self, x):
        x, _ = self.recurrent_block(x)
        x = x[:, -1, :]
        x = self.dense_block(x)
        return x


class Hybrid(nn.Module):
    """
    A hybrid model that uses both a CNN and an RNN to extract features then passes the
    features through a set of fully connected layers.

    By default, the CNN is used to extract features from the input sequence, and the RNN is used to
    to combine those features. The output of the RNN is passed to a set of fully connected
    layers to make the final prediction.

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
        conv_kwargs: dict,
        recurrent_kwargs: dict,
        dense_kwargs: dict = {},
    ):
        super(Hybrid, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs = conv_kwargs
        self.recurrent_kwargs = recurrent_kwargs
        self.dense_kwargs = dense_kwargs

        # Create the blocks
        self.conv1d_tower = towers.Conv1DTower(input_len=input_len, **conv_kwargs)
        self.recurrent_block = blocks.RecurrentBlock(
            input_dim=self.conv1d_tower.out_channels, **recurrent_kwargs
        )
        self.dense_block = blocks.DenseBlock(
            input_dim=self.recurrent_block.out_channels,
            output_dim=output_dim,
            **dense_kwargs
        )

    def forward(self, x):
        x = self.conv1d_tower(x)
        x = x.transpose(1, 2)
        out, _ = self.recurrent_block(x)
        out = self.dense_block(out[:, -1, :])
        return out


class TutorialCNN(nn.Module):
    """Tutorial CNN model

    This is a very simple one layer convolutional model for testing purposes. It is featured in testing and tutorial
    notebooks.

    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    task : str, optional
        Task of the model. Either "regression" or "classification".
    loss_fxn : str, optional
        Loss function.
    **kwargs
        Keyword arguments to pass to the nn.Module class.
    """

    def __init__(
        self,
        input_len: int,
        output_dim: int,
    ):
        super(TutorialCNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim

        # Create the blocks
        self.conv1 = nn.Conv1d(4, 30, 21)
        self.dense = nn.Linear(30, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, x.size()[-1]).flatten(1, -1)
        x = self.dense(x)
        return x


class Inception(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        channels=[4, 64, 128, 256],
        kernel_size2: int = 4,
        kernel_size3: int = 8,
        conv_maxpool_kernel_size: int = 3,
        dense_kwargs: dict = {},
    ):
        super(Inception, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.channels = channels
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.conv_maxpool_kernel_size = conv_maxpool_kernel_size
        self.dense_kwargs = dense_kwargs

        # Create the blocks
        conv_tower = nn.Sequential()
        for i in range(1, len(self.channels)):
            conv_tower.append(
                layers.InceptionConv1D(
                    in_channels=self.channels[i - 1],
                    out_channels=self.channels[i],
                    kernel_size2=self.kernel_size2,
                    kernel_size3=self.kernel_size3,
                    conv_maxpool_kernel_size=self.conv_maxpool_kernel_size,
                )
            )
        self.conv_tower = nn.Sequential(*conv_tower)
        self.dense_block = blocks.DenseBlock(
            input_dim=self.channels[-1] * self.input_len,
            output_dim=self.output_dim,
            **self.dense_kwargs
        )

    def forward(self, x):
        x = self.conv_tower(x)
        x = x.view(x.shape[0], -1)
        x = self.dense_block(x)
        return x
