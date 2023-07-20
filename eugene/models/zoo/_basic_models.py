import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import _layers as layers
from ..base import _blocks as blocks
from ..base import _towers as towers
from typing import Union, Callable, Optional, Any, Tuple, List


class FCN(nn.Module):
    """Basic fully connected network

    Instantiate a fully connected neural network with the specified layers and parameters.

    By default, this architecture flattens the one-hot encoded sequence and passes
    it through a set of layers that are fully connected.

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    input_dims:
        The number of input dimensions. This is equivalent to the alphabet length.
    dense_kwargs:
        The keyword arguments for the fully connected layer. These come from
        the models.DenseBlock class. See the documentation for that class for
        more information on what arguments are available.
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
            input_dim=self.flattened_input_dims, 
            output_dim=output_dim, 
            **dense_kwargs
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dense_block(x)
        return x


class CNN(nn.Module):
    """Basic convolutional network

    Instantiate a CNN model with a set of convolutional layers and a set of fully
    connected layers.

    By default, this architecture passes the one-hot encoded sequence through a set
    1D convolutions with 4 channels followed by a set of fully connected layers.

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    conv_kwargs:
        The keyword arguments for the convolutional layers. These come from the
        models.Conv1DTower class. See the documentation for that class for more
        information on what arguments are available.
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the flattened output of the convolutional layers directly to
        the output layer. These come from the models.DenseBlock class. See the
        documentation for that class for more information on what arguments are
        available.
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
        self.conv1d_tower = towers.Conv1DTower(
            input_len=input_len, 
            **conv_kwargs
        )
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
    """Basic recurrent network

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
    recurrent_kwargs:
        The keyword arguments for the recurrent layers. These come from the
        models.RecurrentBlock class. See the documentation for that class for more
        information on what arguments are available.
    input_dims:
        The number of input dimensions. This is equivalent to the alphabet length.
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the recurrent output of the recurrent layers directly to the
        output layer. These come from the models.DenseBlock class. See the
        documentation for that class for more information on what arguments are
        available.
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
    """Basic hybrid network

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
    conv_kwargs:
        The keyword arguments for the convolutional layers. These come from the
        models.Conv1DTower class. See the documentation for that class for more
        information on what arguments are available.
    recurrent_kwargs:
        The keyword arguments for the recurrent layers. These come from the
        models.RecurrentBlock class. See the documentation for that class for more
        information on what arguments are available.
    dense_kwargs:
        The keyword arguments for the fully connected layer. These come from
        the models.DenseBlock class. See the documentation for that class for
        more information on what arguments are available.
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
        self.conv1d_tower = towers.Conv1DTower(
            input_len=input_len,
            **conv_kwargs
        )
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

    This is a very simple one layer convolutional model for testing purposes. It has 4 input channels,
    a single convolutional layer of width 21 and with 30 output channels, and a single dense layer with
    30 hidden units. This model is featured in testing and tutorial notebooks.

    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
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
    """Inecption1DConv network

    Instantiate a CNN model with a set of convolutional layers and a set of fully connected layers in
    the same way as CNN, but the convolutional layers are replaced with InceptionConv1D layers. These layers
    follow from the 2D inception layers in the original Inception architecture, but have been modified for 1D

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    channels:
        The number of channels in each layer of the InceptionConv1D tower.
    kernel_size2:
        The kernel for the first non 1 unit convolution in each layer
    kernel_size3:
        The kernel for the second non 1 unit convolution in each layer
    conv_maxpool_kernel_size:
        The kernel size for the maxpooling layer in each layer
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the flattened output of the convolutional layers directly to
        the output layer. These come from the models.DenseBlock class. See the
        documentation for that class for more information on what arguments are
        available.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        channels: List[int] = [4, 32, 64, 128],
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

class dsFCN(nn.Module):
    """Basic FCN model with reverse complement

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
    input_dims:
        The number of input dimensions. This is equivalent to the alphabet length.
    aggr:
        The method for aggregating the output of the forward pass of the reverse complement.
        If "concat", the output of the forward pass and the reverse complement pass are
        concatenated and passed to the dense block. If "max" or "avg", the output of the
        forward pass and the reverse complement pass are passed to the dense block and
        the max or average of the two outputs is returned.
    dense_kwargs:
        The keyword arguments for the fully connected layer. These come from
        the models.DenseBlock class. See the documentation for that class for
        more information on what arguments are available.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        input_dims: int = 4,
        aggr: str = "concat",
        dense_kwargs: dict = {}
    ):
        super(dsFCN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.dense_kwargs = dense_kwargs
        self.aggr = aggr
        self.flattened_input_dims = input_len * input_dims

        # Creat the revcomp layer
        self.revcomp = layers.RevComp()

        # Create the block
        if aggr == "concat":
            self.dense_block = blocks.DenseBlock(
                input_dim=self.flattened_input_dims * 2,
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif aggr in ["max", "avg"]:
            self.dense_block = blocks.DenseBlock(
                input_dim=self.flattened_input_dims,
                output_dim=output_dim, 
                **dense_kwargs
            )
        else:
            raise ValueError(f"Invalid aggr, must be one of ['concat', 'max', 'avg'], got {aggr}.")

    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        x = x.flatten(start_dim=1)
        x_rev_comp = x_rev_comp.flatten(start_dim=1)
        if self.aggr == "concat":
            x = torch.cat([x, x_rev_comp], dim=1)
            x = self.dense_block(x)
        elif self.aggr in ["max", "avg"]:
            x = self.dense_block(x)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = (x + x_rev_comp) / 2
        return x
    

class dsCNN(nn.Module):
    """Basic CNN model with reverse complement

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
    conv_kwargs:
        The keyword arguments for the convolutional layers. These come from the
        models.Conv1DTower class. See the documentation for that class for more
        information on what arguments are available.
    aggr:
        The method for aggregating the output of the forward pass of the reverse complement.
        If "concat", the output of the forward pass and the reverse complement pass are
        concatenated and passed to the dense block. If "max" or "avg", the output of the
        forward pass and the reverse complement pass are passed to the dense block and
        the max or average of the two outputs is returned.
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the flattened output of the convolutional layers directly to 
        the output layer. These come from the models.DenseBlock class. See the
        documentation for that class for more information on what arguments are
        available.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        aggr: str = "concat",
        dense_kwargs: dict = {},
    ):
        super(dsCNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs = conv_kwargs
        self.aggr = aggr

        # Create the revcomp layer
        self.revcomp = layers.RevComp()
        
        # Create the conv1d tower
        self.conv1d_tower = towers.Conv1DTower(
            input_len=input_len,
            **conv_kwargs
        )
        
        # Get the dimension of the output of the conv1d tower
        if aggr == "concat":
            self.dense_dim = self.conv1d_tower.flatten_dim * 2
        elif aggr in ["max", "avg"]:
            self.dense_dim = self.conv1d_tower.flatten_dim
        else:
            raise ValueError("aggr must be one of ['concat', 'max', 'avg']")
        
        # Create the dense block
        self.dense_block = blocks.DenseBlock(
            input_dim=self.dense_dim, 
            output_dim=output_dim, 
            **dense_kwargs
        )

    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)

        x_rev_comp = self.conv1d_tower(x_rev_comp)
        x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_tower.flatten_dim)

        if self.aggr == "concat":
            x = torch.cat((x, x_rev_comp), dim=1)
            x = self.dense_block(x)
        elif self.aggr in ["max", "avg"]:
            x = self.dense_block(x)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = (x + x_rev_comp) / 2
        return x
    

class dsRNN(nn.Module):
    """Basic RNN model with reverse complement

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
    recurrent_kwargs:
        The keyword arguments for the recurrent layers. These come from the
        models.RecurrentBlock class. See the documentation for that class for more
        information on what arguments are available.
    input_dims:
        The number of input dimensions. This is equivalent to the alphabet length. You
        do not need to specify this argument in the recurrent_kwargs dictionary.
    aggr:
        The method for aggregating the output of the forward pass of the reverse complement.
        If "concat", the output of the forward pass and the reverse complement pass are
        concatenated and passed to the dense block. If "max" or "avg", the output of the
        forward pass and the reverse complement pass are passed to the dense block and
        the max or average of the two outputs is returned.
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
        aggr: str = "concat",
        input_dims: int = 4,
        dense_kwargs: dict = {},
    ): 
        super(dsRNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.recurrent_kwargs = recurrent_kwargs
        self.dense_kwargs = dense_kwargs

        # Create the revcomp layer
        self.revcomp = layers.RevComp()

        # Create the recurrent block
        self.recurrent_block = blocks.RecurrentBlock(
            input_dim=input_dims,
            **recurrent_kwargs
        )

        # Create the dense block
        if aggr == "concat":
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels * 2,
                output_dim=output_dim, 
                **dense_kwargs
        )
        elif aggr in ["max", "avg"]:
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        
    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        x, _ = self.recurrent_block(x)
        x = x[:, -1, :]
        x_rev_comp, _ = self.recurrent_block(x_rev_comp)
        x_rev_comp = x_rev_comp[:, -1, :]
        if self.aggr == "concat":
            x = torch.cat((x, x_rev_comp), dim=1)
            x = self.dense_block(x)
        elif self.aggr in ["max", "avg"]:
            x = self.dense_block(x)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = (x + x_rev_comp) / 2
        return x


class dsHybrid(nn.Module):
    """Basic hybrid network with reverse complement
    
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
    conv_kwargs:
        The keyword arguments for the convolutional layers. These come from the
        models.Conv1DTower class. See the documentation for that class for more\
        information on what arguments are available.
    recurrent_kwargs:
        The keyword arguments for the recurrent layers. These come from the
        models.RecurrentBlock class. See the documentation for that class for more
        information on what arguments are available.
    aggr:
        The method for aggregating the output of the forward pass of the reverse complement.
        If "concat_cnn", the output of the forward pass and the reverse complement pass are
        concatenated and passed to the recurrent block. If "concat_rnn", the output of the
        recurrent block is concatenated with the output of the reverse complement recurrent
        block and passed to the dense block. If "max" or "avg", the output of the forward
        pass and the reverse complement pass are passed to the dense block and the max
        or average of the two outputs is returned.
    dense_kwargs:
        The keyword arguments for the fully connected layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        recurrent_kwargs: dict,
        aggr="concat_cnn",
        dense_kwargs: dict = {},
    ):
        super(dsHybrid, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs = conv_kwargs
        self.recurrent_kwargs = recurrent_kwargs
        self.dense_kwargs = dense_kwargs
        self.aggr = aggr

        # Create the revcomp layer
        self.revcomp = layers.RevComp()

        # Create the conv1d tower
        self.conv1d_tower = towers.Conv1DTower(
            input_len=input_len,
            **conv_kwargs
        )

        # Create the recurrent block and dense block
        if aggr == "concat_cnn":
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=self.conv1d_tower.out_channels * 2, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
        )
        elif aggr == "concat_rnn":
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=self.conv1d_tower.out_channels, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels*2, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif aggr in ["max", "avg"]:
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=self.conv1d_tower.out_channels, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        else:
            raise ValueError("aggr must be one of ['concat_cnn', 'concat_rnn', 'max', 'avg']")

    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        
        x = self.conv1d_tower(x)
        x = x.transpose(1, 2)

        x_rev_comp = self.conv1d_tower(x_rev_comp)
        x_rev_comp = x_rev_comp.transpose(1, 2)

        if self.aggr == "concat_cnn":
            x = torch.cat((x, x_rev_comp), dim=2)
            out, _ = self.recurrent_block(x)
            out = self.dense_block(out[:, -1, :])
        elif self.aggr == "concat_rnn":
            out, _ = self.recurrent_block(x)
            out_rev_comp, _ = self.recurrent_block(x_rev_comp)
            out = torch.cat((out[:, -1, :], out_rev_comp[:, -1, :]), dim=1)
            out = self.dense_block(out)
        elif self.aggr in ["max", "avg"]:
            out_, _ = self.recurrent_block(x)
            out = self.dense_block(out_[:, -1, :])
            out_rev_comp, _ = self.recurrent_block(x_rev_comp)
            out_rev_comp = self.dense_block(out_rev_comp[:, -1, :])
            if self.aggr == "max":
                out = torch.max(out, out_rev_comp)
            elif self.aggr == "avg":
                out = (out + out_rev_comp) / 2
        return out
    