import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import _layers as layers
from ..base import _regularizers as regularizers
from ..base import _blocks as blocks
from ..base import _towers as towers


class DeepMEL(nn.Module):
    """DeepMEL model implementation from Minnoye et al 2020 in PyTorch

    This is a special case of a Hybrid basic model that has been used to 
    classify results from topic modeling on scATAC-seq data. It is a
    convolutional-recurrent network with a fully connected layer at the end.

    Parameters
    ----------
    input_len : int
        The length of the input sequence.
    output_dim : int
        The dimension of the output.
    conv_kwargs : dict
        The keyword arguments for the convolutional layer. These come from the
        models.Conv1DTower class. See the documentation for that class for more
        information on what arguments are available. If not specified,
        the default parameters from from Minnoye et al 2020 will be used
    recurrent_kwargs : dict
        The keyword arguments for the recurrent layer. These come from the
        models.RecurrentBlock class. See the documentation for that class for more
        information on what arguments are available. If not specified,
        the default parameters from from Minnoye et al 2020 will be used
    dense_kwargs : dict
        The keyword arguments for the fully connected layer. These come from the
        models.DenseBlock class. See the documentation for that class for more
        information on what arguments are available. If not specified,
        the default parameters from from Minnoye et al 2020 will be used
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict = {},
        recurrent_kwargs: dict = {},
        dense_kwargs: dict = {}
    ):
        super(DeepMEL, self).__init__()

        self.input_len = input_len
        self.output_dim = output_dim
        
        self.conv_kwargs, self.recurrent_kwargs, self.dense_kwargs = self.kwarg_handler(
            conv_kwargs, recurrent_kwargs, dense_kwargs
        )
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.recurrent_block = blocks.RecurrentBlock(
            input_dim=self.conv1d_tower.out_channels, 
            **self.recurrent_kwargs
        )
        self.dense_block = blocks.DenseBlock(
            input_dim=self.recurrent_block.out_channels,
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_tower(x)
        x = x.transpose(1, 2)
        out, _ = self.recurrent_block(x)
        out = self.dense_block(out[:, -1, :])
        return out

    def kwarg_handler(self, conv_kwargs, recurrent_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [128])
        conv_kwargs.setdefault("conv_kernels", [20])
        conv_kwargs.setdefault("conv_strides", [1])
        conv_kwargs.setdefault("conv_padding", "valid")
        conv_kwargs.setdefault("pool_kernels", [20])
        conv_kwargs.setdefault("dropout_rates", 0.2)
        conv_kwargs.setdefault("activations", "relu")
        conv_kwargs.setdefault("batchnorm", False)
        recurrent_kwargs.setdefault("unit_type", "lstm")
        recurrent_kwargs.setdefault("hidden_dim", 128)
        #recurrent_kwargs.setdefault("dropout_rates", 0.2)
        recurrent_kwargs.setdefault("bidirectional", True)
        recurrent_kwargs.setdefault("batch_first", True)
        dense_kwargs.setdefault("hidden_dims", [256])
        dense_kwargs.setdefault("dropout_rates", 0.4)
        dense_kwargs.setdefault("batchnorm", False)
        return conv_kwargs, recurrent_kwargs, dense_kwargs


class scBasset(nn.Module):
    """scBasset model implementation from Yuan et al 2022 in PyTorch

    This model has not been fully tested yet.

    Parameters
    ----------
    num_cells : int
        The number of cells in the dataset.
    num_batches : int
        The number of batches in the dataset. If not specified, the model will
        not include batch correction.
    l1 : float
        The L1 regularization parameter for the cell layer.
    l2 : float
        The L2 regularization parameter for the batch layer.
    """
    def __init__(self, num_cells, num_batches=None, l1=0.0, l2=0.0):
        super(scBasset, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=288, kernel_size=17, padding=8),
            nn.BatchNorm1d(288),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.conv_tower = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(in_channels=288, out_channels=288, kernel_size=5, padding=2),
                nn.BatchNorm1d(288),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=288, out_channels=323, kernel_size=5, padding=2),
                nn.BatchNorm1d(323),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=323, out_channels=363, kernel_size=5, padding=2),
                nn.BatchNorm1d(363),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=363, out_channels=407, kernel_size=5, padding=2),
                nn.BatchNorm1d(407),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=407, out_channels=456, kernel_size=5, padding=2),
                nn.BatchNorm1d(456),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=456, out_channels=512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.flatten = nn.Flatten()

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=1792, out_features=32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.GELU(),
        )

        self.fc1 = regularizers.L2(
            nn.Linear(in_features=32, out_features=num_cells), weight_decay=l1
        )
        self.sigmoid = nn.Sigmoid()

        if num_batches is not None:
            self.fc2 = regularizers.L2(
                nn.Linear(in_features=32, out_features=num_batches), weight_decay=l2
            )

        self.l1 = l1
        self.l2 = l2

    def forward(self, x, batch=None):
        x = self.conv1(x)
        x = self.conv_tower(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.bottleneck(x)
        if batch is not None:
            x_batch = self.fc2(x)
            x_batch = torch.matmul(x_batch, batch)
            x = self.fc1(x) + x_batch
        else:
            x = self.fc1(x)

        x = self.sigmoid(x)

        return x
