import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, BiConv1D


class Jores21CNN(BaseModel):
    """
    Jores21CNN model. Uses a custom BiConv1D module converted from tf implementation to PyTorch from
    url (which was adapted from url).

    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    strand : str, optional
        Strand of the input. Either "ss" or "ds".
    task : str, optional
        Task of the model. Either "regression" or "classification".
    aggr : str, optional
        Aggregation method. Either "mean" or "sum".
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

    Returns
    -------
    model : torch.nn.Module
        JORES-21 model.
    """

    def __init__(
        self,
        input_len,
        output_dim,
        strand="ss",
        task="regression",
        aggr=None,
        filters: int = 128,
        kernel_size: int = 13,
        layers: int = 2,
        stride: int = 1,
        dropout: float = 0.15,
        hidden_dim: int = 64,
        **kwargs
    ):
        super().__init__(
            input_len, output_dim, strand=strand, task=task, aggr=aggr, **kwargs
        )
        self.biconv = BiConv1D(
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

    def forward(self, x, x_rev=None):
        x = self.biconv(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Kopp21CNN(BaseModel):
    def __init__(
        self,
        input_len,
        output_dim,
        strand="ds",
        task="binary_classification",
        aggr="max",
        loss_fxn="bce",
        filters: list = [10, 8],
        conv_kernel_size: list = [11, 3],
        maxpool_kernel_size: int = 30,
        stride: int = 1,
        **kwargs
    ):
        super().__init__(
            input_len,
            output_dim,
            strand=strand,
            task=task,
            aggr=aggr,
            loss_fxn=loss_fxn,
            **kwargs
        )
        self.conv = nn.Conv1d(4, filters[0], conv_kernel_size[0], stride=stride)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm1d(filters[0])
        self.conv2 = nn.Conv1d(
            filters[0], filters[1], conv_kernel_size[1], stride=stride
        )
        self.batchnorm2 = nn.BatchNorm1d(filters[1])
        self.linear = nn.Linear(filters[1], self.output_dim)

    def forward(self, x, x_rev):
        x_fwd = F.relu(self.conv(x))
        x_rev = F.relu(self.conv(x_rev))
        if self.aggr == "concat":
            x = torch.cat((x_fwd, x_rev), dim=1)
        elif self.aggr == "max":
            x = torch.max(x_fwd, x_rev)
        elif self.aggr == "ave":
            x = (x_fwd + x_rev) / 2
        elif self.aggr is None:
            x = torch.cat((x_fwd, x_rev), dim=1)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])
        x = self.batchnorm2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
