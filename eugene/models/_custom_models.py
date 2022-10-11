import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, BiConv1D


class TutorialCNN(BaseModel):
    """Tutorial CNN model

    This is a very simple one layer convolutional model for testing purposes. It is featured in testing and tutorial
    notebooks.

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
        Aggregation method. This model only supports "avg"
    loss_fxn : str, optional
        Loss function.
    **kwargs
        Keyword arguments to pass to the BaseModel class.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = "avg",
        loss_fxn: str = "mse",
        **kwargs
    ):
        # Don't worry that we don't pass in the class name to the super call (as is standard for creating new
        # nn.Module subclasses). This is handled by inherting BaseModel
        super().__init__(
            input_len, 
            output_dim, 
            strand=strand, 
            task=task, 
            aggr=aggr, 
            loss_fxn=loss_fxn,
            **kwargs
        )
        self.conv1 = nn.Conv1d(4, 30, 21)
        self.dense = nn.Linear(30, output_dim)
        self.sigmoid = nn.Sigmoid()
            
            
    def forward(self, x, x_rev_comp=None):
        x = F.relu(self.conv1(x))
        
        # emulates global_max_pooling
        x = F.max_pool1d(x, x.size()[-1]).flatten(1, -1)
        x = self.dense(x)
        x = self.sigmoid(x)
        if self.strand == "ds":
            x_rev_comp = F.relu(self.conv1(x_rev_comp))
            x_rev_comp = F.max_pool1d(x_rev_comp, x_rev_comp.size()[-1]).flatten(1, -1)
            x_rev_comp = self.dense(x_rev_comp)
            x_rev_comp = self.sigmoid(x_rev_comp)
            x = (x + x_rev_comp / 2)
        return x


class Jores21CNN(BaseModel):
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
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        filters: int = 128,
        kernel_size: int = 13,
        layers: int = 2,
        stride: int = 1,
        dropout: float = 0.15,
        hidden_dim: int = 64,
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

    def forward(self, x, x_rev_comp=None):
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
    """
    Custom convolutional model used in Kopp et al. 2021 paper

    PyTorch implementation of the TensorFlow model described here:
    https://github.com/wkopp/janggu_usecases/tree/master/01_jund_prediction

    This model can only be run in "ds" mode. The reverse complement must be included in the Dataloader
    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    strand : str, optional
        Strand of the input. This model is only implemented for "ds"
    task : str, optional
        Task for this model. By default "binary_classification" for this mode
    aggr : str, optional
        Aggregation method. Either "concat", "max", or "avg". By default "max" for this model.
    filters : list, optional
        Number of filters in the convolutional layers. 
    conv_kernel_size : list, optional
        Kernel size of the convolutional layers.
    maxpool_kernel_size : int, optional
        Kernel size of the maxpooling layer.
    stride : int, optional
        Stride of the convolutional layers.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str ="ds",
        task: str = "binary_classification",
        aggr: str = "max",
        loss_fxn: str = "bce",
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
        self.conv2 = nn.Conv1d(filters[0], filters[1], conv_kernel_size[1], stride=stride)
        self.batchnorm2 = nn.BatchNorm1d(filters[1])
        self.linear = nn.Linear(filters[1], self.output_dim)

    def forward(self, x, x_rev_comp):
        x_fwd = F.relu(self.conv(x))
        x_rev_comp = F.relu(self.conv(x_rev_comp))
        if self.aggr == "concat":
            x = torch.cat((x_fwd, x_rev_comp), dim=2)
        elif self.aggr == "max":
            x = torch.max(x_fwd, x_rev_comp)
        elif self.aggr == "avg":
            x = (x_fwd + x_rev_comp) / 2
        elif self.aggr is None:
            x = torch.cat((x_fwd, x_rev_comp), dim=1)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])
        x = self.batchnorm2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
