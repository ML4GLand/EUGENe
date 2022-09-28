import torch
import torch.nn as nn
import torch.nn.functional as F
from ._utils import GetFlattenDim, BuildFullyConnected


class BiConv1D(nn.Module):
    """Generates a PyTorch module with the convolutional architecture described in:
    TODO: Add paper reference

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
        dropout_rate: float = 0.15
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
        x_fwd = F.dropout(F.relu(x_fwd), p=self.dropout_rate)
        x_rev = F.conv1d(
            x,
            torch.flip(self.kernels[0], dims=[0, 1]),
            stride=self.stride,
            padding="same",
        )
        x_rev = torch.add(x_rev.transpose(1, 2), self.biases[0]).transpose(1, 2)
        x_rev = F.dropout(F.relu(x_rev), p=self.dropout_rate)
        for layer in range(1, self.layers):
            x_fwd = F.conv1d(
                x_fwd, self.kernels[layer], stride=self.stride, padding="same"
            )
            x_fwd = torch.add(x_fwd.transpose(1, 2), self.biases[layer]).transpose(1, 2)
            x_fwd = F.dropout(F.relu(x_fwd), p=self.dropout_rate)
            x_rev = F.conv1d(
                x_rev,
                torch.flip(self.kernels[layer], dims=[0, 1]),
                stride=self.stride,
                padding="same",
            )
            x_rev = torch.add(x_rev.transpose(1, 2), self.biases[layer]).transpose(1, 2)
            x_rev = F.dropout(F.relu(x_rev), p=self.dropout_rate)
        return torch.add(x_fwd, x_rev)
