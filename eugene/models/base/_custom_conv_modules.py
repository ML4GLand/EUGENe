import torch
import torch.nn as nn
import torch.nn.functional as F
from ._utils import GetFlattenDim, BuildFullyConnected


class BiConv1D(nn.Module):
    def __init__(
        self, filters, kernel_size, input_size=4, layers=2, stride=1, dropout_rate=0.15
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

        self.kernels = []
        self.biases = []
        kernel = torch.zeros(filters, input_size, kernel_size)
        nn.init.xavier_uniform_(kernel)
        self.kernels.append(kernel)
        bias = torch.zeros(filters)
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
            torch.flip(self.kernels[0], dims=[1, 2]),
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
                torch.flip(self.kernels[layer], dims=[1, 2]),
                stride=self.stride,
                padding="same",
            )
            x_rev = torch.add(x_rev.transpose(1, 2), self.biases[layer]).transpose(1, 2)
            x_rev = F.dropout(F.relu(x_rev), p=self.dropout_rate)
        return torch.add(x_fwd, x_rev)
