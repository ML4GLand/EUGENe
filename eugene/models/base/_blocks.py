import torch.nn as nn
from ._utils import GetFlattenDim
from ._activations import ACTIVATION_REGISTRY
from ._layers import POOLING_REGISTRY, RECURRENT_REGISTRY
from typing import Union, Callable, List

class DenseBlock(nn.Module):
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: list = [],
        activations: Union[str, Callable, List[Union[str, Callable]]] = "relu",
        dropout_rates: float = 0.0,
        batchnorm: bool = False,
        batchnorm_first: bool = False,
        biases: bool = True
    ):
        super(DenseBlock, self).__init__()

        # Define the layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) + 1

        # Define the activations
        activations = activations if type(activations) == list else [activations] * len(hidden_dims)
        self.activations = [ACTIVATION_REGISTRY[activation](inplace=False) if isinstance(activation, str) else activation for activation in activations] 

        # Define the dropout rates
        if dropout_rates != 0.0 and dropout_rates is not None:
            if type(dropout_rates) == float:
                self.dropout_rates = [dropout_rates] * len(hidden_dims)
            elif type(dropout_rates) == list:
                self.dropout_rates = dropout_rates 
        
        # Define the batchnorm
        self.batchnorm = batchnorm
        self.batchnorm_first = batchnorm_first

        # Define the biases
        self.biases = biases if isinstance(biases, list) else [biases] * self.num_layers

        # Build the block
        self.layers = nn.Sequential()
        j = 0
        for i in range(self.num_layers-1):
            
            # Add the linear layer
            if i == 0:
                self.layers.append(
                    nn.Linear(
                        self.input_dim,
                        self.hidden_dims[0],
                        bias=self.biases[0]
                    )
                )
            else:
                self.layers.append(
                    nn.Linear(
                        self.hidden_dims[i - 1],
                        self.hidden_dims[i],
                        bias=self.biases[i]
                    )
                )

            # Add batchnorm if specified
            if batchnorm and batchnorm_first:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i]))
            
            # Add activation
            if i < len(self.activations):
                self.layers.append(self.activations[i])

            # Add dropout if specified
            if i < len(self.dropout_rates):
                self.layers.append(nn.Dropout(self.dropout_rates[i]))

            # Add batchnorm if specified
            if batchnorm and not batchnorm_first:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i]))

            # Keep track of the number of layers seen
            j += 1

        # Add the final linear layer
        self.layers.append(
            nn.Linear(
                self.hidden_dims[-1],
                self.output_dim,
                bias=self.biases[-1]
            )
        )
        
        # If enough droupout rates are specified, add the final dropout
        if len(self.dropout_rates) > j:
            self.layers.append(nn.Dropout(self.dropout_rates[j]))

    def forward(self, x):
        return self.layers(x)
    

class Conv1DBlock(nn.Module):

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
        batchnorm_first: bool = False
    ):
        super(Conv1DBlock, self).__init__()

        # Define input parameters
        self.input_len = input_len
        self.input_channels = input_channels

        # Define convolutional layers
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides if conv_strides is not None else [1] * len(conv_channels) 
        self.conv_dilations = conv_dilations if conv_dilations is not None else [1] * len(conv_channels) 
        self.conv_padding = conv_padding if type(conv_padding) == list else [conv_padding] * len(conv_channels) 
        self.conv_biases = conv_biases if type(conv_biases) == list else [conv_biases] * len(conv_channels)
        assert len(self.conv_channels) == len(self.conv_kernels) == len(self.conv_strides) == len(self.conv_dilations) == len(self.conv_padding) == len(self.conv_biases), "Convolutional parameters must be of the same length"

        # Define activation layers
        activations = activations if type(activations) == list else [activations] * len(conv_channels)
        self.activations = [ACTIVATION_REGISTRY[activation](inplace=False) if isinstance(activation, str) else activation for activation in activations]
        
        # Define pooling layers
        pool_types = pool_types if type(pool_types) == list else [pool_types] * len(conv_channels)
        self.pool_types = [POOLING_REGISTRY[pool_type] if isinstance(pool_type, str) else pool_type for pool_type in pool_types]
        self.pool_kernels = pool_kernels if pool_kernels is not None else [1] * len(self.pool_types)
        self.pool_strides = pool_kernels if pool_strides is not None else [1] * len(self.pool_types)
        self.pool_padding = pool_padding if pool_padding is not None else [0] * len(self.pool_types)
        self.pool_dilations = pool_dilations if pool_dilations is not None else [1] * len(self.pool_types)
        assert len(self.pool_types) == len(self.pool_kernels) == len(self.pool_strides) == len(self.pool_padding) == len(self.pool_dilations), "Pooling parameters must be of equal length"

        # Define dropout layers
        if dropout_rates != 0.0 and dropout_rates is not None:
            if type(dropout_rates) == float:
                self.dropout_rates = [dropout_rates] * len(conv_channels)
            elif type(dropout_rates) == list:
                self.dropout_rates = dropout_rates
        
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
                self.layers.append(self.activations[i])
            
            # Add a dropout layer if specified
            if i < len(self.dropout_rates):
                self.layers.append(nn.Dropout(self.dropout_rates[i]))
            
            # Add a pooling layer if specified
            if i < len(self.pool_kernels):
                self.layers.append(
                    self.pool_types[i](
                        kernel_size=self.pool_kernels[i],
                        stride=self.pool_strides[i],
                        padding=self.pool_padding[i]
                    )
                )
            
            # Add a batchnorm layer if specified
            if self.batchnorm and not self.batchnorm_first:
                self.layers.append(nn.BatchNorm1d(conv_channels[i]))
            
            # Define output parameters
            self.out_channels = self.conv_channels[-1]
            self.flatten_dim = GetFlattenDim(self.layers, seq_len=self.input_len) * self.out_channels

    def forward(self, x):
        return self.layers(x)


class RecurrentBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        unit_type: str = "lstm",
        bidirectional: bool = False,
        dropout_rates: float = 0.0,
        bias=True,
        batch_first=True
    ):
        super(RecurrentBlock, self).__init__()

        # Define input parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.unit_type = RECURRENT_REGISTRY[unit_type]
        self.bidirectional = bidirectional
        self.dropout_rates = dropout_rates
        self.bias = bias
        self.batch_first = batch_first

        # Define recurrent layers
        self.layers = self.unit_type(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout_rates,
            bidirectional=self.bidirectional
        )

        # Define output parameters
        self.out_channels = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

    def forward(self, x):
        return self.layers(x)





BLOCK_REGISTRY = {
    "dense": DenseBlock,
    "conv1d": Conv1DBlock,
    "recurrent": RecurrentBlock
}
