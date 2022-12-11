import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from typing import Union, Callable, List
from ._utils import GetFlattenDim
from ._activations import ACTIVATION_REGISTRY
from ._layers import POOLING_REGISTRY, RECURRENT_REGISTRY


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
        self.num_layers = len(hidden_dims) + 1
        self.hidden_dims = hidden_dims if len(hidden_dims) > 0 else [input_dim]

        # Define the activations
        activations = activations if type(activations) == list else [activations] * len(hidden_dims)
        self.activations = [ACTIVATION_REGISTRY[activation](inplace=False) if isinstance(activation, str) else activation for activation in activations] 

        # Define the dropout rates
        if dropout_rates != 0.0 and dropout_rates is not None:
            if type(dropout_rates) == float:
                self.dropout_rates = [dropout_rates] * len(hidden_dims)
            elif type(dropout_rates) == list:
                self.dropout_rates = dropout_rates 
        else:
            self.dropout_rates = []
        
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
                if self.activations[i] is not None:
                    self.layers.append(self.activations[i])

            # Add dropout if specified
            if i < len(self.dropout_rates):
                if self.dropout_rates[i] is not None:
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
        print(pool_types, self.pool_types)
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
        else:
            self.dropout_rates = []
        
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
                if self.activations[i] is not None:
                    self.layers.append(self.activations[i])
            
            # Add a dropout layer if specified
            if i < len(self.dropout_rates):
                if dropout_rates[i] is not None:
                    self.layers.append(nn.Dropout(self.dropout_rates[i]))

            
            # Add a pooling layer if specified
            if i < len(self.pool_types):
                if self.pool_types[i] is not None:
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
            self.output_len = GetFlattenDim(self.layers, seq_len=self.input_len)
            self.flatten_dim = self.output_len * self.out_channels

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


class TransformerBlock(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        head_dim: int, 
        num_heads: int = 1,
        dropout_rates: float = 0.0, 
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.projection_dim = self.num_heads * self.head_dim
        need_projection = not ((self.projection_dim == self.input_dim) and (self.num_heads == 1)) 
        self.dropout_rates = dropout_rates

        self.scale_factor = head_dim ** -0.5
        self.qkv = nn.Linear(
            self.input_dim, 
            self.projection_dim * 3, 
            bias = False
        )
        
        self.softmax = nn.Softmax(dim = -1)
        self.dropout_layer = nn.Dropout(self.dropout_rates)
        
        self.projection_layer = nn.Sequential(
            nn.Linear(self.projection_dim, self.input_dim), 
            nn.Dropout(self.dropout_rates)
        ) if need_projection else nn.Identity()
        
    def forward(self, x, mask):
        qkv = self.qkv(x).chunk(3, dim = -1)  #qkv is a tuple of tensors - need to map to extract individual q,k,v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)  
        
        scaled_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        
        if mask is not None: 
            mask = mask.unsqueeze(1).expand(x.size(0), q.size(2), k.size(2)) # [b,n] --> [b,1,n] --> [b,n,n]
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1) #Tell Zhu-Li we did the thing: [b,n,n] --> [b,h,n,n]    
            scaled_score = scaled_score.masked_fill(mask, torch.finfo(torch.float32).min)
            
        attention = self.softmax(scaled_score)
        attention = self.dropout_layer(attention)
        
        output = torch.matmul(attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projection_layer(output)
        return output


class BiConv1DBlock(nn.Module):
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


class ResidualConv1DBlock(nn.Module):
    """Generates a PyTorch module with the residual binding architecture described in:

    Parameters
    ----------
    input_len : int
        Length of the input sequence

    """
    def __init__(
        self, 
        input_len, 
        input_channels,
        conv_channels, 
        conv_kernels, 
        conv_strides,
        dilations, 
        pool_kernels=None, 
        activation="relu", 
        pool_strides=None, 
        dropout_rates=0.0, 
        padding="same", 
        batchnorm=True
    ):
        super().__init__()
        self.module = Conv1DBlock(
            input_len=input_len,
            input_channels=input_channels,
            conv_channels=channels,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            conv_dilations=dilations,
            conv_padding=padding,
            activations=activation,
            pool_kernels=pool_kernels,
            pool_strides=pool_strides,
            dropout_rates=dropout_rates,
            batchnorm=batchnorm
        )

    def forward(self, x):
        x_fwd = self.module(x)
        return F.relu(x_fwd + x)


BLOCK_REGISTRY = {
    "dense": DenseBlock,
    "conv1d": Conv1DBlock,
    "recurrent": RecurrentBlock
}
