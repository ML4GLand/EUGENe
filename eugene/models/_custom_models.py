import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, BiConv1D, BasicConv1D, BasicFullyConnectedModule


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


class FactorizedBasset(BaseModel):
	def __init__(
		self, 
		input_len: int = 1000,
		output_dim = 1, 
		strand = "ss",
		task = "binary_classification",
		aggr = None,
		loss_fxn = "bce",
		conv1_kwargs = {},
		conv2_kwargs = {},
		conv3_kwargs = {},
		maxpool_kernels = None,
		fc_kwargs = {},
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
		self.conv1_kwargs, self.conv2_kwargs, self.conv3_kwargs, self.maxpool_kernels, self.fc_kwargs = self.kwarg_handler(
			conv1_kwargs, 
			conv2_kwargs, 
			conv3_kwargs, 
			maxpool_kernels, 
			fc_kwargs
		)
		self.convnet1 = BasicConv1D(
			input_len=input_len, 
			**self.conv1_kwargs
		)
		self.maxpool1 = nn.MaxPool1d(self.maxpool_kernels[0])
		self.out1 = self.convnet1.flatten_dim/self.convnet1.out_channels // self.maxpool_kernels[0]
		self.convnet2 = BasicConv1D(
			input_len=self.out1,
			**self.conv2_kwargs
		)
		self.maxpool2 = nn.MaxPool1d(self.maxpool_kernels[1])
		self.out2 = self.convnet2.flatten_dim/self.convnet2.out_channels // self.maxpool_kernels[1]
		self.convnet3 = BasicConv1D(
			input_len=self.out2,
			**self.conv3_kwargs
		)
		self.maxpool3 = nn.MaxPool1d(self.maxpool_kernels[2])
		self.out3 = self.convnet3.flatten_dim/self.convnet3.out_channels // self.maxpool_kernels[2]
		self.fcnet_in = int(self.out3*self.convnet3.out_channels)
		self.fcnet = BasicFullyConnectedModule(
			input_dim=self.fcnet_in,
			output_dim=output_dim, 
			**self.fc_kwargs
		)

	def forward(self, x, x_rev_comp=None):
		x = self.convnet1(x)
		x = self.maxpool1(x)
		x = self.convnet2(x)
		x = self.maxpool2(x)
		x = self.convnet3(x)
		x = self.maxpool3(x)
		x = x.view(x.size(0), self.fcnet_in)
		x = self.fcnet(x)
		return x
        
	def kwarg_handler(self, conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels, fc_kwargs):
		"""Sets default kwargs for conv and fc modules if not specified"""
		conv1_kwargs.setdefault("channels", [4, 48, 64, 100, 150, 300])
		conv1_kwargs.setdefault("conv_kernels", [3, 3, 3, 7, 7])
		conv1_kwargs.setdefault("conv_strides", [1, 1, 1, 1, 1])
		conv1_kwargs.setdefault("padding", [1, 1, 1, 3, 3])
		conv1_kwargs.setdefault("pool_kernels", None)
		conv1_kwargs.setdefault("dropout_rates", 0.0)
		conv1_kwargs.setdefault("batchnorm", True)
		conv1_kwargs.setdefault("activation", "relu")
		conv2_kwargs.setdefault("channels", [300, 200, 200, 200])
		conv2_kwargs.setdefault("conv_kernels", [7, 3, 3])
		conv2_kwargs.setdefault("conv_strides", [1, 1, 1])
		conv2_kwargs.setdefault("padding", [3, 1, 1])
		conv2_kwargs.setdefault("pool_kernels", None)
		conv2_kwargs.setdefault("dropout_rates", 0.0)
		conv2_kwargs.setdefault("batchnorm", True)
		conv2_kwargs.setdefault("activation", "relu")
		conv3_kwargs.setdefault("channels", [200, 200])
		conv3_kwargs.setdefault("conv_kernels", [7])
		conv3_kwargs.setdefault("conv_strides", [1])
		conv3_kwargs.setdefault("padding", [3])
		conv3_kwargs.setdefault("pool_kernels", None)
		conv3_kwargs.setdefault("dropout_rates", 0.0)
		conv3_kwargs.setdefault("batchnorm", True)
		conv3_kwargs.setdefault("activation", "relu")
		maxpool_kernels = [3, 4, 4] if maxpool_kernels is None else maxpool_kernels
		fc_kwargs.setdefault("hidden_dims", [1000, 164])
		fc_kwargs.setdefault("dropout_rate", 0.0)
		fc_kwargs.setdefault("batchnorm", True)
		fc_kwargs.setdefault("activation", "relu")
		return conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels, fc_kwargs


class ResidualBind(BaseModel):
    def __init__(
        self,
        input_len,
        output_dim,
        strand="ss",
        task="regression",
        aggr=None,
        conv_channels=[96],
        conv_kernel_size=[11],
        conv_stride_size=[1],
        conv_dilation_rate=[1],
        conv_padding="same",
        conv_activation="relu",
        conv_batchnorm=True,
        conv_dropout=0.1,
        residual_channels=[3, 3, 3],
        residual_kernel_size=[11, 11, 11],
        residual_stride_size=[1, 1, 1],
        residual_dilation_rate=[1, 1, 1],
        residual_padding="same",
        residual_activation="relu",
        residual_batchnorm=True,
        residual_dropout=0.1,
        pool_kernel_size=10,
        pool_dropout=0.2,
        fc_hidden_dims=[256],
        fc_activation="relu",
        fc_batchnorm=True,
        fc_dropout=0.0,
        **kwargs
    ):
        super().__init__(
            input_len, output_dim, strand=strand, task=task, aggr=aggr, **kwargs
        )
        if isinstance(conv_channels, int):
            conv_channels = [conv_channels]
        self.conv = BasicConv1D(
            input_len=input_len,
            channels=[4] + conv_channels,
            conv_kernels=conv_kernel_size,
            conv_strides=conv_stride_size,
            pool_kernels=None,
            activation=conv_activation,
            pool_strides=None,
            dropout_rates=conv_dropout,
            dilations=conv_dilation_rate,
            padding=conv_padding,
            batchnorm=conv_batchnorm
        )
        res_block_input_len = GetFlattenDim(self.conv.module, seq_len=input_len)
        self.residual_block = ResidualModule(
            input_len=res_block_input_len,
            channels=[self.conv.out_channels] + residual_channels,
            conv_kernels=residual_kernel_size,
            conv_strides=residual_stride_size,
            pool_kernels=None,
            activation=residual_activation,
            pool_strides=None,
            dropout_rates=residual_dropout,
            dilations=residual_dilation_rate,
            padding=residual_padding,
            batchnorm=residual_batchnorm
        )
        self.average_pool = nn.AvgPool1d(pool_kernel_size, stride=1)
        self.dropout = nn.Dropout(pool_dropout)
        self.flatten = nn.Flatten()
        self.fc = BasicFullyConnectedModule(
            input_dim=self.residual_block.module.out_channels*(res_block_input_len-pool_kernel_size+1),
            output_dim=output_dim,
            hidden_dims=fc_hidden_dims,
            activation=fc_activation,
            batchnorm=fc_batchnorm,
            dropout_rate=fc_dropout
        )

    def forward(self, x, x_rev):
        x = self.conv(x)
        x = self.residual_block(x)
        x = self.average_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x