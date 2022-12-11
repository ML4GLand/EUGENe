import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import SequenceModel 
from .base._blocks import DenseBlock, Conv1DBlock, RecurrentBlock, BiConv1DBlock
from .base._utils import GetFlattenDim


class TutorialCNN( SequenceModel):
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
        Keyword arguments to pass to the  SequenceModel class.
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
            
            
    def forward(self, x, x_rev_comp=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, x.size()[-1]).flatten(1, -1)
        x = self.dense(x)
        if self.strand == "ds":
            x_rev_comp = F.relu(self.conv1(x_rev_comp))
            x_rev_comp = F.max_pool1d(x_rev_comp, x_rev_comp.size()[-1]).flatten(1, -1)
            x_rev_comp = self.dense(x_rev_comp)
            x = (x + x_rev_comp / 2)
        return x


class Jores21CNN(SequenceModel):
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
        self.biconv = BiConv1DBlock(
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


class Kopp21CNN(SequenceModel):
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


class FactorizedBasset( SequenceModel):
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
		dense_kwargs = {},
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
		self.conv1_kwargs, self.conv2_kwargs, self.conv3_kwargs, self.maxpool_kernels, self.dense_kwargs = self.kwarg_handler(
			conv1_kwargs, 
			conv2_kwargs, 
			conv3_kwargs, 
			maxpool_kernels, 
		    dense_kwargs	
		)
		self.conv1d_block1 = Conv1DBlock(
			input_len=input_len, 
            input_channels=4,
			**self.conv1_kwargs
		)
		self.conv1d_block2 = Conv1DBlock(
			input_len=self.conv1d_block1.output_len,
            input_channels=self.conv1d_block1.out_channels,
			**self.conv2_kwargs
		)
		self.conv1d_block3 = Conv1DBlock(
			input_len=self.conv1d_block2.output_len,
            input_channels=self.conv1d_block2.out_channels,
			**self.conv3_kwargs
		)
		self.dense_block = DenseBlock(
			input_dim=self.conv1d_block3.flatten_dim,
			output_dim=output_dim, 
			**self.dense_kwargs
		)

	def forward(self, x, x_rev_comp=None):
		x = self.conv1d_block1(x)
		x = self.conv1d_block2(x)
		x = self.conv1d_block3(x)
		x = x.view(x.size(0), self.conv1d_block3.flatten_dim)
		x = self.dense_block(x)
		return x
        
	def kwarg_handler(self, conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels, dense_kwargs):
		"""Sets default kwargs FactorizedBasset"""
		conv1_kwargs.setdefault("conv_channels", [48, 64, 100, 150, 300])
		conv1_kwargs.setdefault("conv_kernels", [3, 3, 3, 7, 7])
		conv1_kwargs.setdefault("conv_strides", [1, 1, 1, 1, 1])
		conv1_kwargs.setdefault("conv_padding", [1, 1, 1, 3, 3])
		conv1_kwargs.setdefault("pool_types", [None, None, None, None, "max"])
		conv1_kwargs.setdefault("pool_kernels", [None, None, None, None, 3])
		conv1_kwargs.setdefault("dropout_rates", 0.0)
		conv1_kwargs.setdefault("batchnorm", True)
		conv1_kwargs.setdefault("batchnorm_first", True)
		conv1_kwargs.setdefault("activations", "relu")
		conv2_kwargs.setdefault("conv_channels", [200, 200, 200])
		conv2_kwargs.setdefault("conv_kernels", [7, 3, 3])
		conv2_kwargs.setdefault("conv_strides", [1, 1, 1])
		conv2_kwargs.setdefault("conv_padding", [3, 1, 1])
		conv2_kwargs.setdefault("pool_types", [None, None, "max"])
		conv2_kwargs.setdefault("pool_kernels", [None, None, 4])
		conv2_kwargs.setdefault("dropout_rates", 0.0)
		conv2_kwargs.setdefault("batchnorm", True)
		conv2_kwargs.setdefault("batchnorm_first", True)
		conv2_kwargs.setdefault("activations", "relu")
		conv3_kwargs.setdefault("conv_channels", [200])
		conv3_kwargs.setdefault("conv_kernels", [7])
		conv3_kwargs.setdefault("conv_strides", [1])
		conv3_kwargs.setdefault("conv_padding", [3])
		conv3_kwargs.setdefault("pool_types", ["max"])
		conv3_kwargs.setdefault("pool_kernels", [4])
		conv3_kwargs.setdefault("dropout_rates", 0.0)
		conv3_kwargs.setdefault("batchnorm", True)
		conv3_kwargs.setdefault("batchnorm_first", True)
		conv3_kwargs.setdefault("activations", "relu")
		dense_kwargs.setdefault("hidden_dims", [1000, 164])
		dense_kwargs.setdefault("dropout_rates", 0.0)
		dense_kwargs.setdefault("batchnorm", True)
		dense_kwargs.setdefault("batchnorm_first", True)
		dense_kwargs.setdefault("activations", "relu")
		return conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels,dense_kwargs 


class ResidualBind( SequenceModel):
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
        self.conv = Conv1DBlock(
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
        self.fc = DenseBlock(
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


class DeepBind( SequenceModel):
    """
    DeepBind model implemented from Alipanahi et al 2015 in PyTorch

    DeepBind is a model that takes in a DNA or RNA sequence and outputs a probability of 
    binding for a given DNA transcription factor or RNA binding protein respectively.
    This is a flexible implementation of the original DeepBind architecture that allows users
    to modify the number of convolutional layers, the number of fully connected layers, and
    many more hyperparameters. If parameters for the CNN and FCN are not passed in, the model
    will be instantiated with the parameters described in Alipanahi et al 2015.

    Like the original DeepBind models, this model can be used for both DNA and RNA binding. For DNA,
    we implemented the "dna" mode which only uses the max pooling of the representation generated by 
    the convolutional layers. For RNA, we implemented the "rbp" mode which uses both the max and average
    pooling of the representation generated by the convolutional layers.
    - For "ss" models, we use the representation generated by the convolutional layers and pass that through a 
        set of fully connected layer to generate the output.
    - For "ds" models, we use the representation generated by the convolutional layers for both the forward and 
        reverse complement strands and pass that through the same set of fully connected layers to generate the output.
    - For "ts" models, we use the representation generated by separate sets of convolutional layers for the forward and
        reverse complement strands and passed that through separate sets of fully connected layers to generate the output.
    
    aggr defines how the output for "ds" and "ts" models is generated. If "max", we take the max of the forward and reverse
    complement outputs. If "avg", we take the average of the forward and reverse complement outputs. There is no "concat" for
    the current implementation of DeepBind models.

    Parameters
    ----------
    input_len : int
        Length of input sequence
    output_dim : int
        Number of output classes
    mode : str
        Mode of model, either "dna" or "rbp"
    strand : str
        Strand of model, either "ss", "ds", or "ts"
    task : str
        Task of model, either "regression" or "classification"
    aggr : str
        Aggregation method of model, either "max" or "avg"
    loss_fxn : str
        Loss function of model, either "mse" or "cross_entropy"
    optimizer : str
        Optimizer of model, either "adam" or "sgd"
    lr : float
        Learning rate of model
    scheduler : str
        Scheduler of model, either "lr_scheduler" or "plateau"
    scheduler_patience : int
        Scheduler patience of model
    mp_kwargs : dict
        Keyword arguments for multiprocessing
    conv_kwargs : dict
        Keyword arguments for convolutional layers
    dense_kwargs : dict
        Keyword arguments for fully connected layers
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = "max",
        loss_fxn: str ="mse",
        mode: str = "rbp",
        conv_kwargs: dict = {},
        dense_kwargs: dict = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.mode = mode
		self.mode_dict = {"dna": 1, "rbp": 2}
        self.mode_multiplier = mode_dict[self.mode]
        self.aggr = aggr
        self.conv1d_block = Conv1DBlock(input_len=input_len, **self.conv_kwargs)
        self.pool_dim = GetFlattenDim(self.conv1d_block.module, seq_len=input_len)
        self.max_pool = nn.MaxPool1d(kernel_size=self.pool_dim)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.pool_dim)
        if self.strand == "ss":
            self.dense_block = DenseBlock(
                input_dim=self.conv1d_block.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )
        elif self.strand == "ds":
            self.dense_block = DenseBlock(
                input_dim=self.conv1d_block.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )
        elif self.strand == "ts":
            self.dense_block = DenseBlock(
                self.conv1d_block.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )
            self.reverse_conv1d_block = Conv1DBlock(
                input_len=input_len, 
                **self.conv_kwargs
                )
            self.reverse_dense_block = DenseBlock(
                self.conv1d_block.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )



def forward(self, x, x_rev_comp=None):

        x = self.conv1d_block(x)
        if self.mode == "rbp":
            x = torch.cat((self.max_pool(x), self.avg_pool(x)), dim=1)
            x = x.view(x.size(0), self.conv1d_block.out_channels * 2)
        elif self.mode == "dna":
            x = self.max_pool(x)
            x = x.view(x.size(0), self.conv1d_block.out_channels)
        x = self.dense_block(x)
        if self.strand == "ds":
            x_rev_comp = self.conv1d_block(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_block.out_channels * 2)
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_block.out_channels)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_conv1d_block(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_block.out_channels * 2)
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_block.out_channels)
            x_rev_comp = self.reverse_dense_block(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x

    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 16])
        conv_kwargs.setdefault("conv_kernels", [16])
        conv_kwargs.setdefault("pool_types", None)
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("dropout_rates", 0.25)
        conv_kwargs.setdefault("batchnorm", False)
        dense_kwargs.setdefault("hidden_dims", [32])
        dense_kwargs.setdefault("dropout_rate", 0.25)
        dense_kwargs.setdefault("batchnorm", False)
        return conv_kwargs,dense_kwargs 


class DeepSEA( SequenceModel):
    """DeepSEA model implementation for EUGENe
    
    Default parameters are those specified in the DeepSEA paper. We currently do not implement a "ds" or "ts" model
    for DeepSEA.

    Parameters
    ----------
    input_len:
        int, input sequence length
    channels:
        list-like or int, channel width for each conv layer. If int each of the three layers will be the same channel width
    conv_kernels:
        list-like or int, conv kernel size for each conv layer. If int will be the same for all conv layers
    pool_kernels:
        list-like or int, maxpooling kernel size for the first two conv layers. If int will be the same for all conv layers
    dropout_rates:
        list-like or float, dropout rates for each conv layer. If int will be the same for all conv layers
    """
    def __init__(
        self,
        input_len: int = 1000,
        output_dim: int = 1,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        conv_kwargs: dict = {},
        dense_kwargs: dict = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.conv1d_block = Conv1DBlock(
            input_len=input_len, 
            **self.conv_kwargs)
        self.dense_block = DenseBlock(
            input_dim=self.conv1d_block.flatten_dim, 
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_block(x)
        x = x.view(x.size(0), self.conv1d_block.flatten_dim)
        x = self.dense_block(x)
        return x

    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 320, 480, 960])
        conv_kwargs.setdefault("conv_kernels", [8, 8, 8])
        conv_kwargs.setdefault("pool_types", [4, 4, 4])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("activation", "relu")
        conv_kwargs.setdefault("dropout_rates", [0.2, 0.2, 0.5])
        conv_kwargs.setdefault("batchnorm", False)
        dense_kwargs.setdefault("hidden_dims", [925])
        return conv_kwargs,dense_kwargs 


class Basset( SequenceModel):
    """
    """
    def __init__(
        self, 
        input_len: int = 1000,
        output_dim = 1, 
        strand = "ss",
        task = "binary_classification",
        aggr = None,
        loss_fxn = "bce",
        conv_kwargs = {},
        dense_kwargs = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.conv1d_block = Conv1DBlock(
            input_len=input_len, 
            **self.conv_kwargs)
        self.dense_block = DenseBlock(
            input_dim=self.conv1d_block.flatten_dim, 
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_block(x)
        x = x.view(x.size(0), self.conv1d_block.flatten_dim)
        x = self.dense_block(x)
        return x
        
    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 300, 200, 200])
        conv_kwargs.setdefault("conv_kernels", [19, 11, 7])
        conv_kwargs.setdefault("conv_strides", [1, 1, 1])
        conv_kwargs.setdefault("padding", [9, 5, 3])
        conv_kwargs.setdefault("pool_types", [3, 4, 4])
        conv_kwargs.setdefault("omit_final_pool", False)
        conv_kwargs.setdefault("dropout_rates", 0.0)
        conv_kwargs.setdefault("batchnorm", True)
        conv_kwargs.setdefault("activation", "relu")
        dense_kwargs.setdefault("hidden_dims", [1000, 164])
        dense_kwargs.setdefault("dropout_rate", 0.0)
        dense_kwargs.setdefault("batchnorm", True)
        dense_kwargs.setdefault("activation", "relu")
        return conv_kwargs,dense_kwargs 


class DanQ( SequenceModel):
    """DanQ model from Quang and Xie, 2016;

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    strand:
        The strand of the model.
    task:
        The task of the model.
    aggr:
        The aggregation function.
    dense_kwargs:
        The keyword arguments for the fully connected layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        loss_fxn: str = "mse",
        aggr: str = None,
        cnn_kwargs: dict = {},
        rnn_kwargs: dict = {},
        dense_kwargs: dict = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(cnn_kwargs, rnn_kwargs, dense_kwargs)
        self.conv1d_block = Conv1DBlock(
            input_len=input_len, 
            **cnn_kwargs)
        self.recurrentnet = BasicRecurrent(
            input_dim=self.conv1d_block.out_channels, 
            **rnn_kwargs
        )
        self.dense_block = DenseBlock(
            input_dim=self.recurrentnet.out_dim, 
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_block(x)
        x = x.transpose(1, 2)
        out, _ = self.recurrentnet(x)
        out = self.dense_block(out[:, -1, :])
        return out

    def kwarg_handler(self, cnn_kwargs, rnn_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        cnn_kwargs.setdefault("channels", [4, 320])
        cnn_kwargs.setdefault("conv_kernels", [26])
        cnn_kwargs.setdefault("conv_strides", [1])
        cnn_kwargs.setdefault("padding", "same")
        cnn_kwargs.setdefault("pool_types", [13])
        cnn_kwargs.setdefault("omit_final_pool", False)
        cnn_kwargs.setdefault("dropout_rates", 0.2)
        cnn_kwargs.setdefault("activation", "relu")
        rnn_kwargs.setdefault("unit_type", "lstm")
        rnn_kwargs.setdefault("output_dim", 320)
        rnn_kwargs.setdefault("bidirectional", True)
        rnn_kwargs.setdefault("batch_first", True)
        dense_kwargs.setdefault("hidden_dims", [925])
        dense_kwargs.setdefault("dropout_rate", 0.5)
        dense_kwargs.setdefault("batchnorm", False)
        return cnn_kwargs,dense_kwargs 


class DeepSTARR( SequenceModel):
    """DeepSTARR model from de Almeida et al., 2022; 
        see <https://www.nature.com/articles/s41588-022-01048-5>


    Parameters
    """
    def __init__(
        self, 
        input_len: int = 249,
        output_dim = 2, 
        strand = "ss",
        task = "regression",
        aggr = None,
        loss_fxn = "mse",
        conv_kwargs = {},
        dense_kwargs = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.conv1d_block = Conv1DBlock(
            input_len=input_len, 
            **self.conv_kwargs)
        self.dense_block = DenseBlock(
            input_dim=self.conv1d_block.flatten_dim, 
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_block(x)
        x = x.view(x.size(0), self.conv1d_block.flatten_dim)
        x = self.dense_block(x)
        return x
        
    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 246, 60, 60, 120])
        conv_kwargs.setdefault("conv_kernels", [7, 3, 5, 3])
        conv_kwargs.setdefault("conv_strides", [1, 1, 1, 1])
        conv_kwargs.setdefault("padding", "same")
        conv_kwargs.setdefault("pool_types", [2, 2, 2, 2])
        conv_kwargs.setdefault("omit_final_pool", False)
        conv_kwargs.setdefault("dropout_rates", 0.0)
        conv_kwargs.setdefault("batchnorm", True)
        dense_kwargs.setdefault("hidden_dims", [256, 256])
        dense_kwargs.setdefault("dropout_rate", 0.4)
        dense_kwargs.setdefault("batchnorm", True)
        return conv_kwargs,dense_kwargs 