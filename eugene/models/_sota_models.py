import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, BasicFullyConnectedModule, BasicConv1D
from .base._utils import GetFlattenDim


mode_dict = {"dna": 1, "rbp": 2}


class DeepBind(BaseModel):
    """
    DeepBind model implemented from Alipanahi et al 2015 in PyTorch

    DeepBind is a model that takes in a DNA sequence and outputs a
    probability of binding for a given transcription factor.

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
    fc_kwargs : dict
        Keyword arguments for fully connected layers
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        mode: str = "rbp",
        strand: str = "ss",
        task: str = "regression",
        aggr: str = "max",
        loss_fxn: str ="mse",
        optimizer: str = "adam",
        lr: float = 1e-3,
        scheduler: str = "lr_scheduler",
        scheduler_patience: int = 2,
        conv_kwargs: dict = {},
        fc_kwargs: dict = {},
        **kwargs
    ):
        super().__init__(
            input_len,
            output_dim,
            strand,
            task,
            aggr,
            loss_fxn,
            optimizer,
            lr,
            scheduler,
            scheduler_patience,
            **kwargs
        )
        self.conv_kwargs, self.fc_kwargs = self.kwarg_handler(
            conv_kwargs, fc_kwargs
        )
        self.mode = mode
        self.mode_multiplier = mode_dict[self.mode]
        self.aggr = aggr

        # Conv stuff
        self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
        self.pool_dim = GetFlattenDim(self.convnet.module, seq_len=input_len)
        self.max_pool = nn.MaxPool1d(kernel_size=self.pool_dim)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.pool_dim)

        # Add strand specific conv and fcn modules
        if self.strand == "ss":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )
        elif self.strand == "ds":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )
        elif self.strand == "ts":
            self.fcn = BasicFullyConnectedModule(
                self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )
            self.reverse_convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.reverse_fcn = BasicFullyConnectedModule(
                self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )

    def forward(self, x, x_rev_comp=None):
        x = self.convnet(x)
        if self.mode == "rbp":
            x = torch.cat((self.max_pool(x), self.avg_pool(x)), dim=1)
            x = x.view(x.size(0), self.convnet.out_channels * 2)
        elif self.mode == "dna":
            x = self.max_pool(x)
            x = x.view(x.size(0), self.convnet.out_channels)
        x = self.fcn(x)

        if self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat(
                    (self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1
                )
                x_rev_comp = x_rev_comp.view(
                    x_rev_comp.size(0), self.convnet.out_channels * 2
                )
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(
                    x_rev_comp.size(0), self.convnet.out_channels
                )
            x_rev_comp = self.fcn(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(
                    dim=1
                )
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat(
                    (self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1
                )
                x_rev_comp = x_rev_comp.view(
                    x_rev_comp.size(0), self.convnet.out_channels * 2
                )
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(
                    x_rev_comp.size(0), self.convnet.out_channels
                )
            x_rev_comp = self.reverse_fcn(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(
                    dim=1
                )
        return x

    def kwarg_handler(self, conv_kwargs, fc_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 16])
        conv_kwargs.setdefault("conv_kernels", [16])
        conv_kwargs.setdefault("pool_kernels", [8])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("dropout_rates", 0.25)
        conv_kwargs.setdefault("batchnorm", False)
        fc_kwargs.setdefault("hidden_dims", [32])
        fc_kwargs.setdefault("dropout_rate", 0.25)
        fc_kwargs.setdefault("batchnorm", False)

        return conv_kwargs, fc_kwargs


class DeepSEA(BaseModel):
    """DeepSEA model implementation for EUGENe
    Default parameters are those specified in the DeepSEA paper

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
        conv_kwargs: dict = {},
        fc_kwargs: dict = {},
    ):
        super().__init__(
            input_len, 
            output_dim, 
            strand, 
            task, 
            aggr
        )
        self.conv_kwargs, self.fc_kwargs = self.kwarg_handler(conv_kwargs, fc_kwargs)
        self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
        self.fcn = BasicFullyConnectedModule(
            input_dim=self.convnet.flatten_dim, output_dim=output_dim, **self.fc_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.convnet(x)
        x = x.view(x.size(0), self.convnet.flatten_dim)
        x = self.fcn(x)
        return x

    def kwarg_handler(self, conv_kwargs, fc_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 320, 480, 960])
        conv_kwargs.setdefault("conv_kernels", [8, 8, 8])
        conv_kwargs.setdefault("pool_kernels", [4, 4, 4])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("activation", "relu")
        conv_kwargs.setdefault("dropout_rates", [0.2, 0.2, 0.5])
        conv_kwargs.setdefault("batchnorm", False)
        fc_kwargs.setdefault("hidden_dims", [925])
        return conv_kwargs, fc_kwargs
