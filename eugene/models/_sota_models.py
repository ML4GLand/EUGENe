import torch
import torch.nn as nn
from .base import BaseModel, BasicFullyConnectedModule, BasicConv1D


class DeepBind(BaseModel):
    def __init__(self, input_len, output_dim, strand="ss", task="regression", aggr=None, mp_kwargs = {}, conv_kwargs = {}, fc_kwargs = {}):
        super().__init__(input_len, output_dim, strand, task, aggr)
        self.flattened_input_dims = 8*input_len
        self.mp_kwargs, self.conv_kwargs, self.fc_kwargs = self.kwarg_handler(mp_kwargs, conv_kwargs, fc_kwargs)
        self.max_pool = nn.MaxPool1d(**self.mp_kwargs)
        self.avg_pool = nn.AvgPool1d(**self.mp_kwargs)

        # Add strand specific modules
        if self.strand == "ss":
            self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.fcn = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim//(mp_kwargs.get("kernel_size")//2), output_dim=output_dim, **self.fc_kwargs)
        elif self.strand == "ds":
            self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.fcn = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim//(mp_kwargs.get("kernel_size")//4), output_dim=output_dim, **self.fc_kwargs)
        elif self.strand == "ts":
            self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.fcn = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim//(mp_kwargs.get("kernel_size")//2), output_dim=output_dim, **self.fc_kwargs)
            self.reverse_convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
            self.reverse_fcn = BasicFullyConnectedModule(input_dim=self.reverse_convnet.flatten_dim//(mp_kwargs.get("kernel_size")//2), output_dim=output_dim, **self.fc_kwargs)

    def forward(self, x, x_rev_comp = None):
        x = self.convnet(x)
        x = x.view(x.size(0), self.convnet.flatten_dim)
        x = torch.cat((self.max_pool(x), self.avg_pool(x)), dim=1)
        if self.strand == "ss":
            x = self.fcn(x)
        elif self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
            x = torch.cat((x, x_rev_comp), dim=1)
            x = self.fcn(x)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
            x = self.fcn(x)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = self.reverse_fcn(x_rev_comp)
            x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x

    # Sets default kwargs if not specified
    def kwarg_handler(self, mp_kwargs, conv_kwargs, fc_kwargs):
        # Add mp_kwargs for stride
        mp_kwargs.setdefault("kernel_size", 4)

        # Add conv_kwargs for stride
        conv_kwargs.setdefault("channels", [4, 16])
        conv_kwargs.setdefault("conv_kernels", [4])
        conv_kwargs.setdefault("pool_kernels", [4])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("dropout_rates", 0.2)
        conv_kwargs.setdefault("batchnorm", False)

        # Add fc_kwargs
        #fc_kwargs.setdefault("output_dim", 1)
        fc_kwargs.setdefault("hidden_dims", [256, 64, 16, 4])
        fc_kwargs.setdefault("dropout_rate", 0.2)
        fc_kwargs.setdefault("batchnorm", False)

        return mp_kwargs, conv_kwargs, fc_kwargs


class DeepSEA(BaseModel):
    def __init__(self, input_len=1000, output_dim=1, strand="ss", task="regression", aggr=None, conv_kwargs = {}, fc_kwargs = {}):
        """
        Generates a PyTorch module with architecture matching the convnet part of DeepSea. Default parameters are those specified in the DeepSea paper
        Parameters
        ----------
        input_len : int, input sequence length
        channels : list-like or int, channel width for each conv layer. If int each of the three layers will be the same channel width
        conv_kernels : list-like or int, conv kernel size for each conv layer. If int will be the same for all conv layers
        pool_kernels : list-like or int, maxpooling kernel size for the first two conv layers. If int will be the same for all conv layers
        dropout_rates : list-like or float, dropout rates for each conv layer. If int will be the same for all conv layers
        """
        super().__init__(input_len, output_dim, strand, task, aggr)
        self.conv_kwargs, self.fc_kwargs = self.kwarg_handler(conv_kwargs, fc_kwargs)
        self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
        self.fcn = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim, output_dim=output_dim, **self.fc_kwargs)

    def forward(self, x, x_rev_comp = None):
        x = self.convnet(x)
        x = x.view(x.size(0), self.convnet.flatten_dim)
        x = self.fcn(x)
        return x

    def kwarg_handler(self, conv_kwargs, fc_kwargs):
        conv_kwargs.setdefault("channels", [4, 320, 480, 960])
        conv_kwargs.setdefault("conv_kernels", [8, 8, 8])
        conv_kwargs.setdefault("pool_kernels", [4, 4, 4])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("activation", "relu")
        conv_kwargs.setdefault("dropout_rates", [0.2, 0.2, 0.5])
        conv_kwargs.setdefault("batchnorm", False)
        fc_kwargs.setdefault("hidden_dims", [925])
        return conv_kwargs, fc_kwargs
