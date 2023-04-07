import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import SequenceModel 
from .base import _layers as layers
from .base import _blocks as blocks
from .base import _towers as towers

import torch.nn as nn
class Inception(SequenceModel):
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        channels = [4, 64, 128, 256],
        kernel_size2: int = 4,
        kernel_size3: int = 8,
        conv_maxpool_kernel_size: int = 3,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
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
        self.channels = channels
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.conv_maxpool_kernel_size = conv_maxpool_kernel_size
        self.dense_kwargs = dense_kwargs
        conv_tower = nn.Sequential()
        for i in range(1, len(self.channels)):
            conv_tower.append(
                layers.InceptionConv1D(
                    in_channels=self.channels[i-1],
                    out_channels=self.channels[i],
                    kernel_size2=self.kernel_size2,
                    kernel_size3=self.kernel_size3,
                    conv_maxpool_kernel_size=self.conv_maxpool_kernel_size,
                )
            )
        self.conv_tower = nn.Sequential(*conv_tower)
        self.dense_block = blocks.DenseBlock(
            input_dim=self.channels[-1] * self.input_len,
            output_dim=self.output_dim,
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv_tower(x)
        x = x.view(x.shape[0], -1)
        x = self.dense_block(x)
        return x