import torch
from ..models.base import BaseModel, BasicFullyConnectedModule, BasicConv1D


class CNN(BaseModel):
    def __init__(self, input_len, output_dim, conv_kwargs, strand="ss", task="regression", aggr=None, fc_kwargs={}):
        """ Initialize the CNN model.
        Args:
            input_len: The length of the input sequence.
            output_dim: The dimension of the output.
            conv_kwargs: The keyword arguments for the convolutional layer.
            strand: The strand of the model.
            task: The task of the model.
            aggr: The aggregation function.
            fc_kwargs: The keyword arguments for the fully connected layer.
        """
        super().__init__(input_len, output_dim, strand, task, aggr)
        if self.strand == "ss":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ds":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim*2, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ts":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.reverse_convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.convnet.flatten_dim*2, output_dim=output_dim, **fc_kwargs)

    def forward(self, x, x_rev_comp):
        x = self.convnet(x)
        x = x.view(x.size(0), self.convnet.flatten_dim)
        if self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.convnet.flatten_dim)
            x = torch.cat([x, x_rev_comp], dim=1)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.convnet.flatten_dim)
            x = torch.cat([x, x_rev_comp], dim=1)
        x = self.fcnet(x)
        return x


if __name__ == "__main__":
    from ..dataloading.dataloaders import SeqDataModule
    from pytorch_lightning.utilities.cli import LightningCLI
    cli = LightningCLI(CNN, SeqDataModule, save_config_overwrite=True)
