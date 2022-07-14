import torch
from ..models.base import BaseModel, BasicFullyConnectedModule, BasicConv1D, BasicRecurrent


class Hybrid(BaseModel):
    def __init__(self, input_len, output_dim, conv_kwargs, rnn_kwargs, strand="ss", task="regression", aggr=None, fc_kwargs={}):
        super().__init__(input_len, output_dim, strand, task, aggr)
        if self.strand == "ss":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.recurrentnet = BasicRecurrent(input_dim=self.convnet.out_channels, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.recurrentnet.out_dim, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ds":
            self.convnet = BasicConv1D(input_len=input_len,**conv_kwargs)
            self.recurrentnet = BasicRecurrent(input_dim=self.convnet.out_channels*2, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.recurrentnet.out_dim, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ts":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.reverse_convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.recurrentnet = BasicRecurrent(input_dim=self.convnet.out_channels*2, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.recurrentnet.out_dim, output_dim=output_dim, **fc_kwargs)

    def forward(self, x, x_rev_comp):
        x = self.convnet(x)
        x = x.transpose(1, 2)
        if self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.transpose(1, 2)
            x = torch.cat([x, x_rev_comp], dim=2)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.transpose(1, 2)
            x = torch.cat([x, x_rev_comp], dim=2)
        out, _ = self.recurrentnet(x)
        out = self.fcnet(out[:, -1, :])
        return out


if __name__ == "__main__":
    from ..dataloading.dataloaders import SeqDataModule
    from pytorch_lightning.utilities.cli import LightningCLI
    cli = LightningCLI(Hybrid, SeqDataModule, save_config_overwrite=True)
