import torch
from .base import BaseModel, BasicFullyConnectedModule


class FCN(BaseModel):
    def __init__(self, input_len, output_dim, strand="ss", task="regression", aggr=None, fc_kwargs={}):
        """ Initialize the FCN model.
        Args:
            input_len: The length of the input sequence.
            output_dim: The dimension of the output.
            strand: The strand of the model.
            task: The task of the model.
            aggr: The aggregation function.
            fc_kwargs: The keyword arguments for the fully connected layer.
        """
        super().__init__(input_len, output_dim, strand, task, aggr)
        self.flattened_input_dims = 4*input_len
        if self.strand == "ss":
            self.fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ds":
            self.fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims*2, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ts":
            self.fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims, output_dim=output_dim, **fc_kwargs)
            self.reverse_fcn = BasicFullyConnectedModule(input_dim=self.flattened_input_dims, output_dim=output_dim, **fc_kwargs)

    def forward(self, x, x_rev_comp):
        x = x.flatten(start_dim=1)
        if self.strand == "ss":
            x = self.fcn(x)
        elif self.strand == "ds":
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x = torch.cat((x, x_rev_comp), dim=1)
            x = self.fcn(x)
        elif self.strand == "ts":
            x = self.fcn(x)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = self.reverse_fcn(x_rev_comp)
            x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x


if __name__ == "__main__":
    from ..dataloading.dataloaders import SeqDataModule
    from pytorch_lightning.utilities.cli import LightningCLI
    cli = LightningCLI(FCN, SeqDataModule, save_config_overwrite=True)
