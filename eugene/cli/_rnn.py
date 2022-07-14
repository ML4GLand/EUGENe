import torch
from ..models.base import BaseModel, BasicFullyConnectedModule, BasicRecurrent


class RNN(BaseModel):
    def __init__(self, input_len, output_dim, rnn_kwargs, strand="ss", task="regression", aggr=None, fc_kwargs={}):
        """ Initialize the RNN model.
        Args:
            input_len: The length of the input sequence.
            output_dim: The dimension of the output.
            rnn_kwargs: The keyword arguments for the RNN layer.
            strand: The strand of the model.
            task: The task of the model.
            aggr: The aggregation function.
            fc_kwargs: The keyword arguments for the fully connected layer.
        """
        super().__init__(input_len, output_dim, strand, task, aggr)
        if self.strand == "ss":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.rnn.out_dim, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ds":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.rnn.out_dim*2, output_dim=output_dim, **fc_kwargs)
        elif self.strand == "ts":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.reverse_rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(input_dim=self.rnn.out_dim*2, output_dim=output_dim, **fc_kwargs)

    def forward(self, x, x_rev_comp):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        if self.strand == "ds":
            x_rev_comp, _ = self.rnn(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            x = torch.cat((x, x_rev_comp), dim=1)
        elif self.strand == "ts":
            x_rev_comp, _ = self.reverse_rnn(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            x = torch.cat((x, x_rev_comp), dim=1)
        x = self.fcnet(x)
        return x


if __name__ == "__main__":
    from ..dataloading.dataloaders import SeqDataModule
    from pytorch_lightning.utilities.cli import LightningCLI
    cli = LightningCLI(RNN, SeqDataModule, save_config_overwrite=True)
