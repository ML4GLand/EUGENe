import torch
from .base import BaseModel, BasicFullyConnectedModule, BasicConv1D, BasicRecurrent


class FCN(BaseModel):
    def __init__(
        self,
        input_len,
        output_dim,
        strand="ss",
        task="regression",
        aggr=None,
        loss_fxn="mse",
        fc_kwargs={},
        **kwargs
    ):
        """Initialize the FCN model.
        Args:
            input_len: The length of the input sequence.
            output_dim: The dimension of the output.
            strand: The strand of the model.
            task: The task of the model.
            aggr: The aggregation function.
            fc_kwargs: The keyword arguments for the fully connected layer.
        """
        super().__init__(input_len, output_dim, strand, task, aggr, loss_fxn, **kwargs)
        self.flattened_input_dims = 4 * input_len
        if self.strand == "ss":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.flattened_input_dims, output_dim=output_dim, **fc_kwargs
            )
        elif self.strand == "ds":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.flattened_input_dims * 2,
                output_dim=output_dim,
                **fc_kwargs
            )
        elif self.strand == "ts":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.flattened_input_dims, output_dim=output_dim, **fc_kwargs
            )
            self.reverse_fcn = BasicFullyConnectedModule(
                input_dim=self.flattened_input_dims, output_dim=output_dim, **fc_kwargs
            )

    def forward(self, x, x_rev_comp=None):
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


class CNN(BaseModel):
    def __init__(
        self,
        input_len,
        output_dim,
        conv_kwargs,
        strand="ss",
        task="regression",
        aggr=None,
        loss_fxn="mse",
        fc_kwargs={},
        **kwargs
    ):
        """Initialize the CNN model.
        Args:
            input_len: The length of the input sequence.
            output_dim: The dimension of the output.
            conv_kwargs: The keyword arguments for the convolutional layer.
            strand: The strand of the model.
            task: The task of the model.
            aggr: The aggregation function.
            fc_kwargs: The keyword arguments for the fully connected layer.
        """
        super().__init__(input_len, output_dim, strand, task, aggr, loss_fxn, **kwargs)
        if self.strand == "ss":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.convnet.flatten_dim, output_dim=output_dim, **fc_kwargs
            )
        elif self.strand == "ds":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.convnet.flatten_dim * 2,
                output_dim=output_dim,
                **fc_kwargs
            )
        elif self.strand == "ts":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.reverse_convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.convnet.flatten_dim * 2,
                output_dim=output_dim,
                **fc_kwargs
            )

    def forward(self, x, x_rev_comp=None):
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


class RNN(BaseModel):
    def __init__(
        self,
        input_len,
        output_dim,
        rnn_kwargs,
        strand="ss",
        task="regression",
        aggr=None,
        fc_kwargs={},
    ):
        """Initialize the RNN model.
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
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.rnn.out_dim, output_dim=output_dim, **fc_kwargs
            )
        elif self.strand == "ds":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.rnn.out_dim * 2, output_dim=output_dim, **fc_kwargs
            )
        elif self.strand == "ts":
            self.rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.reverse_rnn = BasicRecurrent(input_dim=4, **rnn_kwargs)
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.rnn.out_dim * 2, output_dim=output_dim, **fc_kwargs
            )

    def forward(self, x, x_rev_comp=None):
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


class Hybrid(BaseModel):
    """
    The hybrid model.
    Args:
        input_len: The length of the input sequence.
        output_dim: The dimension of the output.
        conv_kwargs: The keyword arguments for the convolutional layer.
        rnn_kwargs: The keyword arguments for the RNN layer.
        strand: The strand of the model.
        task: The task of the model.
        aggr: The aggregation function.
        fc_kwargs: The keyword arguments for the fully connected layer.
    """

    def __init__(
        self,
        input_len,
        output_dim,
        conv_kwargs,
        rnn_kwargs,
        strand="ss",
        task="regression",
        loss_fxn="mse",
        aggr=None,
        fc_kwargs={},
        **kwargs
    ):
        super().__init__(input_len, output_dim, strand, task, aggr, loss_fxn, **kwargs)
        if self.strand == "ss":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.recurrentnet = BasicRecurrent(
                input_dim=self.convnet.out_channels, **rnn_kwargs
            )
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.recurrentnet.out_dim, output_dim=output_dim, **fc_kwargs
            )
        elif self.strand == "ds":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.recurrentnet = BasicRecurrent(
                input_dim=self.convnet.out_channels * 2, **rnn_kwargs
            )
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.recurrentnet.out_dim, output_dim=output_dim, **fc_kwargs
            )
        elif self.strand == "ts":
            self.convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.reverse_convnet = BasicConv1D(input_len=input_len, **conv_kwargs)
            self.recurrentnet = BasicRecurrent(
                input_dim=self.convnet.out_channels * 2, **rnn_kwargs
            )
            self.fcnet = BasicFullyConnectedModule(
                input_dim=self.recurrentnet.out_dim, output_dim=output_dim, **fc_kwargs
            )

    def forward(self, x, x_rev_comp=None):
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
