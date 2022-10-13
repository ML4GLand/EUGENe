import torch
from .base import BaseModel, BasicFullyConnectedModule, BasicConv1D, BasicRecurrent


class FCN(BaseModel):
    """
    Instantiate a fully connected neural network with the specified layers and parameters.
    
    By default, this architecture flattens the one-hot encoded sequence and passes 
    it through a set of layers that are fully connected. The task defines how the output is
    treated (e.g. sigmoid activation for binary classification). The loss function is
    should be matched to the task (e.g. binary cross entropy ("bce") for binary classification).

    - If the model is single-stranded ("ss"), the input is passed through a single set of
        fully connected layers.
    - If the model is double-stranded ("ds"), the forward and reverse sequence are passed
        through the same set of fully connected layers. If aggr is "concat", the input
        forward and reverse sequences are concatenated and passed through a single set
        of fully connected layers. If aggr is "max" or "avg", the output of the forward and
        reverse sequence are passed through a single set of fully connected layers separately 
        and the maximum or average of the two outputs is taken.
    - If the model is twin-stranded ("ts"), the forward and reverse sequence are passed
        through different sets of fully connected layers. "concat" is not supported for
        this model type. If aggr is "max" or "avg", the output of the forward and reverse
        sequence are passed through a single set of fully connected layers and the maximum
        or average of the two outputs is taken.

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
    fc_kwargs:
        The keyword arguments for the fully connected layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        fc_kwargs: dict = {},
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
        self.flattened_input_dims = 4 * input_len
        if self.strand == "ss":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **fc_kwargs
            )
        elif self.strand == "ds":
            if self.aggr == "concat":
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.flattened_input_dims * 2,
                    output_dim=output_dim,
                    **fc_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.flattened_input_dims,
                    output_dim=output_dim,
                    **fc_kwargs
                )
        elif self.strand == "ts":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **fc_kwargs
            )
            self.reverse_fcn = BasicFullyConnectedModule(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **fc_kwargs
            )

    def forward(self, x, x_rev_comp=None):
        x = x.flatten(start_dim=1)
        if self.strand == "ss":
            x = self.fcn(x)
        elif self.strand == "ds":
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            if self.aggr == "concat":
                x = torch.cat((x, x_rev_comp), dim=1)
                x = self.fcn(x)
            elif self.aggr in ["max", "avg"]:
                x = self.fcn(x)
                x_rev_comp = self.fcn(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x = self.fcn(x)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = self.reverse_fcn(x_rev_comp)
            if self.aggr == "concat":
                raise ValueError("Concatenation is not supported for the tsFCN model.")
            elif self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x


class CNN(BaseModel):
    """
    Instantiate a CNN model with a set of convolutional layers and a set of fully
    connected layers.

    By default, this architecture passes the one-hot encoded sequence through a set
    1D convolutions with 4 channels. The task defines how the output is treated (e.g.
    sigmoid activation for binary classification). The loss function is should be matched
    to the task (e.g. binary cross entropy ("bce") for binary classification).

    - If the model is single-stranded ("ss"), the input is passed through a single set of
        convolutions to extract features. The extracted features are then flattened into a 
        single dimensional tensor and passed through a set of fully connected layers.
    - If the model is double-stranded ("ds"), the forward and reverse sequence are passed
        through the same set of convolutions to extract features. If aggr is "concat", the
        extracted features are concatenated and passed through a single set of fully connected
        layers. If aggr is "max" or "avg", the extracted features are passed through the same single
        set of fully connected layers separately and the maximum or average of the two outputs
        is taken.
    - If the model is twin-stranded ("ts"), the forward and reverse sequence are passed
        through different sets of convolutions to extract features. If aggr is "concat"
        the extracted features are concatenated and passed through a single set of fully
        connected layers. If aggr is "max" or "avg", the extracted features are passed through
        separate sets of fully connected layers and the maximum or average of the two outputs
        is taken.

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
        The aggregation function to use.
    fc_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the flattened output of the convolutional layers directly to 
        the output layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        fc_kwargs: dict = {},
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
        if self.strand == "ss":
            self.convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs
            )
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.convnet.flatten_dim, 
                output_dim=output_dim, 
                **fc_kwargs
            )
        elif self.strand == "ds":
            self.convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs
            )
            if self.aggr == "concat":
                self.fcn = BasicFullyConnectedModule(
                input_dim=self.convnet.flatten_dim * 2,
                output_dim=output_dim,
                **fc_kwargs
            )
            elif self.aggr in ["max", "avg"]:
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.convnet.flatten_dim,
                    output_dim=output_dim,
                    **fc_kwargs
                )
        elif self.strand == "ts":
            self.convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs
            )
            self.reverse_convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs
            )
            if aggr == "concat":
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.convnet.flatten_dim * 2,
                    output_dim=output_dim,
                    **fc_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.convnet.flatten_dim,
                    output_dim=output_dim,
                    **fc_kwargs
                )
                self.reverse_fcn = BasicFullyConnectedModule(
                    input_dim=self.reverse_convnet.flatten_dim,
                    output_dim=output_dim,
                    **fc_kwargs
                )

    def forward(self, x, x_rev_comp=None):
        x = self.convnet(x)
        x = x.view(x.size(0), self.convnet.flatten_dim)
        if self.strand == "ss":
            x = self.fcn(x)
        elif self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.convnet.flatten_dim)
            if self.aggr == "concat":
                x = torch.cat([x, x_rev_comp], dim=1)
                x = self.fcn(x)
            elif self.aggr in ["max", "avg"]:
                x = self.fcn(x)
                x_rev_comp = self.fcn(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.reverse_convnet.flatten_dim)
            if self.aggr == "concat":
                x = torch.cat([x, x_rev_comp], dim=1)
                x = self.fcn(x)
            elif self.aggr in ["max", "avg"]:
                x = self.fcn(x)
                x_rev_comp = self.reverse_fcn(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x


class RNN(BaseModel):
    """
    Instantiate an RNN model with a set of recurrent layers and a set of fully
    connected layers.

    By default, this model passes the one-hot encoded sequence through recurrent layers
    and then through a set of fully connected layers. The output of the fully connected
    layers is passed to the output layer.

    - If the model is single-stranded ("ss"), the sequence is passed through a single
        set of recurrent layers and a single set of fully connected layers.
    - If the model is double-stranded ("ds"), the sequence forward and reverse sequence
        are passed through the same set of recurrent layers to extract features. If aggr 
        is "concat", the output of the recurrent layers is concatenated and passed to a
        single set of fully connected layers. If aggr is "max" or "avg", the output of
        the recurrent layers is passed to the same single set of fully connected layers 
        separately and the max or average of the two outputs is passed to the output
        layer.
    - If the model is twin-stranded ("ts"), the sequence forward and reverse sequence
        are passed through separate sets of recurrent layers to extract features. If aggr
        is "concat", the output of the recurrent layers is concatenated and passed to a
        single set of fully connected layers. If aggr is "max" or "avg", the output of
        the recurrent layers is passed to the separate sets of fully connected layers
        separately and the max or average of the two outputs is passed to the output
        layer.

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
    fc_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the recurrent output of the recurrent layers directly to the
        output layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        rnn_kwargs: dict,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        fc_kwargs: dict = {},
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
        if self.strand == "ss":
            self.rnn = BasicRecurrent(
                input_dim=4, 
                **rnn_kwargs
                )
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.rnn.out_dim, 
                output_dim=output_dim, 
                **fc_kwargs
            )
        elif self.strand == "ds":
            self.rnn = BasicRecurrent(
                input_dim=4, 
                **rnn_kwargs
                )
            if self.aggr == "concat":
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.rnn.out_dim * 2, 
                    output_dim=output_dim, **fc_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.rnn.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
        elif self.strand == "ts":
            self.rnn = BasicRecurrent(
                input_dim=4, 
                **rnn_kwargs)
            self.reverse_rnn = BasicRecurrent(
                input_dim=4, 
                **rnn_kwargs)
            if self.aggr == "concat":
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.rnn.out_dim * 2, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.rnn.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
                self.reverse_fcn = BasicFullyConnectedModule(
                    input_dim=self.reverse_rnn.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )

    def forward(self, x, x_rev_comp=None):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        if self.strand == "ss":
            x = self.fcn(x)
        elif self.strand == "ds":
            x_rev_comp, _ = self.rnn(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            if self.aggr == "concat":
                x = torch.cat((x, x_rev_comp), dim=1)
                x = self.fcn(x)
            elif self.aggr in ["max", "avg"]:
                x = self.fcn(x)
                x_rev_comp = self.fcn(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x_rev_comp, _ = self.reverse_rnn(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            if self.aggr == "concat":
                x = torch.cat((x, x_rev_comp), dim=1)
                x = self.fcn(x)
            elif self.aggr in ["max", "avg"]:
                x = self.fcn(x)
                x_rev_comp = self.reverse_fcn(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x


class Hybrid(BaseModel):
    """
    A hybrid model that uses both a CNN and an RNN to extract features then passes the
    features through a set of fully connected layers.
    
    By default, the CNN is used to extract features from the input sequence, and the RNN is used to 
    to combine those features. The output of the RNN is passed to a set of fully connected
    layers to make the final prediction.

    - If the model is single-stranded ("ss"), the sequence is passed through the CNN then the RNN
        to extract features. The output of the RNN is passed to a set of fully connected layers.
    - If the model is double-stranded ("ds"), the sequence and reverse complement sequence are
        passed through the same CNN separately to extract features. If aggr is "concat_cnn", the output of
        the CNN is concatenated and passed to the same RNN. If aggr is "concat_rnn", the output of the
        same RNN is concatenated and passed to a set of fully connected layers. If aggr is "max" or "avg",
        the output of the RNN for each strand is passed to the separate sets of fully connected layers separately 
        and the max or average of the two outputs is passed to the output layer.
    - If the model is twin-stranded ("ts"), the sequence and reverse complement sequence are passed through separate models
        with identical architectures. If aggr is "concat_cnn", the outputs of the CNN are concatenated and passed to the same RNN
        and FCN. If aggr is "concat_rnn", the outputs of the RNN are concatenated and passed to the same FCN. 
        If aggr is "max" or "avg", the outputs of the RNN for each strand are passed to the separate sets of fully connected layers 
        separately and the max or average of the two outputs is passed to the output layer.


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
    fc_kwargs:
        The keyword arguments for the fully connected layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        rnn_kwargs: dict,
        strand: str = "ss",
        task: str = "regression",
        loss_fxn: str = "mse",
        aggr: str = None,
        fc_kwargs: dict = {},
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
        if self.strand == "ss":
            self.convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs)
            self.recurrentnet = BasicRecurrent(
                input_dim=self.convnet.out_channels, 
                **rnn_kwargs
            )
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.recurrentnet.out_dim, 
                output_dim=output_dim, 
                **fc_kwargs
            )
        elif self.strand == "ds":
            self.convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs
            )
            if aggr == "concat_cnn":
                self.recurrentnet = BasicRecurrent(
                    input_dim=self.convnet.out_channels * 2, 
                    **rnn_kwargs
                )
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.recurrentnet.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
            elif aggr == "concat_rnn":
                self.recurrentnet = BasicRecurrent(
                    input_dim=self.convnet.out_channels, 
                    **rnn_kwargs
                )
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.recurrentnet.out_dim * 2, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.recurrentnet = BasicRecurrent(
                    input_dim=self.convnet.out_channels, 
                    **rnn_kwargs
                )
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.recurrentnet.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
        elif self.strand == "ts":
            self.convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs
            )
            self.reverse_convnet = BasicConv1D(
                input_len=input_len, 
                **conv_kwargs
            )
            if aggr == "concat_cnn":
                self.recurrentnet = BasicRecurrent(
                    input_dim=self.convnet.out_channels * 2, 
                    **rnn_kwargs
                )
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.recurrentnet.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
            elif aggr == "concat_rnn":
                self.recurrentnet = BasicRecurrent(
                    input_dim=self.convnet.out_channels, 
                    **rnn_kwargs
                )
                self.reverse_recurrentnet = BasicRecurrent(
                    input_dim=self.reverse_convnet.out_channels, 
                    **rnn_kwargs
                )
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.recurrentnet.out_dim * 2, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.recurrentnet = BasicRecurrent(
                    input_dim=self.convnet.out_channels, 
                    **rnn_kwargs
                )
                self.reverse_recurrentnet = BasicRecurrent(
                    input_dim=self.reverse_convnet.out_channels, 
                    **rnn_kwargs
                )
                self.fcn = BasicFullyConnectedModule(
                    input_dim=self.recurrentnet.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )
                self.reverse_fcn = BasicFullyConnectedModule(
                    input_dim=self.reverse_recurrentnet.out_dim, 
                    output_dim=output_dim, 
                    **fc_kwargs
                )

    def forward(self, x, x_rev_comp=None):
        x = self.convnet(x)
        x = x.transpose(1, 2)
        if self.strand == "ss":
            out, _ = self.recurrentnet(x)
            out = self.fcn(out[:, -1, :])
        elif self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.transpose(1, 2)
            if self.aggr == "concat_cnn":
                x = torch.cat([x, x_rev_comp], dim=2)
                out, _ = self.recurrentnet(x)
                out = self.fcn(out[:, -1, :])
            elif self.aggr == "concat_rnn":
                out, _ = self.recurrentnet(x)
                out_rev_comp, _ = self.recurrentnet(x_rev_comp)
                out = torch.cat([out[:, -1, :], out_rev_comp[:, -1, :]], dim=1)
                out = self.fcn(out)
            elif self.aggr in ["max", "avg"]:
                out, _ = self.recurrentnet(x)
                out = self.fcn(out[:, -1, :]) 
                out_rev_comp, _ = self.recurrentnet(x_rev_comp)
                out_rev_comp = self.fcn(out_rev_comp[:, -1, :])
                if self.aggr == "max":
                    out = torch.max(out, out_rev_comp)
                elif self.aggr == "avg":
                    out = (out + out_rev_comp) / 2
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            x_rev_comp = x_rev_comp.transpose(1, 2)
            if self.aggr == "concat_cnn":
                x = torch.cat([x, x_rev_comp], dim=2)
                out, _ = self.recurrentnet(x)
                out = self.fcn(out[:, -1, :]) 
            elif self.aggr == "concat_rnn":
                out, _ = self.recurrentnet(x)
                out_rev_comp, _ = self.reverse_recurrentnet(x_rev_comp)
                out = torch.cat([out[:, -1, :], out_rev_comp[:, -1, :]], dim=1)
                out = self.fcn(out)
            elif self.aggr in ["max", "avg"]:
                out, _ = self.recurrentnet(x)
                out = self.fcn(out[:, -1, :]) 
                out_rev_comp, _ = self.reverse_recurrentnet(x_rev_comp)
                out_rev_comp = self.reverse_fcn(out_rev_comp[:, -1, :])
                if self.aggr == "max":
                    out = torch.max(out, out_rev_comp)
                elif self.aggr == "avg":
                    out = (out + out_rev_comp) / 2
        return out
