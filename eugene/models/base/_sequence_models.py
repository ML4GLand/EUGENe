import torch
from ._base_models import SequenceModel
from ._blocks import DenseBlock, Conv1DBlock, RecurrentBlock


class FCN(SequenceModel):
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
    :dense_kwargs
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
        self.flattened_input_dims = 4 * input_len
        if self.strand == "ss":
            self.dense_block = DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            if self.aggr == "concat":
                self.dense_block = DenseBlock(
                    input_dim=self.flattened_input_dims * 2,
                    output_dim=output_dim,
                    **dense_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = DenseBlock(
                    input_dim=self.flattened_input_dims,
                    output_dim=output_dim,
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.dense_block = DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )
            self.reverse_dense = DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )

    def forward(self, x, x_rev_comp=None):
        x = x.flatten(start_dim=1)
        if self.strand == "ss":
            x = self.dense_block(x)
        elif self.strand == "ds":
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            if self.aggr == "concat":
                x = torch.cat((x, x_rev_comp), dim=1)
                x = self.dense_block(x)
            elif self.aggr in ["max", "avg"]:
                x = self.dense_block(x)
                x_rev_comp = self.dense_block(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x = self.dense_block(x)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = self.reverse_dense(x_rev_comp)
            if self.aggr == "concat":
                raise ValueError("Concatenation is not supported for the tsdense model.")
            elif self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x


class CNN(SequenceModel):
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
    dense_kwargs:
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
        if self.strand == "ss":
            self.conv1d_block = Conv1DBlock(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            self.dense_block = DenseBlock(
                input_dim=self.conv1d_block.flatten_dim, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            self.conv1d_block = Conv1DBlock(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if self.aggr == "concat":
                self.dense_block = DenseBlock(
                    input_dim=self.conv1d_block.flatten_dim * 2,
                    output_dim=output_dim,
                    **dense_kwargs
            )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = DenseBlock(
                    input_dim=self.conv1d_block.flatten_dim,
                    output_dim=output_dim,
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.conv1d_block = Conv1DBlock(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            self.reverse_conv1d_block = Conv1DBlock(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if aggr == "concat":
                self.dense_block = DenseBlock(
                    input_dim=self.conv1d_block.flatten_dim * 2,
                    output_dim=output_dim,
                    **dense_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.dense_block = DenseBlock(
                    input_dim=self.conv1d_block.flatten_dim,
                    output_dim=output_dim,
                    **dense_kwargs
                )
                self.reverse_dense = DenseBlock(
                    input_dim=self.reverse_conv1d_block.flatten_dim,
                    output_dim=output_dim,
                    **dense_kwargs 
                )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_block(x)
        x = x.view(x.size(0), self.conv1d_block.flatten_dim)
        if self.strand == "ss":
            x = self.dense_block(x)
        elif self.strand == "ds":
            x_rev_comp = self.conv1d_block(x_rev_comp)
            x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_block.flatten_dim)
            if self.aggr == "concat":
                x = torch.cat([x, x_rev_comp], dim=1)
                x = self.dense_block(x)
            elif self.aggr in ["max", "avg"]:
                x = self.dense_block(x)
                x_rev_comp = self.dense_block(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_conv1d_block(x_rev_comp)
            x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.reverse_conv1d_block.flatten_dim)
            if self.aggr == "concat":
                x = torch.cat([x, x_rev_comp], dim=1)
                x = self.dense_block(x)
            elif self.aggr in ["max", "avg"]:
                x = self.dense_block(x)
                x_rev_comp = self.reverse_dense(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x

class RNN(SequenceModel):
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
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the recurrent output of the recurrent layers directly to the
        output layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        recurrent_kwargs: dict,
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
        if self.strand == "ss":
            self.recurrent_block = RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs
            )
            self.dense_block = DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            self.recurrent_block = RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs
            )
            if self.aggr == "concat":
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.recurrent_block = RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs)
            self.reverse_rnn = RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs)
            if self.aggr == "concat":
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
                self.reverse_dense_block = DenseBlock(
                    input_dim=self.reverse_rnn.out_dim, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )

    def forward(self, x, x_rev_comp=None):
        x, _ = self.recurrent_block(x)
        x = x[:, -1, :]
        if self.strand == "ss":
            x = self.dense_block(x)
        elif self.strand == "ds":
            x_rev_comp, _ = self.recurrent_block(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            if self.aggr == "concat":
                x = torch.cat((x, x_rev_comp), dim=1)
                x = self.dense_block(x)
            elif self.aggr in ["max", "avg"]:
                x = self.dense_block(x)
                x_rev_comp = self.dense_block(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x_rev_comp, _ = self.reverse_rnn(x_rev_comp)
            x_rev_comp = x_rev_comp[:, -1, :]
            if self.aggr == "concat":
                x = torch.cat((x, x_rev_comp), dim=1)
                x = self.dense_block(x)
            elif self.aggr in ["max", "avg"]:
                x = self.dense_block(x)
                x_rev_comp = self.reverse_dense_block(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x


class Hybrid(SequenceModel):
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
        and FC. If aggr is "concat_rnn", the outputs of the RNN are concatenated and passed to the same FCN. 
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
    dense_kwargs:
        The keyword arguments for the fully connected layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        recurrent_kwargs: dict,
        strand: str = "ss",
        task: str = "regression",
        loss_fxn: str = "mse",
        aggr: str = None,
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
        if self.strand == "ss":
            self.conv1d_block = Conv1DBlock(
                input_len=input_len,
                input_channels=4, 
                **conv_kwargs
            )
            self.recurrent_block = RecurrentBlock(
                input_dim=self.conv1d_block.out_channels, 
                **recurrent_kwargs
            )
            self.dense_block = DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            self.conv1d_block = Conv1DBlock(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if aggr == "concat_cnn":
                self.recurrent_block = RecurrentBlock(
                    input_dim=self.conv1d_block.out_channels * 2, 
                    **recurrent_kwargs
                )
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr == "concat_rnn":
                self.recurrent_block = RecurrentBlock(
                    input_dim=self.conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.recurrent_block = RecurrentBlock(
                    input_dim=self.conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.conv1d_block = Conv1DBlock(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            self.reverse_conv1d_block = Conv1DBlock(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if aggr == "concat_cnn":
                self.recurrent_block = RecurrentBlock(
                    input_dim=self.conv1d_block.out_channels * 2, 
                    **recurrent_kwargs
                )
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr == "concat_rnn":
                self.recurrent_block = RecurrentBlock(
                    input_dim=self.conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.reverse_recurrent_block = RecurrentBlock(
                    input_dim=self.reverse_conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.recurrent_block = RecurrentBlock(
                    input_dim=self.conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.reverse_recurrent_block = RecurrentBlock(
                    input_dim=self.reverse_conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
                self.reverse_dense_block = DenseBlock(
                    input_dim=self.reverse_recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_block(x)
        x = x.transpose(1, 2)
        if self.strand == "ss":
            out, _ = self.recurrent_block(x)
            out = self.dense_block(out[:, -1, :])
        elif self.strand == "ds":
            x_rev_comp = self.conv1d_block(x_rev_comp)
            x_rev_comp = x_rev_comp.transpose(1, 2)
            if self.aggr == "concat_cnn":
                x = torch.cat([x, x_rev_comp], dim=2)
                out, _ = self.recurrent_block(x)
                out = self.dense_block(out[:, -1, :])
            elif self.aggr == "concat_rnn":
                out, _ = self.recurrent_block(x)
                out_rev_comp, _ = self.recurrent_block(x_rev_comp)
                out = torch.cat([out[:, -1, :], out_rev_comp[:, -1, :]], dim=1)
                out = self.dense_block(out)
            elif self.aggr in ["max", "avg"]:
                out, _ = self.recurrent_block(x)
                out = self.dense_block(out[:, -1, :]) 
                out_rev_comp, _ = self.recurrent_block(x_rev_comp)
                out_rev_comp = self.dense_block(out_rev_comp[:, -1, :])
                if self.aggr == "max":
                    out = torch.max(out, out_rev_comp)
                elif self.aggr == "avg":
                    out = (out + out_rev_comp) / 2
        elif self.strand == "ts":
            x_rev_comp = self.reverse_conv1d_block(x_rev_comp)
            x_rev_comp = x_rev_comp.transpose(1, 2)
            if self.aggr == "concat_cnn":
                x = torch.cat([x, x_rev_comp], dim=2)
                out, _ = self.recurrent_block(x)
                out = self.dense_block(out[:, -1, :]) 
            elif self.aggr == "concat_rnn":
                out, _ = self.recurrent_block(x)
                out_rev_comp, _ = self.reverse_recurrent_block(x_rev_comp)
                out = torch.cat([out[:, -1, :], out_rev_comp[:, -1, :]], dim=1)
                out = self.dense_block(out)
            elif self.aggr in ["max", "avg"]:
                out, _ = self.recurrent_block(x)
                out = self.dense_block(out[:, -1, :]) 
                out_rev_comp, _ = self.reverse_recurrent_block(x_rev_comp)
                out_rev_comp = self.reverse_dense_block(out_rev_comp[:, -1, :])
                if self.aggr == "max":
                    out = torch.max(out, out_rev_comp)
                elif self.aggr == "avg":
                    out = (out + out_rev_comp) / 2
        return out