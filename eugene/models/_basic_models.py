import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import SequenceModel 
from .base import _layers as layers
from .base import _blocks as blocks
from .base import _towers as towers


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
            self.dense_block = blocks.DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            if self.aggr == "concat":
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.flattened_input_dims * 2,
                    output_dim=output_dim,
                    **dense_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.flattened_input_dims,
                    output_dim=output_dim,
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.dense_block = blocks.DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )
            self.reverse_dense = blocks.DenseBlock(
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
            self.conv1d_tower = towers.Conv1DTower(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.conv1d_tower.flatten_dim, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            self.conv1d_tower = towers.Conv1DTower(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if self.aggr == "concat":
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.conv1d_tower.flatten_dim * 2,
                    output_dim=output_dim,
                    **dense_kwargs
            )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.conv1d_tower.flatten_dim,
                    output_dim=output_dim,
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.conv1d_tower = towers.Conv1DTower(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            self.reverse_conv1d_block = towers.Conv1DTower(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if aggr == "concat":
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.conv1d_tower.flatten_dim * 2,
                    output_dim=output_dim,
                    **dense_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.conv1d_tower.flatten_dim,
                    output_dim=output_dim,
                    **dense_kwargs
                )
                self.reverse_dense = blocks.DenseBlock(
                    input_dim=self.reverse_conv1d_block.flatten_dim,
                    output_dim=output_dim,
                    **dense_kwargs 
                )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)
        if self.strand == "ss":
            x = self.dense_block(x)
        elif self.strand == "ds":
            x_rev_comp = self.conv1d_tower(x_rev_comp)
            x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_tower.flatten_dim)
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
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs
            )
            if self.aggr == "concat":
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs)
            self.reverse_rnn = blocks.RecurrentBlock(
                input_dim=4, 
                **recurrent_kwargs)
            if self.aggr == "concat":
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
                self.reverse_dense_block = blocks.DenseBlock(
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
            self.conv1d_tower = towers.Conv1DTower(
                input_len=input_len,
                input_channels=4, 
                **conv_kwargs
            )
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=self.conv1d_tower.out_channels, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            self.conv1d_tower = towers.Conv1DTower(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if aggr == "concat_cnn":
                self.recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.conv1d_tower.out_channels * 2, 
                    **recurrent_kwargs
                )
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr == "concat_rnn":
                self.recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.conv1d_tower.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.conv1d_tower.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.conv1d_tower = towers.Conv1DTower(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            self.reverse_conv1d_block = towers.Conv1DTower(
                input_len=input_len, 
                input_channels=4,
                **conv_kwargs
            )
            if aggr == "concat_cnn":
                self.recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.conv1d_tower.out_channels * 2, 
                    **recurrent_kwargs
                )
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr == "concat_rnn":
                self.recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.conv1d_tower.out_channels, 
                    **recurrent_kwargs
                )
                self.reverse_recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.reverse_conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels * 2, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
            elif aggr in ["max", "avg"]:
                self.recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.conv1d_tower.out_channels, 
                    **recurrent_kwargs
                )
                self.reverse_recurrent_block = blocks.RecurrentBlock(
                    input_dim=self.reverse_conv1d_block.out_channels, 
                    **recurrent_kwargs
                )
                self.dense_block = blocks.DenseBlock(
                    input_dim=self.recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )
                self.reverse_dense_block = blocks.DenseBlock(
                    input_dim=self.reverse_recurrent_block.out_channels, 
                    output_dim=output_dim, 
                    **dense_kwargs
                )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_tower(x)
        x = x.transpose(1, 2)
        if self.strand == "ss":
            out, _ = self.recurrent_block(x)
            out = self.dense_block(out[:, -1, :])
        elif self.strand == "ds":
            x_rev_comp = self.conv1d_tower(x_rev_comp)
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

class TutorialCNN( SequenceModel):
    """Tutorial CNN model

    This is a very simple one layer convolutional model for testing purposes. It is featured in testing and tutorial
    notebooks.

    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    strand : str, optional
        Strand of the input. Only ss is supported for this model
    task : str, optional
        Task of the model. Either "regression" or "classification".
    aggr : str, optional
        Aggregation method. This model only supports "avg"
    loss_fxn : str, optional
        Loss function.
    **kwargs
        Keyword arguments to pass to the  SequenceModel class.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = "avg",
        loss_fxn: str = "mse",
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
        self.conv1 = nn.Conv1d(4, 30, 21)
        self.dense = nn.Linear(30, output_dim)
            
            
    def forward(self, x, x_rev_comp=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, x.size()[-1]).flatten(1, -1)
        x = self.dense(x)
        if self.strand == "ds":
            x_rev_comp = F.relu(self.conv1(x_rev_comp))
            x_rev_comp = F.max_pool1d(x_rev_comp, x_rev_comp.size()[-1]).flatten(1, -1)
            x_rev_comp = self.dense(x_rev_comp)
            x = (x + x_rev_comp / 2)
        return x

class Jores21CNN(SequenceModel):
    """
    Custom convolutional model used in Jores et al. 2021 paper

    PyTorch implementation of the TensorFlow model described here:
    https://github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters

    This model only uses a single strand, but applies convolutions and the reverse complement of the convolutional fitler
    to the same sequence.

    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    strand : str, optional
        Strand of the input. Only ss is supported for this model
    task : str, optional
        Task of the model. Either "regression" or "classification".
    aggr : str, optional
        Aggregation method. Does not apply to this model and will be ignored
    filters : int, optional
        Number of filters in the convolutional layers.
    kernel_size : int, optional
        Kernel size of the convolutional layers.
    layers : int, optional
        Number of convolutional layers.
    stride : int, optional
        Stride of the convolutional layers.
    dropout : float, optional
        Dropout probability.
    hidden_dim : int, optional
        Dimension of the hidden layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        filters: int = 128,
        kernel_size: int = 13,
        layers: int = 2,
        stride: int = 1,
        dropout: float = 0.15,
        hidden_dim: int = 64,
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
        self.biconv = towers.BiConv1DTower(
            filters=filters,
            kernel_size=kernel_size,
            layers=layers,
            stride=stride,
            dropout_rate=dropout,
        )
        self.conv = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=input_len * filters, out_features=hidden_dim)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x, x_rev_comp=None):
        x = self.biconv(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Kopp21CNN(SequenceModel):
    """
    Custom convolutional model used in Kopp et al. 2021 paper

    PyTorch implementation of the TensorFlow model described here:
    https://github.com/wkopp/janggu_usecases/tree/master/01_jund_prediction

    This model can only be run in "ds" mode. The reverse complement must be included in the Dataloader
    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    strand : str, optional
        Strand of the input. This model is only implemented for "ds"
    task : str, optional
        Task for this model. By default "binary_classification" for this mode
    aggr : str, optional
        Aggregation method. Either "concat", "max", or "avg". By default "max" for this model.
    filters : list, optional
        Number of filters in the convolutional layers. 
    conv_kernel_size : list, optional
        Kernel size of the convolutional layers.
    maxpool_kernel_size : int, optional
        Kernel size of the maxpooling layer.
    stride : int, optional
        Stride of the convolutional layers.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str ="ds",
        task: str = "binary_classification",
        aggr: str = "max",
        loss_fxn: str = "bce",
        filters: list = [10, 8],
        conv_kernel_size: list = [11, 3],
        maxpool_kernel_size: int = 30,
        stride: int = 1,
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
        self.conv = nn.Conv1d(4, filters[0], conv_kernel_size[0], stride=stride)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm1d(filters[0])
        self.conv2 = nn.Conv1d(filters[0], filters[1], conv_kernel_size[1], stride=stride)
        self.batchnorm2 = nn.BatchNorm1d(filters[1])
        self.linear = nn.Linear(filters[1], self.output_dim)

    def forward(self, x, x_rev_comp):
        x_fwd = F.relu(self.conv(x))
        x_rev_comp = F.relu(self.conv(x_rev_comp))
        if self.aggr == "concat":
            x = torch.cat((x_fwd, x_rev_comp), dim=2)
        elif self.aggr == "max":
            x = torch.max(x_fwd, x_rev_comp)
        elif self.aggr == "avg":
            x = (x_fwd + x_rev_comp) / 2
        elif self.aggr is None:
            x = torch.cat((x_fwd, x_rev_comp), dim=1)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])
        x = self.batchnorm2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class FactorizedBasset(SequenceModel):
	def __init__(
		self, 
		input_len: int = 1000,
		output_dim = 1, 
		strand = "ss",
		task = "binary_classification",
		aggr = None,
		loss_fxn = "bce",
		conv1_kwargs = {},
		conv2_kwargs = {},
		conv3_kwargs = {},
		maxpool_kernels = None,
		dense_kwargs = {},
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
		self.conv1_kwargs, self.conv2_kwargs, self.conv3_kwargs, self.maxpool_kernels, self.dense_kwargs = self.kwarg_handler(
			conv1_kwargs, 
			conv2_kwargs, 
			conv3_kwargs, 
			maxpool_kernels, 
		    dense_kwargs	
		)
		self.conv1d_tower1 = towers.Conv1DTower(
			input_len=input_len, 
            input_channels=4,
			**self.conv1_kwargs
		)
		self.conv1d_tower2 = towers.Conv1DTower(
			input_len=self.conv1d_tower1.output_len,
            input_channels=self.conv1d_tower1.out_channels,
			**self.conv2_kwargs
		)
		self.conv1d_tower3 = towers.Conv1DTower(
			input_len=self.conv1d_tower2.output_len,
            input_channels=self.conv1d_tower2.out_channels,
			**self.conv3_kwargs
		)
		self.dense_block = blocks.DenseBlock(
			input_dim=self.conv1d_tower3.flatten_dim,
			output_dim=output_dim, 
			**self.dense_kwargs
		)

	def forward(self, x, x_rev_comp=None):
		x = self.conv1d_tower1(x)
		x = self.conv1d_tower2(x)
		x = self.conv1d_tower3(x)
		x = x.view(x.size(0), self.conv1d_tower3.flatten_dim)
		x = self.dense_block(x)
		return x
        
	def kwarg_handler(self, conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels, dense_kwargs):
		"""Sets default kwargs FactorizedBasset"""
		conv1_kwargs.setdefault("conv_channels", [48, 64, 100, 150, 300])
		conv1_kwargs.setdefault("conv_kernels", [3, 3, 3, 7, 7])
		conv1_kwargs.setdefault("conv_strides", [1, 1, 1, 1, 1])
		conv1_kwargs.setdefault("conv_padding", [1, 1, 1, 3, 3])
		conv1_kwargs.setdefault("pool_types", [None, None, None, None, "max"])
		conv1_kwargs.setdefault("pool_kernels", [None, None, None, None, 3])
		conv1_kwargs.setdefault("dropout_rates", 0.0)
		conv1_kwargs.setdefault("batchnorm", True)
		conv1_kwargs.setdefault("batchnorm_first", True)
		conv1_kwargs.setdefault("activations", "relu")
		conv2_kwargs.setdefault("conv_channels", [200, 200, 200])
		conv2_kwargs.setdefault("conv_kernels", [7, 3, 3])
		conv2_kwargs.setdefault("conv_strides", [1, 1, 1])
		conv2_kwargs.setdefault("conv_padding", [3, 1, 1])
		conv2_kwargs.setdefault("pool_types", [None, None, "max"])
		conv2_kwargs.setdefault("pool_kernels", [None, None, 4])
		conv2_kwargs.setdefault("dropout_rates", 0.0)
		conv2_kwargs.setdefault("batchnorm", True)
		conv2_kwargs.setdefault("batchnorm_first", True)
		conv2_kwargs.setdefault("activations", "relu")
		conv3_kwargs.setdefault("conv_channels", [200])
		conv3_kwargs.setdefault("conv_kernels", [7])
		conv3_kwargs.setdefault("conv_strides", [1])
		conv3_kwargs.setdefault("conv_padding", [3])
		conv3_kwargs.setdefault("pool_types", ["max"])
		conv3_kwargs.setdefault("pool_kernels", [4])
		conv3_kwargs.setdefault("dropout_rates", 0.0)
		conv3_kwargs.setdefault("batchnorm", True)
		conv3_kwargs.setdefault("batchnorm_first", True)
		conv3_kwargs.setdefault("activations", "relu")
		dense_kwargs.setdefault("hidden_dims", [1000, 164])
		dense_kwargs.setdefault("dropout_rates", 0.0)
		dense_kwargs.setdefault("batchnorm", True)
		dense_kwargs.setdefault("batchnorm_first", True)
		dense_kwargs.setdefault("activations", "relu")
		return conv1_kwargs, conv2_kwargs, conv3_kwargs, maxpool_kernels,dense_kwargs 

class ResidualBind( SequenceModel):
    def __init__(
        self,
        input_len,
        output_dim,
        strand="ss",
        task="regression",
        aggr=None,
        input_chanels=4,
        conv_channels=[96],
        conv_kernel_size=[11],
        conv_stride_size=[1],
        conv_dilation_rate=[1],
        conv_padding="valid",
        conv_activation="relu",
        conv_batchnorm=True,
        conv_batchnorm_first=True,
        conv_dropout_rates=0.1,
        conv_biases=False,
        residual_channels=[96, 96, 96],
        residual_kernel_size=[3, 3, 3],
        residual_stride_size=[1, 1, 1],
        residual_dilation_rate=[1, 2, 4],
        residual_padding="same",
        residual_activation="relu",
        residual_batchnorm=True,
        residual_batchnorm_first=True,
        residual_dropout_rates=0.1,
        residual_biases=False,
        pool_kernel_size=10,
        pool_dropout_rate=0.2,
        dense_hidden_dims=[256],
        dense_activation="relu",
        dense_batchnorm=True,
        dense_batchnorm_first=True,
        dense_dropout_rates=0.5,
        dense_biases=False,
        **kwargs
    ):
        super().__init__(
            input_len, output_dim, strand=strand, task=task, aggr=aggr, **kwargs
        )
        if isinstance(conv_channels, int):
            conv_channels = [conv_channels]
        
        # Pass through normal conv
        self.conv1d_tower = towers.Conv1DTower(
            input_len=input_len,
            input_channels=input_chanels,
            conv_channels=conv_channels,
            conv_kernels=conv_kernel_size,
            conv_strides=conv_stride_size,
            conv_dilations=conv_dilation_rate,
            conv_padding=conv_padding,
            conv_biases=conv_biases,
            activations=conv_activation,
            pool_types=[None],
            dropout_rates=conv_dropout_rates,
            batchnorm=conv_batchnorm,
            batchnorm_first=conv_batchnorm_first
        )
        
        # Pass through residual block
        res_block_input_len = self.conv1d_tower.output_len
        self.residual_block = layers.Residual(
            towers.Conv1DTower(
                input_len=res_block_input_len,
                input_channels=self.conv1d_tower.out_channels,
                conv_channels=residual_channels,
                conv_kernels=residual_kernel_size,
                conv_strides=residual_stride_size,
                conv_dilations=residual_dilation_rate,
                conv_padding=residual_padding,
                conv_biases=residual_biases,
                activations=residual_activation,
                pool_types=None,
                dropout_rates=residual_dropout_rates,
                batchnorm=residual_batchnorm,
                batchnorm_first=residual_batchnorm_first
                )
        )
        self.average_pool = nn.AvgPool1d(pool_kernel_size, stride=1)
        self.dropout = nn.Dropout(pool_dropout_rate)
        self.flatten = nn.Flatten()
        self.dense_block = blocks.DenseBlock(
            input_dim=self.residual_block.wrapped.out_channels*(res_block_input_len-pool_kernel_size+1),
            output_dim=output_dim,
            hidden_dims=dense_hidden_dims,
            activations=dense_activation,
            batchnorm=dense_batchnorm,
            batchnorm_first=dense_batchnorm_first,
            dropout_rates=dense_dropout_rates,
            biases=dense_biases
        )

    def forward(self, x, x_rev=None):
        x = self.conv1d_tower(x)
        x = self.residual_block(x)
        x = self.average_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        return x

class DeepBind( SequenceModel):
    """
    DeepBind model implemented from Alipanahi et al 2015 in PyTorch

    DeepBind is a model that takes in a DNA or RNA sequence and outputs a probability of 
    binding for a given DNA transcription factor or RNA binding protein respectively.
    This is a flexible implementation of the original DeepBind architecture that allows users
    to modify the number of convolutional layers, the number of fully connected layers, and
    many more hyperparameters. If parameters for the CNN and FCN are not passed in, the model
    will be instantiated with the parameters described in Alipanahi et al 2015.

    Like the original DeepBind models, this model can be used for both DNA and RNA binding. For DNA,
    we implemented the "dna" mode which only uses the max pooling of the representation generated by 
    the convolutional layers. For RNA, we implemented the "rbp" mode which uses both the max and average
    pooling of the representation generated by the convolutional layers.
    - For "ss" models, we use the representation generated by the convolutional layers and pass that through a 
        set of fully connected layer to generate the output.
    - For "ds" models, we use the representation generated by the convolutional layers for both the forward and 
        reverse complement strands and pass that through the same set of fully connected layers to generate the output.
    - For "ts" models, we use the representation generated by separate sets of convolutional layers for the forward and
        reverse complement strands and passed that through separate sets of fully connected layers to generate the output.
    
    aggr defines how the output for "ds" and "ts" models is generated. If "max", we take the max of the forward and reverse
    complement outputs. If "avg", we take the average of the forward and reverse complement outputs. There is no "concat" for
    the current implementation of DeepBind models.

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
    dense_kwargs : dict
        Keyword arguments for fully connected layers
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = "max",
        loss_fxn: str ="mse",
        mode: str = "rbp",
        conv_kwargs: dict = {},
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
        self.mode = mode
        self.mode_dict = {"dna": 1, "rbp": 2}
        self.mode_multiplier = self.mode_dict[self.mode]
        self.aggr = aggr
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.pool_dim = self.conv1d_tower.output_len
        self.max_pool = nn.MaxPool1d(kernel_size=self.pool_dim)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.pool_dim)
        if self.strand == "ss":
            self.dense_block = blocks.DenseBlock(
                input_dim=self.conv1d_tower.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )
        elif self.strand == "ds":
            self.dense_block = DenseBlock(
                input_dim=self.conv1d_tower.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )
        elif self.strand == "ts":
            self.dense_block = DenseBlock(
                self.conv1d_tower.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )
            self.reverse_conv1d_block = towers.Conv1DTower(
                input_len=input_len, 
                **self.conv_kwargs
                )
            self.reverse_dense_block = DenseBlock(
                self.conv1d_tower.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.dense_kwargs
            )

    def forward(self, x, x_rev_comp=None):

        x = self.conv1d_tower(x)
        if self.mode == "rbp":
            x = torch.cat((self.max_pool(x), self.avg_pool(x)), dim=1)
            x = x.view(x.size(0), self.conv1d_tower.out_channels * 2)
        elif self.mode == "dna":
            x = self.max_pool(x)
            x = x.view(x.size(0), self.conv1d_tower.out_channels)
        x = self.dense_block(x)
        if self.strand == "ds":
            x_rev_comp = self.conv1d_tower(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_tower.out_channels * 2)
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_tower.out_channels)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_conv1d_block(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_tower.out_channels * 2)
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_tower.out_channels)
            x_rev_comp = self.reverse_dense_block(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x

    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [16])
        conv_kwargs.setdefault("conv_kernels", [16])
        conv_kwargs.setdefault("pool_types", None)
        conv_kwargs.setdefault("dropout_rates", 0.25)
        conv_kwargs.setdefault("batchnorm", False)
        dense_kwargs.setdefault("hidden_dims", [32])
        dense_kwargs.setdefault("dropout_rates", 0.25)
        dense_kwargs.setdefault("batchnorm", False)
        return conv_kwargs, dense_kwargs 

class DeepSEA( SequenceModel):
    """DeepSEA model implementation for EUGENe
    
    Default parameters are those specified in the DeepSEA paper. We currently do not implement a "ds" or "ts" model
    for DeepSEA.

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
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        conv_kwargs: dict = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.dense_block = blocks.DenseBlock(
            input_dim=self.conv1d_tower.flatten_dim, 
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)
        x = self.dense_block(x)
        return x

    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [320, 480, 960])
        conv_kwargs.setdefault("conv_kernels", [8, 8, 8])
        conv_kwargs.setdefault("pool_types", ["max", "max", None])
        conv_kwargs.setdefault("pool_kernels", [4, 4, None])
        conv_kwargs.setdefault("activations", "relu")
        conv_kwargs.setdefault("dropout_rates", [0.2, 0.2, 0.5])
        conv_kwargs.setdefault("batchnorm", False)
        dense_kwargs.setdefault("hidden_dims", [925])
        return conv_kwargs,dense_kwargs 

class Basset(SequenceModel):
    """
    """
    def __init__(
        self, 
        input_len: int,
        output_dim: int, 
        strand = "ss",
        task = "multilabel_classification",
        aggr = None,
        loss_fxn = "bce",
        conv_kwargs = {},
        dense_kwargs = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.dense_block = blocks.DenseBlock(
            input_dim=self.conv1d_tower.flatten_dim, 
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)
        x = self.dense_block(x)
        return x
        
    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [300, 200, 200])
        conv_kwargs.setdefault("conv_kernels", [19, 11, 7])
        conv_kwargs.setdefault("conv_strides", [1, 1, 1])
        conv_kwargs.setdefault("conv_padding", [9, 5, 3])
        conv_kwargs.setdefault("pool_kernels", [3, 4, 4])
        conv_kwargs.setdefault("dropout_rates", 0.0)
        conv_kwargs.setdefault("activations", "relu")
        conv_kwargs.setdefault("batchnorm", True)
        conv_kwargs.setdefault("batchnorm_first", True)
        dense_kwargs.setdefault("hidden_dims", [1000, 164])
        dense_kwargs.setdefault("dropout_rates", 0.0)
        dense_kwargs.setdefault("batchnorm", True)
        dense_kwargs.setdefault("batchnorm_first", True)
        dense_kwargs.setdefault("activations", "relu")
        return conv_kwargs,dense_kwargs 

class DanQ( SequenceModel):
    """DanQ model from Quang and Xie, 2016;

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
        strand: str = "ss",
        task: str = "regression",
        loss_fxn: str = "mse",
        aggr: str = None,
        conv_kwargs: dict = {},
        recurrent_kwargs: dict = {},
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
        self.conv_kwargs, self.recurrent_kwargs, self.dense_kwargs = self.kwarg_handler(
            conv_kwargs, recurrent_kwargs, dense_kwargs
        )
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.recurrent_block = blocks.RecurrentBlock(
            input_dim=self.conv1d_tower.out_channels, 
            **self.recurrent_kwargs
        )
        self.dense_block = blocks.DenseBlock(
            input_dim=self.recurrent_block.out_channels,
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_tower(x)
        x = x.transpose(1, 2)
        out, _ = self.recurrent_block(x)
        out = self.dense_block(out[:, -1, :])
        return out

    def kwarg_handler(self, conv_kwargs, recurrent_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [320])
        conv_kwargs.setdefault("conv_kernels", [26])
        conv_kwargs.setdefault("conv_strides", [1])
        conv_kwargs.setdefault("conv_padding", "same")
        conv_kwargs.setdefault("pool_kernels", [13])
        conv_kwargs.setdefault("dropout_rates", 0.2)
        conv_kwargs.setdefault("activations", "relu")
        recurrent_kwargs.setdefault("unit_type", "lstm")
        recurrent_kwargs.setdefault("hidden_dim", 320)
        recurrent_kwargs.setdefault("bidirectional", True)
        recurrent_kwargs.setdefault("batch_first", True)
        dense_kwargs.setdefault("hidden_dims", [925])
        dense_kwargs.setdefault("dropout_rates", 0.5)
        dense_kwargs.setdefault("batchnorm", False)
        return conv_kwargs, recurrent_kwargs, dense_kwargs

class DeepSTARR( SequenceModel):
    """DeepSTARR model from de Almeida et al., 2022; 
        see <https://www.nature.com/articles/s41588-022-01048-5>


    Parameters
    """
    def __init__(
        self, 
        input_len: int,
        output_dim: int, 
        strand = "ss",
        task = "regression",
        aggr = None,
        loss_fxn = "mse",
        conv_kwargs = {},
        dense_kwargs = {},
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
        self.conv_kwargs, self.dense_kwargs = self.kwarg_handler(conv_kwargs, dense_kwargs)
        self.conv1d_tower = towers.Conv1DTower(**self.conv_kwargs)
        self.dense_block = blocks.DenseBlock(
            input_dim=self.conv1d_tower.flatten_dim, 
            output_dim=output_dim, 
            **self.dense_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)
        x = self.dense_block(x)
        return x
        
    def kwarg_handler(self, conv_kwargs, dense_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("input_len", self.input_len)
        conv_kwargs.setdefault("input_channels", 4)
        conv_kwargs.setdefault("conv_channels", [246, 60, 60, 120])
        conv_kwargs.setdefault("conv_kernels", [7, 3, 5, 3])
        conv_kwargs.setdefault("conv_strides", [1, 1, 1, 1])
        conv_kwargs.setdefault("conv_padding", "same")
        conv_kwargs.setdefault("pool_kernels", [2, 2, 2, 2])
        conv_kwargs.setdefault("dropout_rates", 0.0)
        conv_kwargs.setdefault("batchnorm", True)
        conv_kwargs.setdefault("batchnorm_first", True)
        dense_kwargs.setdefault("hidden_dims", [256, 256])
        dense_kwargs.setdefault("dropout_rates", 0.4)
        dense_kwargs.setdefault("batchnorm", True)
        dense_kwargs.setdefault("batchnorm_first", True)
        return conv_kwargs, dense_kwargs 