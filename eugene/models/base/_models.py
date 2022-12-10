import torch
from . import BaseModel
from . import DenseBlock, ConvBlock1D, RecurrentBlock


class Dense(BaseModel):
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
            self.dense = DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif self.strand == "ds":
            if self.aggr == "concat":
                self.dense = DenseBlock(
                    input_dim=self.flattened_input_dims * 2,
                    output_dim=output_dim,
                    **dense_kwargs
                )
            elif self.aggr in ["max", "avg"]:
                self.dense = DenseBlock(
                    input_dim=self.flattened_input_dims,
                    output_dim=output_dim,
                    **dense_kwargs
                )
        elif self.strand == "ts":
            self.dense = DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )
            self.reverse_fcnet = DenseBlock(
                input_dim=self.flattened_input_dims, 
                output_dim=output_dim, 
                **dense_kwargs
            )

    def forward(self, x, x_rev_comp=None):
        x = x.flatten(start_dim=1)
        if self.strand == "ss":
            x = self.dense(x)
        elif self.strand == "ds":
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            if self.aggr == "concat":
                x = torch.cat((x, x_rev_comp), dim=1)
                x = self.dense(x)
            elif self.aggr in ["max", "avg"]:
                x = self.dense(x)
                x_rev_comp = self.dense(x_rev_comp)
                if self.aggr == "max":
                    x = torch.max(x, x_rev_comp)
                elif self.aggr == "avg":
                    x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x = self.dense(x)
            x_rev_comp = x_rev_comp.flatten(start_dim=1)
            x_rev_comp = self.reverse_fcnet(x_rev_comp)
            if self.aggr == "concat":
                raise ValueError("Concatenation is not supported for the tsfcnet model.")
            elif self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x
