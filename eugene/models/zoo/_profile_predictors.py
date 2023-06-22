import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import _layers as layers
from ..base import _blocks as blocks
from ..base import _towers as towers


class BPNet(nn.Module):
    """
    This nn.Module was 

    A basic BPNet model with stranded profile and total count prediction.
    This is a reference implementation for BPNet. The model takes in
    one-hot encoded sequence, runs it through:
    (1) a single wide convolution operation
    THEN
    (2) a user-defined number of dilated residual convolutions
    THEN
    (3a) profile predictions done using a very wide convolution layer
    that also takes in stranded control tracks
    AND
    (3b) total count prediction done using an average pooling on the output
    from 2 followed by concatenation with the log1p of the sum of the
    stranded control tracks and then run through a dense layer.
    This implementation differs from the original BPNet implementation in
    two ways:
    (1) The model concatenates stranded control tracks for profile
    prediction as opposed to adding the two strands together and also then
    smoothing that track
    (2) The control input for the count prediction task is the log1p of
    the strand-wise sum of the control tracks, as opposed to the raw
    counts themselves.
    (3) A single log softmax is applied across both strands such that
    the logsumexp of both strands together is 0. Put another way, the
    two strands are concatenated together, a log softmax is applied,
    and the MNLL loss is calculated on the concatenation.
    (4) The count prediction task is predicting the total counts across
    both strands. The counts are then distributed across strands according
    to the single log softmax from 3.
    Parameters
    ----------
    n_filters: int, optional
            The number of filters to use per convolution. Default is 64.
    n_layers: int, optional
            The number of dilated residual layers to include in the model.
            Default is 8.
    n_outputs: int, optional
            The number of profile outputs from the model. Generally either 1 or 2
            depending on if the data is unstranded or stranded. Default is 2.
    alpha: float, optional
            The weight to put on the count loss.
    name: str or None, optional
            The name to save the model to during training.
    trimming: int or None, optional
            The amount to trim from both sides of the input window to get the
            output window. This value is removed from both sides, so the total
            number of positions removed is 2*trimming.
    verbose: bool, optional
            Whether to display statistics during training. Setting this to False
            will still save the file at the end, but does not print anything to
            screen during training. Default is True.
    """

    def __init__(
        self,
        input_len,
        output_dim,
        n_filters=64,
        n_layers=8,
        n_outputs=2,
        n_control_tracks=2,
        alpha=1,
        profile_output_bias=True,
        count_output_bias=True,
        name=None,
        trimming=None,
        verbose=True,
    ):
        super(BPNet, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.alpha = alpha
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
        self.trimming = trimming or 2**n_layers

        # Build the model
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.irelu = torch.nn.ReLU()

        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.rrelus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(1, self.n_layers + 1)]
        )

        self.fconv = torch.nn.Conv1d(
            n_filters + n_control_tracks,
            n_outputs,
            kernel_size=75,
            padding=37,
            bias=profile_output_bias,
        )

        n_count_control = 1 if n_control_tracks > 0 else 0
        self.linear = torch.nn.Linear(
            n_filters + n_count_control, 1, bias=count_output_bias
        )

    def forward(self, X, X_ctl=None):
        """A forward pass of the model.
        This method takes in a nucleotide sequence X, a corresponding
        per-position value from a control track, and a per-locus value
        from the control track and makes predictions for the profile
        and for the counts. This per-locus value is usually the
        log(sum(X_ctl_profile)+1) when the control is an experimental
        read track but can also be the output from another model.
        Parameters
        ----------
        X: torch.tensor, shape=(batch_size, 4, sequence_length)
                The one-hot encoded batch of sequences.
        X_ctl: torch.tensor, shape=(batch_size, n_strands, sequence_length)
                A value representing the signal of the control at each position in
                the sequence.
        Returns
        -------
        y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
                The output predictions for each strand.
        """
        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.irelu(self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.rrelus[i](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        if X_ctl is None:
            X_w_ctl = X
        else:
            X_w_ctl = torch.cat([X, X_ctl], dim=1)

        y_profile = self.fconv(X_w_ctl)[:, :, start:end]

        # counts prediction
        X = torch.mean(X[:, :, start - 37 : end + 37], dim=2)

        if X_ctl is not None:
            X_ctl = torch.sum(X_ctl[:, :, start - 37 : end + 37], dim=(1, 2))
            X_ctl = X_ctl.unsqueeze(-1)
            X = torch.cat([X, torch.log(X_ctl + 1)], dim=-1)

        y_counts = self.linear(X).reshape(X.shape[0], 1)
        return y_profile, y_counts
