from typing import List, Tuple, Union

import torch

class RandomRC:
    """Copied from https://github.com/p-koo/evoaug/blob/master/evoaug/augment.py
    
    Randomly applies a reverse-complement transformation to each sequence in a training batch
    Takes in a user-defined probability, rc_prob. This is applied to each sequence independently.

    Parameters
    ----------
    rc_prob : float, optional
        Probability to apply a reverse-complement transformation, defaults to 0.5.
    """

    def __init__(self, rc_prob=0.5):
        """Creates random reverse-complement object usable by EvoAug."""
        self.rc_prob = rc_prob

    def __call__(self, *x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Randomly transforms sequences in a batch with a reverse-complement transformation.

        Parameters
        ----------
        x : torch.Tensor
            Batch (or tuple of batches) of one-hot sequences (shape: (N, A, L)).

        Returns
        -------
        torch.Tensor
            Sequences with random reverse-complements applied.
        """
        n = x[0].shape[0]
        # randomly select sequences to apply rc transformation
        ind_rc = torch.rand(n) < self.rc_prob
        
        out: List[torch.Tensor] = []
        for _x in x:
            # make a copy of the sequence
            x_aug = torch.clone(_x)

            # apply reverse-complement transformation
            x_aug[ind_rc] = torch.flip(x_aug[ind_rc], dims=[1, 2])
            
            out.append(x_aug)
        
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)

class RandomJitter:
    def __init__(self, max_jitter: int, length_axis: int) -> None:
        """Randomly jitter a sequence that has been padded on either side to support jittering by `max_jitter` amount.

        Parameters
        ----------
        max_jitter : int
        length_axis : int
            Axis that corresponds to the length dimension.
        """
        self.max_jitter = max_jitter
        self.length_axis = length_axis

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random jittering.

        Parameters
        ----------
        x : torch.Tensor
            Batch of sequences where the length axis corresponds to the `length_axis`
            parameter given at initialization. E.g. shape: (N, A, L), then
            `length_axis` should be 2.

        Returns
        -------
        torch.Tensor
            Jittered sequence.
        """
        length = x.shape[self.length_axis]
        start = torch.randint(0, self.max_jitter, (1,))
        end = length - self.max_jitter * 2 + start
        return x[(slice(None),) * (self.length_axis % x.ndim) + (slice(start, end),)]
