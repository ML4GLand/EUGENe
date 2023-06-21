from typing import List, Tuple, Union

import torch


class RandomDeletion:
    """Randomly deletes a contiguous stretch of nucleotides from sequences in a training
    batch according to a random number between a user-defined delete_min and delete_max.
    A different deletion is applied to each sequence.

    Parameters
    ----------
    delete_min : int, optional
        Minimum size for random deletion (defaults to 0).
    delete_max : int, optional
        Maximum size for random deletion (defaults to 20).
    """

    def __init__(self, delete_min=0, delete_max=20):
        self.delete_min = delete_min
        self.delete_max = delete_max

    def __call__(self, x):
        """Randomly delete segments in a set of one-hot DNA sequences.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (N, A, L)).

        Returns
        -------
        torch.Tensor
            Sequences with randomly deleted segments (padded to correct shape
            with random DNA)
        """
        N, A, L = x.shape

        # sample random DNA
        a = torch.eye(A)
        p = torch.tensor([1 / A for _ in range(A)])
        padding = torch.stack(
            [
                a[p.multinomial(self.delete_max, replacement=True)].transpose(0, 1)
                for _ in range(N)
            ]
        ).to(x.device)

        # sample deletion length for each sequence
        delete_lens = torch.randint(self.delete_min, self.delete_max + 1, (N,))

        # sample locations to delete for each sequence
        delete_inds = torch.randint(
            L - self.delete_max + 1, (N,)
        )  # deletion must be in boundaries of seq.

        # loop over each sequence
        x_aug = []
        for seq, pad, delete_len, delete_ind in zip(
            x, padding, delete_lens, delete_inds
        ):
            # get index of half delete_len (to pad random DNA at beginning of sequence)
            pad_begin_index = torch.div(delete_len, 2, rounding_mode="floor").item()

            # index for other half (to pad random DNA at end of sequence)
            pad_end_index = delete_len - pad_begin_index

            # removes deletion and pads beginning and end of sequence with random DNA to ensure same length
            x_aug.append(
                torch.cat(
                    [
                        pad[:, :pad_begin_index],  # random dna padding
                        seq[:, :delete_ind],  # sequence up to deletion start index
                        seq[
                            :, delete_ind + delete_len :
                        ],  # sequence after deletion end index
                        pad[:, self.delete_max - pad_end_index :],
                    ],  # random dna padding
                    -1,
                )
            )  # concatenation axis
        return torch.stack(x_aug)


class RandomInsertion:
    """Randomly inserts a contiguous stretch of nucleotides from sequences in a training
    batch according to a random number between a user-defined insert_min and insert_max.
    A different insertions is applied to each sequence. Each sequence is padded with random
    DNA to ensure same shapes.

    Parameters
    ----------
    insert_min : int, optional
        Minimum size for random insertion, defaults to 0
    insert_max : int, optional
        Maximum size for random insertion, defaults to 20
    """

    def __init__(self, insert_min=0, insert_max=20):
        self.insert_min = insert_min
        self.insert_max = insert_max

    def __call__(self, x):
        """Randomly inserts segments of random DNA to a set of DNA sequences.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (N, A, L)).

        Returns
        -------
        torch.Tensor
            Sequences with randomly inserts segments of random DNA. All sequences
            are padded with random DNA to ensure same shape.
        """
        N, A, L = x.shape

        # sample random DNA
        a = torch.eye(A)
        p = torch.tensor([1 / A for _ in range(A)])
        insertions = torch.stack(
            [
                a[p.multinomial(self.insert_max, replacement=True)].transpose(0, 1)
                for _ in range(N)
            ]
        ).to(x.device)

        # sample insertion length for each sequence
        insert_lens = torch.randint(self.insert_min, self.insert_max + 1, (N,))

        # sample locations to insertion for each sequence
        insert_inds = torch.randint(L, (N,))

        # loop over each sequence
        x_aug = []
        for seq, insertion, insert_len, insert_ind in zip(
            x, insertions, insert_lens, insert_inds
        ):
            # get index of half insert_len (to pad random DNA at beginning of sequence)
            insert_beginning_len = torch.div(
                (self.insert_max - insert_len), 2, rounding_mode="floor"
            ).item()

            # index for other half (to pad random DNA at end of sequence)
            self.insert_max - insert_len - insert_beginning_len

            # removes deletion and pads beginning and end of sequence with random DNA to ensure same length
            x_aug.append(
                torch.cat(
                    [
                        insertion[:, :insert_beginning_len],  # random dna padding
                        seq[:, :insert_ind],  # sequence up to insertion start index
                        insertion[
                            :, insert_beginning_len : insert_beginning_len + insert_len
                        ],  # random insertion
                        seq[:, insert_ind:],  # sequence after insertion end index
                        insertion[
                            :, insert_beginning_len + insert_len : self.insert_max
                        ],
                    ],  # random dna padding
                    -1,
                )
            )  # concatenation axis
        return torch.stack(x_aug)


class RandomTranslocation:
    """Randomly cuts sequence in two pieces and shifts the order for each in a training
    batch. This is implemented with a roll transformation with a user-defined shift_min
    and shift_max. A different roll (positive or negative) is applied to each sequence.
    Each sequence is padded with random DNA to ensure same shapes.

    Parameters
    ----------
    shift_min : int, optional
        Minimum size for random shift, defaults to 0.
    shift_max : int, optional
        Maximum size for random shift, defaults to 20.
    """

    def __init__(self, shift_min=0, shift_max=20):
        self.shift_min = shift_min
        self.shift_max = shift_max

    def __call__(self, x):
        """Randomly shifts sequences in a batch, x.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (N, A, L)).

        Returns
        -------
        torch.Tensor
            Sequences with random translocations.
        """
        N = x.shape[0]

        # determine size of shifts for each sequence
        shifts = torch.randint(self.shift_min, self.shift_max + 1, (N,))

        # make some of the shifts negative
        ind_neg = torch.rand(N) < 0.5
        shifts[ind_neg] = -1 * shifts[ind_neg]

        # apply random shift to each sequence
        x_rolled = []
        for i, shift in enumerate(shifts):
            x_rolled.append(torch.roll(x[i], shift.item(), -1))
        x_rolled = torch.stack(x_rolled).to(x.device)
        return x_rolled


class RandomInversion:
    """Randomly inverts a contiguous stretch of nucleotides from sequences in a training
    batch according to a user-defined invert_min and invert_max. A different insertions
    is applied to each sequence. Each sequence is padded with random DNA to ensure same
    shapes.

    Parameters
    ----------
    invert_min : int, optional
        Minimum size for random insertion, defaults to 0.
    invert_max : int, optional
        Maximum size for random insertion, defaults to 20.
    """

    def __init__(self, invert_min=0, invert_max=20):
        self.invert_min = invert_min
        self.invert_max = invert_max

    def __call__(self, x):
        """Randomly inverts segments of random DNA to a set of one-hot DNA sequences.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (N, A, L)).

        Returns
        -------
        torch.Tensor
            Sequences with randomly inverted segments of random DNA.
        """
        N, A, L = x.shape

        # set random inversion size for each seequence
        inversion_lens = torch.randint(self.invert_min, self.invert_max + 1, (N,))

        # randomly select start location for each inversion
        inversion_inds = torch.randint(
            L - self.invert_max + 1, (N,)
        )  # inversion must be in boundaries of seq.

        # apply random inversion to each sequence
        x_aug = []
        for seq, inversion_len, inversion_ind in zip(x, inversion_lens, inversion_inds):
            x_aug.append(
                torch.cat(
                    [
                        seq[:, :inversion_ind],  # sequence up to inversion start index
                        torch.flip(
                            seq[:, inversion_ind : inversion_ind + inversion_len],
                            dims=[0, 1],
                        ),  # reverse-complement transformation
                        seq[:, inversion_ind + inversion_len :],
                    ],  # sequence after inversion
                    -1,
                )
            )  # concatenation axis
        return torch.stack(x_aug)


class RandomMutation:
    """Randomly mutates sequences in a training batch according to a user-defined
    mutate_frac. A different set of mutations is applied to each sequence.

    Parameters
    ----------
    mutate_frac : float, optional
        Probability of mutation for each nucleotide, defaults to 0.05.
    """

    def __init__(self, mutate_frac=0.05):
        self.mutate_frac = mutate_frac

    def __call__(self, x):
        """Randomly introduces mutations to a set of one-hot DNA sequences.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (N, A, L)).

        Returns
        -------
        torch.Tensor
            Sequences with randomly mutated DNA.
        """
        N, A, L = x.shape

        # determine the number of mutations per sequence
        num_mutations = round(
            self.mutate_frac / 0.75 * L
        )  # num. mutations per sequence (accounting for silent mutations)

        # randomly determine the indices to apply mutations
        mutation_inds = torch.argsort(torch.rand(N, L))[
            :, :num_mutations
        ]  # see <https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146>0

        # create random DNA (to serve as random mutations)
        a = torch.eye(A)
        p = torch.tensor([1 / A for _ in range(A)])
        mutations = torch.stack(
            [
                a[p.multinomial(num_mutations, replacement=True)].transpose(0, 1)
                for _ in range(N)
            ]
        ).to(x.device)

        # make a copy of the batch of sequences
        x_aug = torch.clone(x)

        # loop over sequences and apply mutations
        for i in range(N):
            x_aug[i, :, mutation_inds[i]] = mutations[i]
        return x_aug


class RandomRC:
    """Randomly applies a reverse-complement transformation to each sequence in a training
    batch according to a user-defined probability, rc_prob. This is applied to each sequence
    independently.

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


class RandomNoise:
    """Randomly add Gaussian noise to a batch of sequences with according to a user-defined
    noise_mean and noise_std. A different set of noise is applied to each sequence.

    Parameters
    ----------
    noise_mean : float, optional
        Mean of the Gaussian noise, defaults to 0.0.
    noise_std : float, optional
        Standard deviation of the Gaussian noise, defaults to 0.2.
    """

    def __init__(self, noise_mean=0.0, noise_std=0.2):
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __call__(self, x):
        """Randomly adds Gaussian noise to a set of one-hot DNA sequences.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (N, A, L)).

        Returns
        -------
        torch.Tensor
            Sequences with random noise.
        """
        return x + torch.normal(self.noise_mean, self.noise_std, x.shape).to(x.device)


class RandomJitter:
    def __init__(self, max_jitter: int, length_axis: int) -> None:
        """Randomly jitter a sequence that has been padded on either side to support
        jittering by `max_jitter` amount.

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
