import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ..._settings import settings

class ProfileDataset(Dataset):
	"""A data generator for BPNet inputs.
	This generator takes in an extracted set of sequences, output signals,
	and control signals, and will return a single element with random
	jitter and reverse-complement augmentation applied. Jitter is implemented
	efficiently by taking in data that is wider than the in/out windows by
	two times the maximum jitter and windows are extracted from that.
	Essentially, if an input window is 1000 and the maximum jitter is 128, one
	would pass in data with a length of 1256 and a length 1000 window would be
	extracted starting between position 0 and 256. This  generator must be 
	wrapped by a PyTorch generator object.
	Parameters
	----------
	sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		A one-hot encoded tensor of `n` example sequences, each of input 
		length `in_window`. See description above for connection with jitter.
	signals: torch.tensor, shape=(n, t, out_window+2*max_jitter)
		The signals to predict, usually counts, for `n` examples with
		`t` output tasks (usually 2 if stranded, 1 otherwise), each of 
		output length `out_window`. See description above for connection 
		with jitter.
	controls: torch.tensor, shape=(n, t, out_window+2*max_jitter) or None, optional
		The control signal to take as input, usually counts, for `n`
		examples with `t` strands and output length `out_window`. If
		None, does not return controls.
	in_window: int, optional
		The input window size. Default is 2114.
	out_window: int, optional
		The output window size. Default is 1000.
	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.
	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is False.
	random_state: int or None, optional
		Whether to use a deterministic seed or not.
	"""

	def __init__(
		self, 
		sequences, 
		signals, 
		controls=None, 
		in_window=2114, 
		out_window=1000, 
		max_jitter=0, 
		reverse_complement=False, 
		random_state=None
	):
		self.in_window = in_window
		self.out_window = out_window
		self.max_jitter = max_jitter
		
		self.reverse_complement = reverse_complement
		self.random_state = np.random.RandomState(random_state)

		self.signals = signals
		self.controls = controls
		self.sequences = sequences	

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		#i = self.random_state.choice(len(self.sequences))
		j = 0 if self.max_jitter == 0 else self.random_state.randint(self.max_jitter*2) 

		X = self.sequences[idx][:, j:j+self.in_window]
		y = self.signals[idx][:, j:j+self.out_window]

		if self.controls is not None:
			X_ctl = self.controls[idx][:, j:j+self.in_window]

		if self.reverse_complement and self.random_state.choice(2) == 1:
			X = torch.flip(X, [0, 1])
			y = torch.flip(y, [0, 1])

			if self.controls is not None:
				X_ctl = torch.flip(X_ctl, [0, 1])

		if self.controls is not None:
			return X, X_ctl, y

		return X, y
	
	def to_dataloader(
		self, 
        batch_size=None, 
        pin_memory=True, 
        shuffle=True, 
        num_workers=0, 
        **kwargs
    ):
		"""Convert the dataset to a PyTorch DataLoader

		Parameters:
		----------
		batch_size (int, optional):
			batch size for dataloader
		pin_memory (bool, optional):
			whether to pin memory for dataloader
		shuffle (bool, optional):
			whether to shuffle the dataset
		num_workers (int, optional):
			number of workers for dataloader
		**kwargs:
			additional arguments to pass to DataLoader
		"""
		batch_size = batch_size if batch_size is not None else settings.batch_size
		return DataLoader(
			self,
			batch_size=batch_size,
			pin_memory=pin_memory,
			shuffle=shuffle,
			num_workers=num_workers,
			**kwargs
		)