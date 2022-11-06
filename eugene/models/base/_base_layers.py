# from yuzu

import torch.nn as nn


class Flatten(torch.nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.permute(0, 2, 1).contiguous().view(x.shape[0], -1)


class Unsqueeze(torch.nn.Module):
	def __init__(self, dim):
		super(Unsqueeze, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)