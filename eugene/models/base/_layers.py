import torch.nn as nn

# from yuzu
class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

class Unsqueeze(nn.Module):
	def __init__(self, dim):
		super(Unsqueeze, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)

# custom
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class Clip(nn.Module):
	def __init__(self, min, max):
		super().__init__()
		self.min = min
		self.max = max

	def forward(self, x):
		return torch.clamp(x, self.min, self.max)
		
class Residual(nn.Module):
	def __init__(self, module):
		super(Residual, self).__init__()
		self.module = module

	def forward(self, x):
		return x + self.module(x)

RECURRENT_REGISTRY = {
	"rnn": nn.RNN,
	"lstm": nn.LSTM,
	"gru": nn.GRU
}

POOLING_REGISTRY = {
	"max": nn.MaxPool1d,
	"avg": nn.AvgPool1d,
	"sum": nn.AdaptiveAvgPool1d,
}

GLUER_REGISTRY = {
	"flatten": Flatten,
	"unsqueeze": Unsqueeze
}

WRAPPER_REGISTRY = {
	"residual": Residual
}