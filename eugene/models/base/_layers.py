import torch
import torch.nn as nn
import torch.nn.functional as F

# CONVOLUTIONS -- Layers that convolve the input
class BiConv1D(nn.Module):
	def __init__(
		self, 
		in_channels, 
		out_channels, 
		kernel_size, 
		stride=1, 
		padding="same", 
		dilation=1, 
		groups=1, 
		bias=True,
		dropout_rate=0.0, 
		device=None,
		dtype=None,
	):
		super(BiConv1D, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.weight = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size)))
		if bias:   
			self.bias = nn.Parameter(torch.zeros(out_channels))
		else:
			self.bias = None
		if dropout_rate != 0.0 and dropout_rate is not None:
			self.dropout_rate = dropout_rate
		else:
			self.dropout_rate = None
			
	def forward(self, x):
		x_fwd = F.conv1d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)  		
		x_rev = F.conv1d(x, torch.flip(self.weight, dims=[0, 1]), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
		if self.bias is not None:
			x_fwd = torch.add(x_fwd.transpose(1, 2), self.bias).transpose(1, 2)
			x_rev = torch.add(x_rev.transpose(1, 2), self.bias).transpose(1, 2)
		if self.dropout_rate is not None:
			x_fwd = F.dropout(F.relu(x_fwd), p=self.dropout_rate)
			x_rev = F.dropout(F.relu(x_rev), p=self.dropout_rate)
		return torch.add(x_fwd, x_rev)


	def __repr__(self):
		return "BiConv1D({}, {}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={})".format(
			self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None
		)


CONVOLUTION_REGISTRY = {
	"conv1d": nn.Conv1d,
	"biconv1d": BiConv1D,
}

# POOLERS -- Layers that reduce the size of the input
POOLING_REGISTRY = {
	"max": nn.MaxPool1d,
	"avg": nn.AvgPool1d,
	"sum": nn.AdaptiveAvgPool1d,
}

# RECURRENCES -- Layers that can be used in a recurrent context
RECURRENT_REGISTRY = {
	"rnn": nn.RNN,
	"lstm": nn.LSTM,
	"gru": nn.GRU
}

# NORMALIZERS -- Layers that normalize the input
NORMALIZER_REGISTRY = {
	"batch_norm": nn.BatchNorm1d,
}

# WRAPPERS -- Layers that wrap other layers
class Residual(nn.Module):
	def __init__(self, module):
		super(Residual, self).__init__()
		self.module = module

	def forward(self, x):
		return x + self.module(x)

WRAPPER_REGISTRY = {
	"residual": Residual
}

# GLUERS -- Layers that go in between other layers
# from yuzu
class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

# from yuzu
class Unsqueeze(nn.Module):
	def __init__(self, dim):
		super(Unsqueeze, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)

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
		
GLUER_REGISTRY = {
	"flatten": Flatten,
	"unsqueeze": Unsqueeze,
	"view": View
}

# MISC -- Layers that modify the input in some way
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Clip(nn.Module):
	def __init__(self, min, max):
		super().__init__()
		self.min = min
		self.max = max

	def forward(self, x):
		return torch.clamp(x, self.min, self.max)

MISC_REGISTRY = {
	"identity": Identity,
	"clip": Clip
}