import torch
from seqpro import ohe_seq

class OneHotEncode(object):
    """One-hot encode the input sequence if its not already one-hot encoded"""

    def __init__(self):
        pass

    def __call__(self, sample):
        sample["seq"]= torch.tensor(ohe_seq(sample["seq"]), dtype=torch.float32)
        return sample
    
class ReverseComplement(object):
    """Reverse complement an input sequence"""

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, sample):
        seq = sample["seq"]
        if torch.rand(1) > self.p:
            sample["seq"] = torch.flip(seq, [0])
        return sample

class Transpose(object):
    """Transpose the input sequence"""

    def __init__(self):
        pass

    def __call__(self, sample):
        sample["seq"] = torch.transpose(sample["seq"], 0, 1)
        return sample