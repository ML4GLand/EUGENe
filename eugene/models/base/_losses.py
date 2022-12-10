import torch.functional as F

def adversarial_loss(self, y_hat, y):
    return F.binary_cross_entropy(y_hat, y)