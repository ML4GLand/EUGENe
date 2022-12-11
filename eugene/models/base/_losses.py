import torch.nn.functional as F

def adversarial_loss(self, y_hat, y):
    return F.binary_cross_entropy(y_hat, y)

LOSS_REGISTRY = {
    "mse": F.mse_loss,
    "poisson": F.poisson_nll_loss,
    "bce": F.binary_cross_entropy_with_logits,
    "cross_entropy": F.cross_entropy,
    "adversarial": adversarial_loss,
}



