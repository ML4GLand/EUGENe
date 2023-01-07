import torch
import torch.nn.functional as F

def adversarial_loss(y_hat, y):
    return F.binary_cross_entropy(y_hat, y)

def rmsle_loss(y_hat, y):
    torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(actual + 1)))
        
def tweedie_loss(y_hat, y):
    '''
    Custom loss fuction designed to minimize the deviance using stochastic gradient descent
    tweedie deviance from McCullagh 1983

    '''
    p = torch.tensor(1.5)
    QLL = QLL = torch.pow(y_hat, (-p))*(((y_hat*y)/(1-p)) - ((torch.pow(y_hat, 2))/(2-p)))
    d = -2*QLL(y_hat, y)
    return torch.mean(d)

def logcosh_loss(y_hat, y):
    ey_t = y - y_hat
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
    
LOSS_REGISTRY = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "poisson": F.poisson_nll_loss,
    "bce": F.binary_cross_entropy_with_logits,
    "cross_entropy": F.cross_entropy,
    "adversarial": adversarial_loss,
}



