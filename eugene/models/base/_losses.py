import torch
import torch.nn.functional as F


def adversarial_loss(y_hat, y):
    return F.binary_cross_entropy(y_hat, y)


def rmsle_loss(y_hat, y):
    torch.sqrt(F.mse_loss(torch.log(y_hat + 1), torch.log(y + 1)))


def tweedie_loss(y_hat, y):
    """
    Custom loss fuction designed to minimize the deviance using stochastic gradient descent
    tweedie deviance from McCullagh 1983

    """
    p = torch.tensor(1.5)
    QLL = QLL = torch.pow(y_hat, (-p)) * (
        ((y_hat * y) / (1 - p)) - ((torch.pow(y_hat, 2)) / (2 - p))
    )
    d = -2 * QLL(y_hat, y)
    return torch.mean(d)


def logcosh_loss(y_hat, y):
    ey_t = y - y_hat
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def MNLLLoss(logps, true_counts):
    """A loss function based on the multinomial negative log-likelihood.
    This loss function takes in a tensor of normalized log probabilities such
    that the sum of each row is equal to 1 (e.g. from a log softmax) and
    an equal sized tensor of true counts and returns the probability of
    observing the true counts given the predicted probabilities under a
    multinomial distribution. Can accept tensors with 2 or more dimensions
    and averages over all except for the last axis, which is the number
    of categories.
    Adapted from Alex Tseng.
    Parameters
    ----------
    logps: torch.tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.
    true_counts: torch.tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.
    Returns
    -------
    loss: float
            The multinomial log likelihood loss of the true counts given the
            predicted probabilities, averaged over all examples and all other
            dimensions.
    """

    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1)
    return -log_fact_sum + log_prod_fact - log_prod_exp


def log1pMSELoss(log_predicted_counts, true_counts):
    """A MSE loss on the log(x+1) of the inputs.
    This loss will accept tensors of predicted counts and a vector of true
    counts and return the MSE on the log of the labels. The squared error
    is calculated for each position in the tensor and then averaged, regardless
    of the shape.
    Note: The predicted counts are in log space but the true counts are in the
    original count space.
    Parameters
    ----------
    log_predicted_counts: torch.tensor, shape=(n, ...)
            A tensor of log predicted counts where the first axis is the number of
            examples. Important: these values are already in log space.
    true_counts: torch.tensor, shape=(n, ...)
            A tensor of the true counts where the first axis is the number of
            examples.
    Returns
    -------
    loss: torch.tensor, shape=(n, 1)
            The MSE loss on the log of the two inputs, averaged over all examples
            and all other dimensions.
    """

    log_true = torch.log(true_counts + 1)
    return torch.mean(torch.square(log_true - log_predicted_counts), dim=-1)


LOSS_REGISTRY = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "poisson": F.poisson_nll_loss,
    "bce": F.binary_cross_entropy_with_logits,
    "cross_entropy": F.cross_entropy,
    "adversarial": adversarial_loss,
    "log1pMSELoss": log1pMSELoss,
    "MNLLLoss": MNLLLoss,
}

DEFAULT_LOSS_REGISTRY = {
    "regression": "mse",
    "binary_classification": "bce",
    "multiclass_classification": "cross_entropy",
    "multilabel_classification": "bce",
}
