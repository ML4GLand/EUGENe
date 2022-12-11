import torch.optim as opt

OPTIMIZER_REGISTRY = {
    "adam": opt.Adam, 
    "sgd": opt.SGD
}