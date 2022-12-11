import torch.nn.init as init


INITIALIZERS_REGISTRY = {
    "uniform": init.uniform_,
    "normal": init.normal_,
    "constant": init.constant_,
    "eye": init.eye_,
    "dirac": init.dirac_,
    "xavier_uniform": init.xavier_uniform_,
    "xavier_normal": init.xavier_normal_,
    "kaiming_uniform": init.kaiming_uniform_,
    "kaiming_normal": init.kaiming_normal_,
    "orthogonal": init.orthogonal_,
    "sparse": init.sparse_,
    "ones": init.ones_,
    "zeros": init.zeros_
}