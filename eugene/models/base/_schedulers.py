from torch.optim.lr_scheduler import ReduceLROnPlateau

SCHEDULER_REGISTRY = {"reduce_lr_on_plateau": ReduceLROnPlateau}
