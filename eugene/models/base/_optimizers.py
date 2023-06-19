import torch.optim as opt

OPTIMIZER_REGISTRY = {"adam": opt.Adam, "sgd": opt.SGD}


def configure_optimizer(
    model, lr=0.001, weight_decay=1e-6, decay_factor=0.1, patience=5, monitor="val_loss"
):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=decay_factor, patience=patience
            ),
            "monitor": monitor,
        },
    }
