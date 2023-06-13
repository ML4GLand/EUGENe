from os import PathLike
from typing import List, Union
import seqdata as sd
import numpy as np
import xarray as xr
import ray
import importlib
from eugene import dataload as dl
from eugene import models, settings
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import (
    ASHAScheduler,
    MedianStoppingRule,
    PopulationBasedTraining,
)
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from torch.utils.data import DataLoader


def hyperopt_with_tune(
    config: dict,
    sdata=None,
    seq_key: str = "ohe_seq",
    target_keys: Union[str, List[str]] = None,
    train_key: str = "train_val",
    epochs: int = 10,
    gpus: int = None,
    num_workers: int = None,
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
    seq_transforms: List[str] = None,
    drop_last=True,
    seed: int = None,
):
    module_name = config.pop("module")
    arch_config = config["model"].pop("arch")
    arch_name = config["model"]["arch_name"]
    arch_type = getattr(importlib.import_module("eugene.models.zoo"), arch_name)
    arch = arch_type(**arch_config)
    module_type = getattr(importlib.import_module("eugene.models"), module_name)
    model = module_type(arch=arch, **config["model"])
    print(f"Model: {model}")
    models.init_weights(model)
    gpus = gpus if gpus is not None else settings.gpus
    print(f"Using {gpus} GPUs.")
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    print(f"Using {num_workers} dataloader workers.")
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    print(f"Logging to {log_dir}.")
    name = name if name is not None else config["model"]["model_name"]
    print(f"Experiment name: {name}.")
    seed_everything(seed, workers=True) if seed is not None else seed_everything(
        settings.seed
    )
    batch_size = config["batch_size"]
    if train_dataloader is not None:
        assert val_dataloader is not None
    elif sdata is not None:
        if target_keys is not None:
            sdata["target"] = xr.concat(
                [sdata[target_key] for target_key in target_keys], dim="_targets"
            ).transpose("_sequence", "_targets")
            targs = sdata["target"].values
            if len(targs.shape) == 1:
                nan_mask = np.isnan(targs)
            else:
                nan_mask = np.any(np.isnan(targs), axis=1)
            print(f"Dropping {nan_mask.sum()} sequences with NaN targets.")
            sdata = sdata.isel(_sequence=~nan_mask)
        train_mask = np.where(sdata[train_key])[0]
        train_sdata = sdata.isel(_sequence=train_mask)
        val_sdata = sdata.isel(_sequence=~train_mask)
        train_dataloader = sd.get_torch_dataloader(
            train_sdata,
            sample_dims=["_sequence"],
            variables=[seq_key, "target"],
            transforms=seq_transforms,
            prefetch_factor=None,
            shuffle=True,
            drop_last=drop_last,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        val_dataloader = sd.get_torch_dataloader(
            val_sdata,
            sample_dims=["_sequence"],
            variables=[seq_key, "target"],
            transforms=seq_transforms,
            prefetch_factor=None,
            shuffle=False,
            drop_last=drop_last,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError("No data provided to train on.")
    logger = TensorBoardLogger(log_dir, name=name, version=version)
    callbacks = []
    metrics = {"loss": "val_loss_epoch"}
    callbacks.append(TuneReportCallback(metrics, on="validation_end"))
    trainer = Trainer(
        max_epochs=epochs,
        devices=gpus,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


scheduler_dict = {
    "ASHAScheduler": ASHAScheduler,
    "MedianStoppingRule": MedianStoppingRule,
    "PopulationBasedTraining": PopulationBasedTraining,
}
default_scheduler_args = {
    "ASHAScheduler": {
        "metric": "loss",
        "mode": "min",
        "max_t": 10,
        "grace_period": 1,
        "reduction_factor": 2,
    },
    "MedianStoppingRule": {"metric": "loss", "mode": "min", "grace_period": 1},
    "PopulationBasedTraining": {"metric": "loss", "mode": "min", "grace_period": 1},
}
algo_dict = {
    "BayesOptSearch": BayesOptSearch,
    "HyperOptSearch": HyperOptSearch,
    "BasicVariantGenerator": BasicVariantGenerator,
}
default_algo_args = {
    "BayesOptSearch": {"metric": "loss", "mode": "min"},
    "HyperOptSearch": {"metric": "loss", "mode": "min"},
    "BasicVariantGenerator": {},
}


def hyperopt(
    config,
    sdata=None,
    seq_key: str = "ohe_seq",
    target_keys: Union[str, List[str]] = None,
    train_key: str = "train_val",
    scheduler="ASHAScheduler",
    algorithm="BasicVariantGenerator",
    num_samples: int = 10,
    epochs: int = 10,
    gpus: int = None,
    cpus: int = 1,
    num_workers: int = None,
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
    seq_transforms: List[str] = None,
    seed: int = None,
    scheduler_kwargs: dict = None,
    algorithm_kwargs: dict = None,
    **kwargs,
):
    """Perform hyperparameter optimization using HyperOpt."""
    print("Performing hyperparameter optimization using HyperOpt.")
    trainable = tune.with_parameters(
        hyperopt_with_tune,
        sdata=sdata,
        seq_key=seq_key,
        target_keys=target_keys,
        train_key=train_key,
        epochs=epochs,
        gpus=gpus,
        num_workers=num_workers,
        log_dir=log_dir,
        name=name,
        version=version,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        seq_transforms=seq_transforms,
        seed=seed,
        **kwargs,
    )
    if scheduler_kwargs is None or len(scheduler_kwargs) == 0:
        scheduler_kwargs = default_scheduler_args[scheduler]
    scheduler = scheduler_dict[scheduler](**scheduler_kwargs)
    if algorithm_kwargs is None or len(algorithm_kwargs) == 0:
        algorithm_kwargs = default_algo_args[algorithm]
    algo = algo_dict[algorithm](**algorithm_kwargs)
    analysis = tune.run(
        trainable,
        config=config,
        scheduler=scheduler,
        search_alg=algo,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        storage_path=settings.logging_dir,
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-val_loss_epoch",
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        name=name,
    )
    return analysis
