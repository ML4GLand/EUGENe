from os import PathLike
from typing import List, Union, Optional
import seqdata as sd
import numpy as np
import xarray as xr
import importlib
from eugene import models, settings
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
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
    sdata: Optional[xr.Dataset] = None,
    seq_var: str = "ohe_seq",
    target_vars: Optional[Union[str, List[str]]] = None,
    train_var: str = "train_val",
    epochs: int = 10,
    gpus: Optional[int] = None,
    num_workers: Optional[int] = None,
    log_dir: Optional[PathLike] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    train_dataloader: Optional[DataLoader] = None,
    val_dataloader: Optional[DataLoader] = None,
    transforms: Optional[List[str]] = None,
    drop_last: bool = False,
    seed: Optional[int] = None,
):
    """The trainable for hyperparameter optimization using RayTune

    This function makes use of the PyTorch Lightning TuneReportCallback
    to perform hyperparameter optimization. This function serves as the
    callable (or trainable) argument to the tune.run function. The function
    is called with the config argument that should define the search space for
    the hyperparameter optimization as well as the model architecture and training
    parameters. The function also takes in either a SeqData object or  a PyTorch
    DataLoader object for training and validation. If a SeqData object is provided,
    the function will create the PyTorch DataLoader objects using the
    provided SeqData object. If a PyTorch DataLoader object is provided,
    the function will use the provided DataLoader objects for training and
    validation.

    Parameters
    ----------
    config : dict
        A dictionary containing the hyperparameter search space,
        model architecture, and training parameters.
        See the hyperopt tutorial for more information.
    sdata : Optional[xr.Dataset], optional
        A SeqData object containing the data to train on, by default None
    seq_var : str, optional
        The name of the sequence variable in the SeqData object,
        by default "ohe_seq"
    target_vars : Optional[Union[str, List[str]]], optional
        The name(s) of the target variable(s) in the SeqData object,
        by default None
    train_var : str, optional
        The name of the training variable in the SeqData object,
        by default "train_val"
    epochs : int, optional
        The number of epochs to train for, by default 10
    gpus : Optional[int], optional
        The number of GPUs to use for training, by default None
    num_workers : Optional[int], optional
        The number of workers to use for the PyTorch DataLoader,
        by default None
    log_dir : Optional[PathLike], optional
        The path to the directory to log to, by default None
    name : Optional[str], optional
        The name of the experiment, by default None
    version : Optional[str], optional
        The version of the experiment, by default None
    train_dataloader : Optional[DataLoader], optional
        A PyTorch DataLoader object to use for training,
        by default None
    val_dataloader : Optional[DataLoader], optional
        A PyTorch DataLoader object to use for validation,
        by default None
    transforms : Optional[List[str]], optional
        A list of transforms to apply to the data, by default None
    drop_last : bool, optional
        Whether or not to drop the last batch of data if it is smaller
        than the batch size, by default False
    seed : Optional[int], optional
        The seed to use for reproducibility, by default None
    """
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
        if target_vars is not None:
            sdata["target"] = xr.concat(
                [sdata[target_var] for target_var in target_vars], dim="_targets"
            ).transpose("_sequence", "_targets")
            targs = sdata["target"].values
            if len(targs.shape) == 1:
                nan_mask = np.isnan(targs)
            else:
                nan_mask = np.any(np.isnan(targs), axis=1)
            print(f"Dropping {nan_mask.sum()} sequences with NaN targets.")
            sdata = sdata.isel(_sequence=~nan_mask)
        train_mask = np.where(sdata[train_var])[0]
        train_sdata = sdata.isel(_sequence=train_mask)
        val_sdata = sdata.isel(_sequence=~train_mask)
        train_dataloader = sd.get_torch_dataloader(
            train_sdata,
            sample_dims=["_sequence"],
            variables=[seq_var, "target"],
            transforms=transforms,
            prefetch_factor=None,
            shuffle=True,
            drop_last=drop_last,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        val_dataloader = sd.get_torch_dataloader(
            val_sdata,
            sample_dims=["_sequence"],
            variables=[seq_var, "target"],
            transforms=transforms,
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
    config: dict,
    sdata: xr.Dataset = None,
    seq_var: str = "ohe_seq",
    target_vars: Optional[Union[str, List[str]]] = None,
    train_var: str = "train_val",
    scheduler: str = "ASHAScheduler",
    algorithm: str = "BasicVariantGenerator",
    num_samples: int = 10,
    epochs: int = 10,
    gpus: Optional[int] = None,
    cpus: int = 1,
    num_workers: Optional[int] = None,
    log_dir: Optional[PathLike] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    train_dataloader: Optional[DataLoader] = None,
    val_dataloader: Optional[DataLoader] = None,
    transforms: Optional[List[str]] = None,
    seed: Optional[int] = None,
    scheduler_kwargs: Optional[dict] = None,
    algorithm_kwargs: Optional[dict] = None,
    **kwargs,
):
    """Perform hyperparameter optimization using RayTune

    This function performs hyperparameter optimization using RayTune.
    The function takes in a dictionary defining the hyperparameter search space,
    model architecture, and training parameters. The function also takes in
    either a SeqData object or a PyTorch DataLoader object for training and validation.
    If a SeqData object is provided, the function will create the PyTorch DataLoader
    objects using the provided SeqData object. If a PyTorch DataLoader object is
    provided, the function will use the provided DataLoader objects for
    training and validation.

    Parameters
    ----------
    config : dict
        A dictionary containing the hyperparameter search space,
        model architecture, and training parameters.
        See the hyperopt tutorial for more information.
    sdata : xr.Dataset, optional
        A SeqData object containing the data to train on, by default None
    seq_var : str, optional
        The name of the sequence variable in the SeqData object,
        by default "ohe_seq"
    target_vars : Optional[Union[str, List[str]]], optional
        The name(s) of the target variable(s) in the SeqData object,
        by default None
    train_var : str, optional
        The name of the training variable in the SeqData object, by default "train_val"
    scheduler : str, optional
        The name of the scheduler to use for hyperparameter optimization,
        by default "ASHAScheduler"
    algorithm : str, optional
        The name of the algorithm to use for hyperparameter optimization,
        by default "BasicVariantGenerator"
    num_samples : int, optional
        The number of hyperparameter combinations to try, by default 10
    epochs : int, optional
        The number of epochs to train for, by default 10
    gpus : Optional[int], optional
        The number of GPUs to use for training, by default None
    cpus : int, optional
        The number of CPUs to use for training, by default 1
    num_workers : Optional[int], optional
        The number of workers to use for the PyTorch DataLoader,
        by default None
    log_dir : Optional[PathLike], optional
        The path to the directory to log to, by default None
    name : Optional[str], optional
        The name of the experiment, by default None
    version : Optional[str], optional
        The version of the experiment, by default None
    train_dataloader : Optional[DataLoader], optional
        A PyTorch DataLoader object to use for training, by default None
    val_dataloader : Optional[DataLoader], optional
        A PyTorch DataLoader object to use for validation, by default None
    transforms : Optional[List[str]], optional
        A list of transforms to apply to the data, by default None
    seed : Optional[int], optional
        The seed to use for reproducibility, by default None
    scheduler_kwargs : Optional[dict], optional
        A dictionary of keyword arguments to pass to the scheduler,
        by default None
    algorithm_kwargs : Optional[dict], optional
        A dictionary of keyword arguments to pass to the algorithm,
        by default None

    Returns
    -------
    analysis : ray.tune.Analysis
        The analysis object returned by ray.tune.run
    """
    trainable = tune.with_parameters(
        hyperopt_with_tune,
        sdata=sdata,
        seq_var=seq_var,
        target_vars=target_vars,
        train_var=train_var,
        epochs=epochs,
        gpus=gpus,
        num_workers=num_workers,
        log_dir=log_dir,
        name=name,
        version=version,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        transforms=transforms,
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
        # metric="loss",
        # mode="min",
        num_samples=num_samples,
        # storage_path=settings.logging_dir,
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-val_loss_epoch",
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        name=name,
    )
    return analysis
