import os
from os import PathLike
from typing import Dict, List, Type, Union, Literal, Optional

import seqdata as sd
import xarray as xr
from ..models import SequenceModule
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from eugene import settings

LOGGER_REGISTRY: Dict[str, Type[Logger]] = {
    "csv": CSVLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandbLogger,
}


def fit(
    model: LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    epochs: int = 10,
    gpus: Optional[int] = None,
    logger: Union[str, Logger] = "tensorboard",
    log_dir: Optional[PathLike] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    early_stopping_metric: str = "val_loss_epoch",
    early_stopping_patience: int = 5,
    early_stopping_verbose: bool = False,
    model_checkpoint_k: int = 1,
    model_checkpoint_monitor: str = "val_loss_epoch",
    seed: Optional[int] = None,
    return_trainer: bool = False,
    **kwargs,
) -> Optional[Trainer]:
    """Fit a model using PyTorch Lightning.

    This is a generic fit function that can be used to train any PyTorch LightningModule. All that's
    required is a LightningModule, a training dataloader, and optionally a validation dataloader.

    Parameters
    ----------
    model : LightningModule
        The model to train.
    train_dataloader : DataLoader
        The training dataloader to use.
    val_dataloader : DataLoader
        The validation dataloader to use.
    epochs : int
        The number of epochs to train for.
    gpus : int
        The number of gpus to use. EUGENe will automatically use all available gpus if available.
    logger : str or Logger
        The logger to use. If a string, must be one of "csv", "tensorboard", or "wandb".
    log_dir : PathLike
        The directory to save the logs to.
    name : str
        The name of the experiment. Appended to the end of the log directory
    version : str
        The version of the experiment. Appended to the end of the log directory/name
    early_stopping_metric : str
        The metric to use for early stopping.
    early_stopping_patience : int
        The number of epochs to wait before stopping.
    early_stopping_verbose : bool
        Whether to print early stopping messages.
    seed : int
        The seed to use for reproducibility.
    kwargs : dict
        Additional varword arguments to pass to the PL Trainer.

    Returns
    -------
    trainer : Trainer
        The PyTorch Lightning Trainer object.
    """
    # Set-up a seed
    seed_everything(seed, workers=True) if seed is not None else print("No seed set")

    # Logger
    logger = LOGGER_REGISTRY[logger](save_dir=log_dir, name=name, version=version)

    # Set-up callbacks
    callbacks = []
    if model_checkpoint_monitor is not None:
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                logger.save_dir, logger.name, logger.version, "checkpoints"
            ),
            save_top_k=model_checkpoint_k,
            monitor=model_checkpoint_monitor,
        )
        callbacks.append(model_checkpoint_callback)
    if early_stopping_metric is not None:
        early_stopping_callback = EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            mode="min",
            verbose=early_stopping_verbose,
        )
        callbacks.append(early_stopping_callback)
    if model.scheduler is not None:
        callbacks.append(LearningRateMonitor())

    # Trainer
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        devices=gpus,
        accelerator="auto",
        callbacks=callbacks,
        **kwargs,
    )

    # Fit
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    if return_trainer:
        return trainer


def fit_sequence_module(
    model: SequenceModule,
    sdata: xr.Dataset,
    seq_var: Optional[str] = None,
    target_vars: Optional[Union[str, List[str]]] = None,
    in_memory: bool = False,
    train_var: str = "train_val",
    epochs: int = 10,
    gpus: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    prefetch_factor: int = None,
    transforms: Optional[Dict] = None,
    drop_last: bool = False,
    logger: str = "tensorboard",
    log_dir: Optional[PathLike] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    early_stopping_metric: str = "val_loss_epoch",
    early_stopping_patience: int = 5,
    early_stopping_verbose: bool = False,
    model_checkpoint_k: int = 1,
    model_checkpoint_monitor: str = "val_loss_epoch",
    seed: Optional[int] = None,
    return_trainer: bool = False,
    **kwargs,
) -> Optional[Trainer]:
    """Fit a SequenceModule using PyTorch Lightning. 
    
    This function is a wrapper around the fit function, but builds the dataloaders from a SeqData object.

    Parameters
    ----------
    model : 
        The model to train.
    sdata : SeqData
        The SeqData object to train on.
    target_vars : str or list of str
        The target vars in sdata to use aas labels for training
    in_memory : bool
        Whether to load the data into memory before training. Default is False.
    train_var : str
        The var in sdata to use to split into train and validation set
    epochs : int
        The number of epochs to train for.
    gpus : int
        The number of gpus to use. EUGENe will automatically use all available gpus if available.
    batch_size : int
        The batch size to use.
    num_workers : int
        The number of workers to use for the dataloader.
    prefetch_factor : int
        The prefetch factor to use for the dataloader.
    transforms : dict
        The transforms to apply to the data. This should be a dictionary of the form
        {"var": transform function to apply}. See the documentation for SeqData for more
        information.
    drop_last : bool
        Whether to drop the last batch if it is smaller than the batch size.
    logger : str or Logger
        The logger to use. If a string, must be one of "csv", "tensorboard", or "wandb".
    log_dir : PathLike
        The directory to save the logs to.
    name : str
        The name of the experiment.
    version : str
        The version of the experiment.
    early_stopping_metric : str
        The metric to use for early stopping.
    early_stopping_patience : int
        The number of epochs to wait before stopping.
    early_stopping_verbose : bool
        Whether to print early stopping messages.
    model_checkpoint_k : int
        The number of models to save.
    model_checkpoint_monitor : str
        The metric to use for model checkpointing.
    seed : int
        The seed to use for reproducibility.
    return_trainer : bool
        Whether to return the trainer object.
    kwargs : dict
        Additional varword arguments to pass to the PL Trainer.

    Returns
    -------
    trainer : Trainer
        The PyTorch Lightning Trainer object.
    """

    # Set-up dataloaders
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    if target_vars is not None:
        if isinstance(target_vars, str):
            target_vars = [target_vars]
        if len(target_vars) == 1:
            sdata["target"] = sdata[target_vars[0]]
        else:
            sdata["target"] = xr.concat(
                [sdata[target_var] for target_var in target_vars], dim="_targets"
            ).transpose("_sequence", "_targets")
        nan_mask = sdata['target'].isnull()
        if sdata["target"].ndim > 1:
            nan_mask = nan_mask.any('_targets')
        print(f"Dropping {nan_mask.sum().compute().item()} sequences with NaN targets.")
    if in_memory:
        print(f"Loading {seq_var} and {target_vars} into memory")
        sdata[seq_var].load()
        sdata["target"].load()
    sdata[train_var].load()
    train_sdata = sdata.sel(_sequence=(sdata[train_var] == True).compute())  # noqa
    val_sdata = sdata.sel(_sequence=(sdata[train_var] == False).compute())  # noqa
    train_dataloader = sd.get_torch_dataloader(
        train_sdata,
        sample_dims=["_sequence"],
        variables=[seq_var, "target"],
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        transforms=transforms,
        shuffle=True,
        drop_last=drop_last,
    )
    val_dataloader = sd.get_torch_dataloader(
        val_sdata,
        sample_dims=["_sequence"],
        variables=[seq_var, "target"],
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        transforms=transforms,
        shuffle=False,
        drop_last=drop_last,
    )

    # Set training parameters
    gpus = gpus if gpus is not None else settings.gpus
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    model_name = model.__class__.__name__
    name = name if name is not None else model_name

    # Use the above fit function
    trainer = fit(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        gpus=gpus,
        logger=logger,
        log_dir=log_dir,
        name=name,
        version=version,
        early_stopping_metric=early_stopping_metric,
        early_stopping_patience=early_stopping_patience,
        early_stopping_verbose=early_stopping_verbose,
        model_checkpoint_k=model_checkpoint_k,
        model_checkpoint_monitor=model_checkpoint_monitor,
        seed=seed,
        return_trainer=return_trainer,
        **kwargs,
    )

    if return_trainer:
        return trainer
