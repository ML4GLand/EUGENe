import os
from os import PathLike
from typing import Dict, List, Type, Union

import seqdata as sd
import xarray as xr
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from eugene import settings

# Note that CSVLogger is currently hanging training with SequenceModule right now
# Note that if you use wandb logger, it comes with a few extra steps. Show a notebook for this
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
    gpus: int = None,
    logger: Union[str, Logger] = "tensorboard",
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    early_stopping_metric: str = "val_loss_epoch",
    early_stopping_patience=5,
    early_stopping_verbose=False,
    model_checkpoint_k=1,
    model_checkpoint_monitor: str = "val_loss_epoch",
    seed: int = None,
    return_trainer: bool = False,
    **kwargs,
):
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


# Have a couple of fit methods that are meant to take in a SeqData and call the above function
def fit_sequence_module(
    model: LightningModule,
    sdata: "xr.Dataset",
    seq_key: str = None,
    target_keys: Union[str, List[str]] = None,
    in_memory: bool = False,
    train_key: str = "train_val",
    epochs: int = 10,
    gpus: int = None,
    batch_size: int = None,
    num_workers: int = None,
    prefetch_factor: int = None,
    transforms=None,
    drop_last=True,
    logger: str = "tensorboard",
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    early_stopping_metric: str = "val_loss_epoch",
    early_stopping_patience=5,
    early_stopping_verbose=False,
    model_checkpoint_k=1,
    model_checkpoint_monitor: str = "val_loss_epoch",
    seed: int = None,
    return_trainer: bool = False,
    **kwargs,
):
    """
    Train the model using PyTorch Lightning.

    Parameters
    ----------
    model : BaseModel
        The model to train.
    sdata : SeqData
        The SeqData object to train on.
    target_keys : str or list of str
        The target keys in sdata's seqs_annot attribute to train on.
    train_key : str
        The key in sdata's seqs_annot attribute to split into train and validation set
    epochs : int
        The number of epochs to train for.
    gpus : int
        The number of gpus to use. EUGENe will automatically use all available gpus if available.
    batch_size : int
        The batch size to use.
    num_workers : int
        The number of workers to use for the dataloader.
    log_dir : PathLike
        The directory to save the logs to.
    name : str
        The name of the experiment.
    version : str
        The version of the experiment.
    train_dataset :Dataset
        The training dataset to use. If None, will be created from sdata.
    val_dataset :Dataset
        The validation dataset to use. If None, will be created from sdata.
    train_dataloader : DataLoader
        The training dataloader to use. If None, will be created from train_dataset.
    val_dataloader : DataLoader
        The validation dataloader to use. If None, will be created from val_dataset.
    transforms : list of str
        The sequence transforms to apply to the data.
    transform_kwargs : dict
        The keyword arguments to pass to the sequence transforms.
    early_stopping_metric : str
        The metric to use for early stopping.
    early_stopping_patience : int
        The number of epochs to wait before stopping.
    early_stopping_verbose : bool
        Whether to print early stopping messages.
    seed : int
        The seed to use for reproducibility.
    verbosity : int
        The verbosity level.
    kwargs : dict
        Additional keyword arguments to pass to the PL Trainer.

    Returns
    -------
    trainer : Trainer
        The PyTorch Lightning Trainer object.
    """

    # Set-up dataloaders
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    if target_keys is not None:
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        if len(target_keys) == 1:
            sdata["target"] = sdata[target_keys[0]]
        else:
            sdata["target"] = xr.concat(
                [sdata[target_key] for target_key in target_keys], dim="_targets"
            ).transpose("_sequence", "_targets")
        nan_mask = sdata['target'].isnull()
        if sdata["target"].ndim > 1:
            nan_mask = nan_mask.any('_targets')
        print(f"Dropping {nan_mask.sum().compute().item()} sequences with NaN targets.")
    if in_memory:
        print(f"Loading {seq_key} and {target_keys} into memory")
        sdata[seq_key].load()
        sdata["target"].load()
    sdata[train_key].load()
    train_sdata = sdata.sel(_sequence=(sdata[train_key] == True).compute())  # noqa
    val_sdata = sdata.sel(_sequence=(sdata[train_key] == False).compute())  # noqa
    train_dataloader = sd.get_torch_dataloader(
        train_sdata,
        sample_dims=["_sequence"],
        variables=[seq_key, "target"],
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
        variables=[seq_key, "target"],
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


def fit_profile_module():
    """
    Fit a profile module.
    """
    pass
