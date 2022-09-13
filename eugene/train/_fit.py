import logging
import numpy as np
import pandas as pd
from os import PathLike
from typing import Union, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from ..dataload import SeqData, SeqDataset
from .._settings import settings


def fit(
    model: LightningModule,
    sdata: SeqData = None,
    target_keys: Union[str, List[str]] = None,
    train_key: str = "train_val",
    epochs=10,
    gpus=None,
    batch_size: int = None,
    num_workers: int = None,
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    train_dataset: SeqDataset = None,
    val_dataset: SeqDataset = None,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
    seq_transforms: List[str] = None,
    transform_kwargs: dict = {},
    early_stopping_callback: bool = True,
    early_stopping_metric="val_loss",
    early_stopping_patience=5,
    early_stopping_verbose=False,
    seed: int = None,
    verbosity = None,
    **kwargs
):
    """
    Train the model.
    """
    gpus = gpus if gpus is not None else settings.gpus
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    model_name = model.strand + model.__class__.__name__ + "_" + model.task
    name = name if name is not None else model_name
    seed_everything(seed, workers=True) if seed is not None else seed_everything(settings.seed)
    if train_dataloader is not None:
        assert val_dataloader is not None
    elif train_dataset is not None:
        assert val_dataset is not None
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
    elif sdata is not None:
        assert target_keys is not None
        targs = sdata.seqs_annot[target_keys].values  
        if len(targs.shape) == 1:
            nan_mask = np.isnan(targs)
        else:
            nan_mask = np.any(np.isnan(targs), axis=1)
        print(f"Dropping {nan_mask.sum()} sequences with NaN targets.")
        sdata = sdata[~nan_mask]
        train_idx = np.where(sdata.seqs_annot[train_key] == True)[0]
        train_dataset = sdata[train_idx].to_dataset(
            target_keys=target_keys,
            seq_transforms=seq_transforms,
            transform_kwargs=transform_kwargs,
        )
        train_dataloader = train_dataset.to_dataloader(
            batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        val_idx = np.where(sdata.seqs_annot[train_key] == False)[0]
        val_dataset = sdata[val_idx].to_dataset(
            target_keys=target_keys,
            seq_transforms=seq_transforms,
            transform_kwargs=transform_kwargs,
        )
        val_dataloader = val_dataset.to_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        raise ValueError("No data provided to train on.")
    logger = TensorBoardLogger(log_dir, name=name, version=version)
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=logger.log_dir + "/checkpoints", save_top_k=1, monitor="val_loss"
        )
    )
    if early_stopping_callback:
        early_stopping_callback = EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            mode="min",
            verbose=early_stopping_verbose,
        )
        callbacks.append(early_stopping_callback)
    if model.scheduler is not None:
        callbacks.append(LearningRateMonitor())
    trainer = Trainer(
        max_epochs=epochs, logger=logger, gpus=gpus, callbacks=callbacks, **kwargs
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
