import numpy as np
from os import PathLike
from seqdata import SeqData
from typing import Union, List
from .._settings import settings
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning import LightningModule, Trainer, seed_everything


def fit(
    model: LightningModule,
    sdata: SeqData = None,
    target_keys: Union[str, List[str]] = None,
    train_key: str = "train_val",
    epochs: int = 10,
    gpus: int = None,
    batch_size: int = None,
    num_workers: int = None,
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
    seq_transforms: List[str] = None,
    transform_kwargs: dict = {},
    early_stopping_metric: str = "val_loss",
    drop_last=True,
    early_stopping_callback: bool = True,
    early_stopping_patience=5,
    early_stopping_verbose=False,
    model_checkpoint_k = 1,
    model_checkpoint_monitor: str ="val_loss",
    seed: int = None,
    return_trainer: bool = False,
    **kwargs
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
    seq_transforms : list of str
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
    # Set training parameters
    gpus = gpus if gpus is not None else settings.gpus
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    model_name = model.__class__.__name__
    name = name if name is not None else model_name
    seed_everything(seed, workers=True) if seed is not None else seed_everything(settings.seed)

    # Set-up dataloaders
    if train_dataloader is not None:
        assert val_dataloader is not None
    elif train_dataset is not None:
        assert val_dataset is not None
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    elif sdata is not None:
        if target_keys is not None:
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
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=True,
            drop_last=drop_last
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
            drop_last=drop_last
        )
    else:
        raise ValueError("No data provided to train on.")
    
    # Set-up callbacks
    logger = CSVLogger(log_dir, name=name, version=version)
    callbacks = []
    if model_checkpoint_monitor is not None:
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=logger.log_dir + "/checkpoints", 
            save_top_k=model_checkpoint_k, 
            monitor=model_checkpoint_monitor
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
    trainer = Trainer(
        max_epochs=epochs, 
        logger=logger, 
        devices=gpus, 
        accelerator="auto",
        callbacks=callbacks, 
        **kwargs
    )
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )
    if return_trainer:
        return trainer