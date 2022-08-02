import os
import logging
import numpy as np
import pandas as pd
from typing import Union, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataloading import SeqData, SeqDataset
from ..utils._decorators import track
from .._settings import settings

logging.disable(logging.ERROR)
seed_everything(settings.seed, workers=True)
logging.disable(logging.NOTSET)


@track
def predictions(
    model: LightningModule,
    sdata: SeqData = None,
    target: Union[str, List[str]] = None,
    gpus: int = None,
    batch_size: int = None,
    num_workers: int = None,
    out_dir: os.PathLike = None,
    name: str = None,
    version: str = "",
    file_label: str = "",
    sdataset: SeqDataset = None,
    sdataloader: DataLoader = None,
    seq_transforms: List = None,
    transform_kwargs={"transpose": True},
    copy: bool = False,
):
    """
    Predict on the model.

    Params:
    -------

    model: LightningModule
       The model to predict on.
    sdata: SeqData
       The data to predict on.
    target: Union[str, List[str]]
       The target to predict on.
    gpus: int
       The number of GPUs to use.
    batch_size: int
       The batch size to use.
    num_workers: int
       The number of workers to use.
    out_dir: os.PathLike
       The directory to save the predictions to.
    name: str
       The name of the model.
    version: str
       The version of the model.
    file_label: str
       The label to add to the file name.
    sdataset: SeqDataset
       The dataset to predict on.
    sdataloader: DataLoader
       The dataloader to predict on.
    seq_transforms: List
       The sequence transforms to use.
    transform_kwargs: dict
       The transform kwargs to use.
    copy: bool
       Whether to copy the data or not.

    Returns:
    --------

    """
    gpus = gpus if gpus is not None else settings.gpus
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    target = [target] if type(target) == str else target
    out_dir = out_dir if out_dir is not None else settings.output_dir
    model_name = model.strand + model.__class__.__name__ + "_" + model.task
    name = name if name is not None else model_name
    out_dir = os.path.join(out_dir, name, version)

    # Save the final predictions to sdata if applicable
    if sdata is not None:
        sdataset = sdata.to_dataset(
            target=target,
            seq_transforms=seq_transforms,
            transform_kwargs=transform_kwargs,
        )
        sdataloader = sdataset.to_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        if out_dir is not None:
            from ..utils._custom_callbacks import PredictionWriter

            predictor = Trainer(
                logger=False,
                callbacks=PredictionWriter(output_dir=out_dir, file_label=file_label),
                gpus=gpus,
            )
        else:
            predictor = Trainer(logger=False, gpus=gpus)
        ps = np.concatenate(predictor.predict(model, sdataloader), axis=0)
        num_outs = model.output_dim
        preds = pd.DataFrame(index=ps[:, 0], data=ps[:, 1 : num_outs + 1])
        sannot_cols = [f"{lab}_predictions" for lab in target]
        ordered_preds = preds.loc[sdata.seqs_annot.index].astype(float)
        sdata.seqs_annot[sannot_cols] = ordered_preds
        return sdata if copy else None


@track
def train_val_predictions(
    model: LightningModule,
    sdata: SeqData = None,
    target: Union[str, List[str]] = None,
    train_key: str = "train",
    gpus: int = None,
    batch_size: int = None,
    num_workers: int = None,
    out_dir: os.PathLike = None,
    name: str = None,
    version: str = "",
    file_label: str = "",
    sdataset: SeqDataset = None,
    sdataloader: DataLoader = None,
    seq_transforms: List = None,
    transform_kwargs={"transpose": True},
    copy: bool = False,
):
    """
    Predict on the model.

    Params:
    -------

    model: LightningModule
       The model to predict on.
    sdata: SeqData
       The data to predict on.
    target: Union[str, List[str]]
       The target to predict on.
    gpus: int
       The number of GPUs to use.
    batch_size: int
       The batch size to use.
    num_workers: int
       The number of workers to use.
    out_dir: os.PathLike
       The directory to save the predictions to.
    name: str
       The name of the model.
    version: str
       The version of the model.
    file_label: str
       The label to add to the file name.
    sdataset: SeqDataset
       The dataset to predict on.
    sdataloader: DataLoader
       The dataloader to predict on.
    seq_transforms: List
       The sequence transforms to use.
    transform_kwargs: dict
       The transform kwargs to use.
    copy: bool
       Whether to copy the data or not.

    Returns:
    --------

    """
    gpus = gpus if gpus is not None else settings.gpus
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    target = [target] if type(target) == str else target
    out_dir = out_dir if out_dir is not None else settings.output_dir
    model_name = model.strand + model.__class__.__name__ + "_" + model.task
    name = name if name is not None else model_name
    out_dir = os.path.join(out_dir, name, version)

    # Save the final predictions to sdata if applicable
    if sdata is not None:
        train_idx = np.where(sdata.seqs_annot[train_key] == True)[0]
        train_dataset = sdata[train_idx].to_dataset(
            target=target,
            seq_transforms=seq_transforms,
            transform_kwargs=transform_kwargs,
        )
        train_dataloader = train_dataset.to_dataloader(
            batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        val_idx = np.where(sdata.seqs_annot[train_key] == False)[0]
        val_dataset = sdata[val_idx].to_dataset(
            target=target,
            seq_transforms=seq_transforms,
            transform_kwargs=transform_kwargs,
        )
        val_dataloader = val_dataset.to_dataloader(
            batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        if out_dir is not None:
            from ..utils._custom_callbacks import PredictionWriter

            train_predictor = Trainer(
                logger=False,
                callbacks=PredictionWriter(out_dir, file_label="train"),
                gpus=gpus,
            )
            val_predictor = Trainer(
                logger=False,
                callbacks=PredictionWriter(out_dir, file_label="val"),
                gpus=gpus,
            )
        else:
            train_predictor = Trainer(logger=False, gpus=gpus)
            val_predictor = Trainer(logger=False, gpus=gpus)
        t = np.concatenate(train_predictor.predict(model, train_dataloader), axis=0)
        v = np.concatenate(val_predictor.predict(model, val_dataloader), axis=0)
        num_outs = model.output_dim
        preds = pd.concat(
            [
                pd.DataFrame(index=t[:, 0], data=t[:, 1 : num_outs + 1]),
                pd.DataFrame(index=v[:, 0], data=v[:, 1 : num_outs + 1]),
            ],
            axis=0,
        )
        sdata.seqs_annot[[f"{label}_predictions" for label in target]] = preds.loc[
            sdata.seqs_annot.index
        ].astype(float)
        return sdata if copy else None
