import os
import logging
import numpy as np
import pandas as pd
from typing import Union, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataload import SeqData, SeqDataset
from ..utils._decorators import track
from .._settings import settings


@track
def predictions(
   model: LightningModule,
   sdata: SeqData = None,
   target_keys: Union[str, List[str]] = None,
   gpus: int = None,
   batch_size: int = None,
   num_workers: int = None,
   out_dir: os.PathLike = None,
   name: str = None,
   version: str = "",
   file_label: str = "",
   prefix: str = "",
   suffix: str = "",
   sdataset: SeqDataset = None,
   sdataloader: DataLoader = None,
   seq_transforms: List = None,
   transform_kwargs={},
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
   target_keys: Union[str, List[str]]
      The target_keys to predict on.
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
   target_keys = [target_keys] if type(target_keys) == str else target_keys 
   out_dir = out_dir if out_dir is not None else settings.output_dir
   model_name = model.strand + model.__class__.__name__ + "_" + model.task
   name = name if name is not None else model_name
   out_dir = os.path.join(out_dir, name, version)

    # Save the final predictions to sdata if applicable
   if sdata is not None:
      if sdata.names is None:
         sdata.names = sdata.seqs_annot.index
      sdataset = sdata.to_dataset(
         target_keys=target_keys,
         seq_transforms=seq_transforms,
         transform_kwargs=transform_kwargs,
      )
      sdataloader = sdataset.to_dataloader(
         batch_size=batch_size, num_workers=num_workers, shuffle=False
      )
   elif sdataset is not None:
      sdataloader = sdataset.to_dataloader(
         batch_size=batch_size, num_workers=num_workers, shuffle=False
      )
   else:
      assert sdataloader is not None, "No data to predict on."
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
   sannot_cols = [f"{prefix}{lab}_predictions{suffix}" for lab in target_keys]
   ordered_preds = preds.loc[sdata.seqs_annot.index].astype(float)
   sdata.seqs_annot[sannot_cols] = ordered_preds
   return sdata if copy else None


@track
def train_val_predictions(
   model: LightningModule,
   sdata: SeqData = None,
   target_keys: Union[str, List[str]] = None,
   train_key: str = "train_val",
   gpus: int = None,
   batch_size: int = None,
   num_workers: int = None,
   out_dir: os.PathLike = None,
   name: str = None,
   version: str = "",
   prefix: str = "",
   suffix: str = "",
   train_dataset: SeqDataset = None,
   val_dataset: SeqDataset = None,
   train_dataloader: DataLoader = None,
   val_dataloader: DataLoader = None,
   seq_transforms: List = None,
   transform_kwargs={},
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
   target_keys: Union[str, List[str]]
      The target_keys to predict on.
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
   target_keys = [target_keys] if type(target_keys) == str else target_keys 
   out_dir = out_dir if out_dir is not None else settings.output_dir
   model_name = model.strand + model.__class__.__name__ + "_" + model.task
   name = name if name is not None else model_name
   out_dir = os.path.join(out_dir, name, version)

   # Save the final predictions to sdata if applicable
   if train_dataloader is not None:
      assert val_dataloader is not None
   elif train_dataset is not None:
      assert val_dataset is not None
      train_dataloader = DataLoader(
         train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
      )
      val_dataloader = DataLoader(
         val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
      )
   elif sdata is not None:
      train_idx = np.where(sdata.seqs_annot[train_key] == True)[0]
      train_dataset = sdata[train_idx].to_dataset(
         target_keys=target_keys,
         seq_transforms=seq_transforms,
         transform_kwargs=transform_kwargs,
      )
      train_dataloader = train_dataset.to_dataloader(
         batch_size=batch_size, num_workers=num_workers, shuffle=False
      )
      val_idx = np.where(sdata.seqs_annot[train_key] == False)[0]
      val_dataset = sdata[val_idx].to_dataset(
         target_keys=target_keys,
         seq_transforms=seq_transforms,
         transform_kwargs=transform_kwargs,
      )
      val_dataloader = val_dataset.to_dataloader(
         batch_size=batch_size, num_workers=num_workers, shuffle=False
      )
   else:
      raise ValueError("No data to predict on.")
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
   sdata.seqs_annot[[f"{prefix}{label}_predictions{suffix}" for label in target_keys]] = preds.loc[
      sdata.seqs_annot.index
   ].astype(float)
   return sdata if copy else None
