from cmath import log
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataloading import SeqData, SeqDataset
from torch.utils.data import DataLoader
from ..utils._decorators import track
from ..utils import suppress_stdout
from .._settings import settings

with suppress_stdout():
   seed_everything(settings.seed, workers=True)


@track
def predictions(
   model: LightningModule,
   sdata: SeqData = None,
   sdataset: SeqDataset = None,
   sdataloader: DataLoader = None,
   seq_transforms=None,
   label = "",
   target_label = None,
   gpus=None,
   batch_size: int = None,
   num_workers: int = None,
   log_dir = None,
   out_dir = None,
   copy=False
):
   """
   Predict on the model.
   """
   gpus = gpus if gpus is not None else settings.gpus
   batch_size = batch_size if batch_size is not None else settings.batch_size
   num_workers = num_workers if num_workers is not None else settings.dl_num_workers
   target_label = [target_label] if type(target_label) == str else target_label
   print(target_label)

   # Save the final predictions to sdata if applicable
   if sdata is not None:
      sdataset = sdata.to_dataset(label=target_label, seq_transforms=seq_transforms, transform_kwargs={"transpose": True})
      sdataloader = sdataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)

      if out_dir is not None:
         from ..utils._custom_callbacks import PredictionWriter
         predictor = Trainer(logger=False, callbacks=PredictionWriter(out_dir + f"{label}_"), gpus=gpus)
      else:
         predictor = Trainer(logger=False, gpus=gpus)

      ps = np.concatenate(predictor.predict(model, sdataloader), axis=0)
      num_outs = model.output_dim
      preds = pd.DataFrame(index=ps[:, 0], data=ps[:, 1:num_outs+1])
      sdata.seqs_annot[[f"{label}_PREDICTIONS" for label in target_label]] = preds.loc[sdata.seqs_annot.index].astype(float)
      return sdata if copy else None


@track
def train_val_predictions(
   model: LightningModule,
   sdata: SeqData = None,
   sdataset: SeqDataset = None,
   sdataloader: DataLoader = None,
   seq_transforms=None,
   target_label = "TARGETS",
   train_idx_label = "TRAIN",
   gpus=None,
   batch_size: int = None,
   num_workers: int = None,
   log_dir = None,
   out_dir = None,
   copy=False
):
   """
   Predict on the model.
   """
   gpus = gpus if gpus is not None else settings.gpus
   batch_size = batch_size if batch_size is not None else settings.batch_size
   num_workers = num_workers if num_workers is not None else settings.dl_num_workers

   # Save the final predictions to sdata if applicable
   if sdata is not None:
      train_idx = np.where(sdata.seqs_annot[train_idx_label] == True)[0]
      train_dataset = sdata[train_idx].to_dataset(label=target_label, seq_transforms=seq_transforms, transform_kwargs={"transpose": True})
      train_dataloader = train_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)
      val_idx = np.where(sdata.seqs_annot[train_idx_label] == False)[0]
      val_dataset = sdata[val_idx].to_dataset(label=target_label, seq_transforms=seq_transforms, transform_kwargs={"transpose": True})
      val_dataloader = val_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)

      if out_dir is not None:
         from ..utils._custom_callbacks import PredictionWriter
         train_predictor = Trainer(logger=False, callbacks=PredictionWriter(out_dir + "train_"), gpus=gpus)
         val_predictor = Trainer(logger=False, callbacks=PredictionWriter(out_dir + "val_"), gpus=gpus)
      else:
         train_predictor = Trainer(logger=False, gpus=gpus)
         val_predictor = Trainer(logger=False, gpus=gpus)

      t = np.concatenate(train_predictor.predict(model, train_dataloader), axis=0)
      v = np.concatenate(val_predictor.predict(model, val_dataloader), axis=0)
      preds = pd.concat([pd.DataFrame(index=t[:, 0], data=t[:, 1]), pd.DataFrame(index=v[:, 0], data=v[:, 1])], axis=0)
      sdata.seqs_annot[f"{target_label}_PREDICTIONS"] = preds.loc[sdata.seqs_annot.index].astype(float)
      return sdata if copy else None
