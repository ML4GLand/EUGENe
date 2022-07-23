from cmath import log
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataloading import SeqData, SeqDataset
from torch.utils.data import DataLoader
from ..utils import suppress_stdout
from .._settings import settings

with suppress_stdout():
   seed_everything(settings.seed, workers=True)


def fit(
   model: LightningModule,
   sdata: SeqData = None,
   train_idx_label = "TRAIN",
   target_label = "TARGETS",
   gpus = None,
   train_dataset: SeqDataset = None,
   val_dataset: SeqDataset = None,
   seq_transforms=None,
   batch_size: int = None,
   num_workers: int = None,
   epochs = 10,
   log_dir = None,
   early_stopping_path = None,
   early_stopping_metric = None,
   early_stopping_metric_min = None,
   early_stopping_metric_patience = None,
   **kwargs
   ):
   """
   Train the model.
   """

   batch_size = batch_size if batch_size is not None else settings.batch_size
   num_workers = num_workers if num_workers is not None else settings.dl_num_workers
   log_dir = log_dir if log_dir is not None else settings.logging_dir

   # First try to train with a SeqDataset if available
   if train_dataset is not None:
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
      val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

   # Move on to sdata if not
   elif sdata is not None:
      train_idx = np.where(sdata.seqs_annot[train_idx_label] == True)[0]
      train_dataset = sdata[train_idx].to_dataset(label=target_label, seq_transforms=seq_transforms, transform_kwargs={"transpose": True})
      train_dataloader = train_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)
      val_idx = np.where(sdata.seqs_annot[train_idx_label] == False)[0]
      val_dataset = sdata[val_idx].to_dataset(label=target_label, seq_transforms=seq_transforms, transform_kwargs={"transpose": True})
      val_dataloader = val_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)

   else:
      raise ValueError("No data provided to train on.")

   # Set-up a trainer with a logger and callbacks (if applicable)
   logger = TensorBoardLogger(log_dir)
   trainer = Trainer(max_epochs=epochs, logger=logger, gpus=gpus, **kwargs)

   # Fit the model
   trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
