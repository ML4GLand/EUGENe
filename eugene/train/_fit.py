from cmath import log
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataloading import SeqData, SeqDataset
from torch.utils.data import DataLoader

from ..utils._decorators import track
seed_everything(13, workers=True)

@track
def fit(model: LightningModule,
        sdata: SeqData = None,
        train_dataset: SeqDataset = None,
        val_dataset: SeqDataset = None,
        batch_size: int = 32,
        num_workers: int = 0,
        epochs = 10,
        log_dir = "_logs",
        out_dir = None,
        train_idx_label = "TRAIN",
        target_label = "TARGETS",
        save_preds = True,
        early_stopping_path = None,
        early_stopping_metric = None,
        early_stopping_metric_min = None,
        early_stopping_metric_patience = None,
        copy=False):
   """
    Train the model.
    """

   # First try to train with a SeqDataset if available
   if train_dataset is not None:
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
      val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

   # Move on to sdata if not
   elif sdata is not None:
      train_idx = np.where(sdata.seqs_annot[train_idx_label] == True)[0]
      train_dataset = sdata[train_idx].to_dataset(label=target_label, seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
      train_dataloader = train_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)
      val_idx = np.where(sdata.seqs_annot[train_idx_label] == False)[0]
      val_dataset = sdata[val_idx].to_dataset(label=target_label, seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
      val_dataloader = val_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)

   # Set-up a trainer with a logger and callbacks (if applicable)
   logger = TensorBoardLogger(log_dir)
   trainer = Trainer(max_epochs=epochs, logger=logger)

   # Fit the model
   trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

   # Save the predictions to sdata if applicable
   if sdata is not None and save_preds:
      if out_dir is not None:
         from ..utils._custom_callbacks import PredictionWriter
         train_predictor= Trainer(logger=False, callbacks=PredictionWriter(out_dir + "train_"))
         val_predictor = Trainer(logger=False, callbacks=PredictionWriter(out_dir + "val_"))
      else:
         train_predictor = Trainer(logger=False)
         val_predictor = Trainer(logger=False)
      t = np.concatenate(train_predictor.predict(model, train_dataloader), axis=0)
      v = np.concatenate(val_predictor.predict(model, val_dataloader), axis=0)
      preds = pd.concat([pd.DataFrame(index=t[:, 0], data=t[:, 1]), pd.DataFrame(index=v[:, 0], data=v[:, 1])], axis=0)
      sdata.seqs_annot["PREDICTIONS"] = preds.loc[sdata.seqs_annot.index].astype(float)
      return sdata if copy else None
