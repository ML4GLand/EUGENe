from cmath import log
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataloading import SeqData, SeqDataset
from torch.utils.data import DataLoader

seed_everything(13, workers=True)


def predictions(
    model: LightningModule,
    sdata: SeqData = None,
    sdataset: SeqDataset = None,
    sdataloader: DataLoader = None,
    target_label = "TARGETS",
    batch_size: int = 1,
    num_workers: int = 0,
    log_dir="_logs",
    out_dir="./test_",
):
   logger = TensorBoardLogger(log_dir)
   if sdata is not None:
      sdataset = sdata.to_dataset(label=target_label, seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
      sdataloader = sdataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)

   elif sdataset is not None:
      sdataloader = DataLoader(sdataset, batch_size=batch_size, num_workers=num_workers)

   elif sdataset is not None:
      sdataloader = sdataloader

   logger = TensorBoardLogger(log_dir)
   if out_dir is not None:
      from ..utils import PredictionWriter
      trainer = Trainer(logger=False, callbacks=PredictionWriter(out_dir, write_interval="epoch"))
      return
   else:
      trainer = Trainer(logger=False)
      return trainer.predict(model, sdataloader)

def train_val_predictions(model: LightningModule,
   sdata: SeqData = None,
   train_dataset: SeqDataset = None,
   val_dataset: SeqDataset = None,
   target_label = "TARGETS",
   batch_size: int = 1,
   num_workers: int = 0,
   log_dir="_logs",
   out_dir="./"):
   """
   Predict on a training and val set
   """
   if sdata is not None:
      train_idx = np.where(sdata.seqs_annot[train_idx_label] == True)[0]
      train_dataset = sdata[train_idx].to_dataset(label=target_label, seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
      train_dataloader = train_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)
      val_idx = np.where(sdata.seqs_annot[train_idx_label] == False)[0]
      val_dataset = sdata[val_idx].to_dataset(label=target_label, seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
      val_dataloader = val_dataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)

   elif train_dataset is not None:
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
      val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

   logger = TensorBoardLogger(log_dir)
   if out_dir is not None:
      from ..utils import TrainAndValWriter
      trainer = Trainer(logger=False, callbacks=TrainAndValWriter(out_dir, write_interval="epoch"))
   else:
      trainer = Trainer(logger=False)
   trainer.predict(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
