from cmath import log
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataloading import SeqData, SeqDataset
from torch.utils.data import DataLoader
seed_everything(13, workers=True)


def fit(model: LightningModule,
        sdata: SeqData = None,
        train_dataset: SeqDataset = None,
        val_dataset: SeqDataset = None,
        batch_size: int = 32,
        num_workers: int = 0,
        epochs = 10,
        log_dir = "_logs",
        train_idx_label = "TRAIN",
        target_label = "TARGETS",
        early_stopping_path = None,
        early_stopping_metric = None,
        early_stopping_metric_min = None,
        early_stopping_metric_patience = None):
   """
    Train the model.
    """

   logger = TensorBoardLogger(log_dir)
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
   trainer = Trainer(max_epochs=epochs,logger=logger)
   trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
