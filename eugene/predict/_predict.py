from cmath import log
import numpy as np
import pandas as pd
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
   save_preds=None,
):
   logger = TensorBoardLogger(log_dir)
   if sdata is not None:
      sdataset = sdata.to_dataset(label=target_label, seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
      sdataloader = sdataset.to_dataloader(batch_size=batch_size, num_workers=num_workers)

   elif sdataset is not None:
      sdataloader = DataLoader(sdataset, batch_size=batch_size, num_workers=num_workers)

   elif sdataset is not None:
      sdataloader = sdataloader

   if out_dir is not None:
      from ..utils import PredictionWriter
      trainer = Trainer(logger=False, callbacks=PredictionWriter(out_dir, write_interval="epoch"))
   else:
      trainer = Trainer(logger=False)

   t = np.concatenate(np.array(trainer.predict(model, sdataloader)), axis=0)
   preds = pd.DataFrame(index=t[:, 0], data=t[:, 1])

   if save_preds is not None:
      sdata.seqs_annot[save_preds] = preds.loc[sdata.seqs_annot.index].astype(float)
   else:
      return preds
