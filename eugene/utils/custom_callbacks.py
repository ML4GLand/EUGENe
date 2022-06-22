import os
import pandas as pd
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


@CALLBACK_REGISTRY
class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module: 'LightningModule', prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int):
        np.save(os.path.join(self.output_dir, dataloader_idx, "{}_predictions".format(str(batch_idx))), predictions)

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions, batch_indices):
        predictions = np.concatenate(predictions[0], axis=0)
        pred_df = pd.DataFrame(data=predictions, columns=["NAME", "PREDICTION", "TARGET"])
        out = self.output_dir.rsplit("/", maxsplit=1)[0]
        print(out)
        if not os.path.exists(out):
            os.makedirs(out)
        pred_df.to_csv(os.path.join(self.output_dir + "predictions.tsv"), sep="\t", index=False)

        
@CALLBACK_REGISTRY
class TrainAndValPredictionWriter(Callback):
    def __init__(self):
        self.train_preds = []
        self.val_preds = []

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        if trainer.current_epoch == trainer.max_epochs - 1:
            self.train_preds.append(outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1:
            predictions = np.concatenate(self.train_preds, axis=0)
            pred_df = pd.DataFrame(data=predictions, columns=["NAME", "PREDICTION", "TARGET"])
            pred_df.to_csv(os.path.join(self.output_dir + "train_predictions.tsv"), sep="\t", index=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
         if trainer.current_epoch == trainer.max_epochs - 1:
             self.val_preds.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
         if trainer.current_epoch == trainer.max_epochs - 1:
            predictions = np.concatenate(self.val_preds, axis=0)
            pred_df = pd.DataFrame(data=predictions, columns=["NAME", "PREDICTION", "TARGET"])
            pred_df.to_csv(os.path.join(self.output_dir + "val_predictions.tsv"), sep="\t", index=False)