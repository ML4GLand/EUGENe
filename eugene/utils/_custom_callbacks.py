import os
import pandas as pd
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


#@CALLBACK_REGISTRY
class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        out = self.output_dir.rsplit("/", maxsplit=1)[0]
        if not os.path.exists(out):
            os.makedirs(out)

    def write_on_batch_end(self, trainer, pl_module, outputs, batch_indices, batch, batch_idx: int, dataloader_idx: int):
        np.save(os.path.join(self.output_dir, dataloader_idx, "{}_outputs".format(str(batch_idx))), outputs)

    def write_on_epoch_end(self, trainer, pl_module, outputs, batch_indices):
        outputs = np.concatenate(outputs[0], axis=0)
        num_outputs = pl_module.output_dim
        print(num_outputs)
        pred_cols = [f"PREDICTIONS{i}" for i in range(num_outputs)]
        target_cols = [f"TARGET{i}" for i in range(num_outputs)]
        pred_df = pd.DataFrame(data=outputs, columns=["NAMES"] + target_cols + pred_cols)
        pred_df.to_csv(os.path.join(self.output_dir + "predictions.tsv"), sep="\t", index=False)


#@CALLBACK_REGISTRY
class TrainAndValPredictionWriter(Callback):
    """_summary_ only works with Hybrid

    Args:
        Callback (_type_): _description_
    """
    def __init__(self, output_dir: str):
        self.train_preds = []
        self.val_preds = []
        self.output_dir = output_dir
        out = self.output_dir.rsplit("/", maxsplit=1)[0]
        if not os.path.exists(out):
            os.makedirs(out)

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        if trainer.current_epoch == trainer.max_epochs - 1:
            self.train_preds.append(outputs["predictions"])

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1:
            outputs = np.concatenate(self.train_preds, axis=0)
            pred_df = pd.DataFrame(data=outputs, columns=["NAMES", "PREDICTIONS", "TARGETS"])
            pred_df.to_csv(os.path.join(self.output_dir + "train_predictions.tsv"), sep="\t", index=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
         if trainer.current_epoch == trainer.max_epochs - 1:
            self.val_preds.append(outputs["predictions"])

    def on_validation_epoch_end(self, trainer, pl_module):
         if trainer.current_epoch == trainer.max_epochs - 1:
            outputs = np.concatenate(self.val_preds, axis=0)
            pred_df = pd.DataFrame(data=outputs, columns=["NAMES", "PREDICTIONS", "TARGETS"])
            pred_df.to_csv(os.path.join(self.output_dir + "val_predictions.tsv"), sep="\t", index=False)
