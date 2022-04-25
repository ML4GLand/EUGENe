import os
import pandas as pd
import numpy as np
import torch

from pytorch_lightning.callbacks import BasePredictionWriter

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: str, prefix= ""):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.prefix = prefix

    def write_on_batch_end(self, trainer, pl_module: 'LightningModule', prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int):
        np.save(os.path.join(self.output_dir, dataloader_idx, "{}_predictions".format(str(batch_idx))), predictions)

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions, batch_indices):
        predictions = np.concatenate(predictions[0], axis=0)
        pred_df = pd.DataFrame(data=predictions, columns=["NAME", "PREDICTION", "TARGET"])
        pred_df.to_csv(os.path.join(self.output_dir, self.prefix + "predictions.tsv"), sep="\t", index=False)