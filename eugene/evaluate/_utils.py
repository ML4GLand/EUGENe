import os
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, file_label: str, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.file_label = file_label

    def write_on_epoch_end(self, trainer, pl_module, outputs, batch_indices):
        outputs = np.concatenate(outputs, axis=0)
        num_outputs = pl_module.output_dim
        pred_cols = [f"predictions_{i}" for i in range(num_outputs)]
        target_cols = [f"target_{i}" for i in range(num_outputs)]
        pred_df = pd.DataFrame(data=outputs, columns=pred_cols + target_cols)
        pred_df.to_csv(
            os.path.join(self.output_dir, self.file_label) + "_predictions.tsv",
            sep="\t",
            index=False,
        )
