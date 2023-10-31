import os
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    """Custom prediction writer for PyTorch Lightning trainer.

    Parameters
    ----------

    output_dir : str
        Directory to write predictions to.
    file_label : str
        Label to append to the file name.
    write_interval : str, optional
        When to write predictions. One of "epoch" or "step". Defaults to "epoch".
    

    Examples
    --------
    >>> from eugene.evaluate._utils import PredictionWriter
    >>> from pytorch_lightning import Trainer

    >>> model = ...
    >>> dataloader = ...
    >>> trainer = Trainer(callbacks=[PredictionWriter("predictions", "test")])
    >>> trainer.predict(model, dataloader)    
    """
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
