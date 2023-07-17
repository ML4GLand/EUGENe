import gc
import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        gc.collect()
        torch.cuda.empty_cache()

        try:
            seed_everything(seed=42)

            numerai_model = NumeraiModel(
                dropout=config.dropout,
                initial_bn=config.initial_bn,
                learning_rate=config.learning_rate,
                wd=config.wd,
            )
            wandb_logger = WandbLogger(log_model=True)
            checkpoint_callback = ModelCheckpoint(monitor="1_val/sharpe", mode="max")
            trainer = Trainer(
                gpus=1,
                max_epochs=config.epochs,
                logger=wandb_logger,
                callbacks=[checkpoint_callback],
            )

            trainer.fit(numerai_model)

        except Exception as e:
            print(e)

        del numerai_model
        del wandb_logger
        del checkpoint_callback
        del trainer

        gc.collect()
        torch.cuda.empty_cache()
