# Classics
import os
import torch
import importlib
import pandas as pd
import numpy as np
from argparse import ArgumentParser

# Data def
from eugene.dataloading.SeqDataModule import SeqDataModule

# Define the cli
cli = ArgumentParser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grab the config
def parse_config(config):
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities.cli import LightningArgumentParser
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(SeqDataModule, nested_key="data")
    yml = parser.parse_path(cfg_path=config)
    return yml

def load_data(path):
    if "yaml" in path:
        data_yml = parse_config(path)
        datamod = SeqDataModule(**data_yml["data"])
        datamod.setup()
        from archive.seq_utils import ascii_decode
        print(ascii_decode(datamod.val_dataloader().dataset[0][0]))
        return datamod

def load_from_pl_ckt(ckt_pth, model_type):
    from eugene.models.hybrid import hybrid
    from eugene.models.cnn import CNN
    from eugene.models.fcn import FCN
    from eugene.models.rnn import RNN
    if model_type in ["fcn", "cnn", "rnn"]:
        module = getattr(importlib.import_module(f"eugene.models.{model_type}"), model_type.upper())
    elif model_type in ["hybrid"]:
        module = getattr(importlib.import_module(f"eugene.models.{model_type.lower()}"), model_type.lower())
    model = module.load_from_checkpoint(checkpoint_path=ckt_pth, map_location=device)
    return model

def load_model(path, model_type):
    if "ckpt" in path:
        return load_from_pl_ckt(path, model_type)

def predict(args):
    from pytorch_lightning.utilities.seed import seed_everything
    print(f"Executing on {device}")
    seed_everything(args.seed, workers=True)
    model = load_model(args.model, args.model_type)
    model.eval()
    datamod = load_data(args.data)
    if datamod.test:
        dataloaders = {"test": datamod.test_dataloader()}
    else:
        dataloaders = {"train": datamod.train_dataloader(), "val": datamod.val_dataloader()}
    from pytorch_lightning import Trainer
    trainer = Trainer(gpus=1 if device.type == "cuda" else 0, logger=False)
    for loader in dataloaders.keys():
        print(loader)
        preds = trainer.predict(model=model, dataloaders=dataloaders[loader])
        preds = np.concatenate(preds)
        pred_df = pd.DataFrame(data=preds, columns=["NAME", "PREDICTION", "TARGET"])
        pred_df.to_csv(os.path.join(args.out, loader + "_preds.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    cli.add_argument("--model", type=str, help="The model to use")
    cli.add_argument("--data", type=str, help="The data to make predictions on")
    cli.add_argument("--model_type", type=str, default="hybrid", help="The model type (e.g. CNN)")
    cli.add_argument("--seed", type=int, default=13, help="random seed for reproducibility")
    cli.add_argument("--out", type=str, default="./", help="Output directory")
    args = cli.parse_args()
    predict(args)
