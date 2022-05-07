# Classics
import os
import torch
import importlib
import numpy as np
from argparse import ArgumentParser

# Model defs
from eugene.models.hybrid import hybrid
from eugene.models.cnn import CNN
from eugene.models.fcn import FCN
from eugene.models.rnn import RNN

# Data def
from eugene.dataloading.SeqDataModule import SeqDataModule

# Define the cli
cli = ArgumentParser()
subparsers = cli.add_subparsers(dest="subcommand")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def subcommand(args=[], parent=subparsers):
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)
    return decorator

def argument(*name_or_flags, **kwargs):
    return ([*name_or_flags], kwargs)

# Grab the config
def parse_config(config):
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities.cli import LightningArgumentParser
    parser = LightningArgumentParser()
    #parser.add_lightning_class_args(Trainer, nested_key="trainer")
    parser.add_lightning_class_args(SeqDataModule, nested_key="data")
    #parser.add_lightning_class_args(hybrid, nested_key="model")
    yml = parser.parse_path(cfg_path=config)
    return yml

def load_data(path):
    if "yaml" in path:
        data_yml = parse_config(path)
        datamod = SeqDataModule(**data_yml["data"])
        datamod.setup()
        dataloader = datamod.test_dataloader()
        return dataloader
            
def load_from_pl_ckt(ckt_pth, model_type):
    if model_type in ["fcn", "cnn", "rnn"]:
        module = getattr(importlib.import_module(f"eugene.models.{model_type}"), model_type.upper())
    elif model_type in ["hybrid"]:
        module = getattr(importlib.import_module(f"eugene.models.{model_type.lower()}"), model_type.lower())
    model = module.load_from_checkpoint(checkpoint_path=ckt_pth, map_location=device)
    return model

def load_model(path, model_type):
    if "ckpt" in path:
        return load_from_pl_ckt(path, model_type) 
            
@subcommand([argument("--model", type=str, help="The model to interpret"),
             argument("--data", type=str, help="The data to score importances"),
             argument("--model_type", type=str, default="hybrid", help="The model type (e.g. CNN)"),
             argument("--out", type=str, default="./", help="Output directory")])
def score(args):  
    from eugene.interpret.nn_explain import get_importances
    model = load_model(args.model, args.model_type)
    model.eval()
    dataloader = load_data(args.data)
    imps = get_importances(model, dataloader, device=device)
    np.save(os.path.join(args.out, "nt_importances"), imps)

@subcommand([argument("--model", type=str, help="The model to interpret"),
             argument("--model_type", type=str, default="hybrid", help="The model type (e.g. CNN)"),
             argument("--out", type=str, default="./", help="Output directory")])
def pwm(args):
    from eugene.interpret.nn_explain import get_first_conv_layer
    model = load_model(args.model, args.model_type)
    model.eval()
    pwms = get_first_conv_layer(model).detach().numpy()
    np.save(os.path.join(args.out, "pwms"), pwms)
     
if __name__ == "__main__":   
    args = cli.parse_args()
    print(f"Executing on {device}")
    if args.subcommand is None:
        cli.print_help()
    else:
        args.func(args)