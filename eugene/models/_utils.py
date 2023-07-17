import importlib
from typing import List, Tuple, Union
from pathlib import Path
import os
from os import PathLike
import torch
import yaml
from .._settings import settings


def list_available_layers(model):
    """List all layers in a model"""
    return [name for name, _ in model.named_modules() if len(name) > 0]


def get_layer(model, layer_name):
    """Get a layer from a model by name"""
    return dict([*model.named_modules()])[layer_name]


def load_config(
    config_path: Union[str, PathLike],
    **kwargs
):
    """Instantiate a module or architecture from a config file

    This function is used to instantiate a module or architecture from a
    config file. The config file must be a YAML file with TODO

    Parameters
    ----------
    config_path : str or PathLike
        Path to a YAML config file
    **kwargs
        Additional keyword arguments to pass to the module or architecture

    Returns
    -------
    Union[SequenceModule, ProfileModule, nn.Module]
    
    """
    if "/" not in config_path:
        config_path = os.path.join(settings.config_dir, config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if "module" in config:
        module_name = config.pop("module")
        model_params = config.pop("model")
        arch_name = model_params["arch_name"]
        arch = model_params["arch"]
        model_type = getattr(importlib.import_module("eugene.models.zoo"), arch_name)
        model = model_type(**arch)
        module_type = getattr(importlib.import_module("eugene.models"), module_name)
        module = module_type(model, **config, **kwargs)
        return module
    elif "model" in config:
        model_params = config.pop("model")
        arch = model_params["arch"]
        arch_name = model_params["arch_name"]
        model_type = getattr(importlib.import_module("eugene.models.zoo"), arch_name)
        model = model_type(**arch)
    else:
        raise ValueError("Config file must contain either a 'model' or 'module' key")
