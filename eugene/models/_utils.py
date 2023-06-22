import importlib
import os
import torch
import yaml
from .._settings import settings


def get_layer(model, layer_name):
    return dict([*model.named_modules()])[layer_name]


def list_available_layers(model):
    return [name for name, _ in model.named_modules() if len(name) > 0]


def load_config(config_path, **kwargs):
    # If config path is just a filename, assume it's in the default config directory
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


def load_model(model_path):
    model_state = torch.load(model_path)
    arch = model_state["hyper_parameters"]["arch"]
    model_type = getattr(importlib.import_module("eugene.models.zoo"), arch)
    model = model_type(**model_state["hyper_parameters"])
    model.load_state_dict(model_state["state_dict"])
    return model
