import importlib

import torch
import yaml

from .base._initializers import init_weights


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    module_name = config.pop("module")
    model_params = config.pop("model")
    arch_name = model_params["arch_name"]
    arch = model_params["arch"]
    model_type = getattr(importlib.import_module("eugene.models"), arch_name)
    model = model_type(**arch)
    module_type = getattr(importlib.import_module("eugene.models"), module_name)
    module = module_type(model, **config)
    return module

def load_model(model_path):
    model_state = torch.load(model_path)
    arch = model_state["hyper_parameters"]["arch"]
    model_type = getattr(importlib.import_module("eugene.models"), arch)
    model = model_type(**model_state["hyper_parameters"])
    model.load_state_dict(model_state["state_dict"])
    return model

def get_model(arch, model_config):
    model_type = getattr(importlib.import_module("eugene.models"), arch)
    model = model_type(**model_config)
    return model
