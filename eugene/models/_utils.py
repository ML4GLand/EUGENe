import importlib
from os import PathLike
from ._initialize import init_weights


def load_config(arch, model_config):
    from pytorch_lightning.utilities.cli import LightningArgumentParser

    parser = LightningArgumentParser()
    model_type = getattr(importlib.import_module("eugene.models"), arch)
    parser.add_lightning_class_args(model_type, nested_key="model")
    model_yml = parser.parse_path(cfg_path=model_config)
    model = model_type(**model_yml["model"])
    return model


def get_model(arch, model_config):
    model_type = getattr(importlib.import_module("eugene.models"), arch)
    model = model_type(**model_config)
    return model


def prep_new_model(
    arch,
    config
):
    # Instantiate the model
    model = load_config(
        arch=arch,
        model_config=config
    )

    # Initialize the model prior to conv filter initialization
    init_weights(model)
 
    # Return the model
    return model 