import yaml
import importlib
from .base._initializers import init_weights


def load_config(config_path):
    from pytorch_lightning.utilities.cli import LightningArgumentParser
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    arch = config["model"]["arch"]
    parser = LightningArgumentParser()
    model_type = getattr(importlib.import_module("eugene.models"), arch)
    parser.add_lightning_class_args(model_type, nested_key="model")
    model_yml = parser.parse_path(cfg_path=config_path)
    model = model_type(**model_yml["model"])
    return model

def get_model(arch, model_config):
    model_type = getattr(importlib.import_module("eugene.models"), arch)
    model = model_type(**model_config)
    return model

def prep_new_model(
    config
):
    # Instantiate the model
    model = load_config(config_path=config)

    # Initialize the model prior to conv filter initialization
    init_weights(model)
 
    # Return the model
    return model 