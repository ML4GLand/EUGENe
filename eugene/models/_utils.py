import importlib
import torch
from os import PathLike
from typing import Union, Dict, List, Tuple, Optional
from .base import init_conv
from ..utils import Motif, MinimalMEME
from ..utils._motif import _create_kernel_matrix


def load_config(arch, model_config):
    from pytorch_lightning.utilities.cli import LightningArgumentParser

    parser = LightningArgumentParser()
    model_type = getattr(importlib.import_module("eugene.models"), arch)
    parser.add_lightning_class_args(model_type, nested_key="model")
    model_yml = parser.parse_path(cfg_path=model_config)
    model = model_type(**model_yml["model"])
    return model
