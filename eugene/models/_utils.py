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


def init_from_motifs(
    model,
    motifs: Union[PathLike, Dict[str, Motif]],
    layer_name: str,
    kernel_name: str = None,
    kernel_number: int = None,
    module_number: int = None,
):
    """Initialize a model's convolutional layer from a set of motifs.

    Parameters
    ----------
    model : LightningModule
        The model to initialize.
    layer_name : str
        The name of the layer to initialize.
    motifs : Union[PathLike, Dict[str, Motif]]
        A path to a file containing motifs, or a dictionary of motifs.
    kwargs : dict
        Additional arguments to pass to `model.init_from_motifs`.
    """
    if isinstance(motifs, PathLike):
        motifs = MinimalMEME(motifs)
    if kernel_name is None:
        assert module_number is not None
        assert isinstance(model.__getattr__(layer_name).module[module_number], torch.nn.Conv1d)
        layer_size = model.__getattr__(layer_name).module[module_number].weight.size()
    else:
        assert kernel_number is not None
        assert isinstance(
            model.__getattr__(layer_name).__getattr__(kernel_name)[kernel_number],
            torch.Tensor,
        )
        layer_size = (
            model.__getattr__(layer_name).__getattr__(kernel_name)[kernel_number].size()
        )
    #print(layer_size)
    kernel = _create_kernel_matrix(layer_size, motifs)
    init_conv(model, kernel, layer_name, kernel_name, kernel_number, module_number=module_number)
