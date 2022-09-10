import importlib
import torch
from os import PathLike
from typing import Union, Dict, List, Tuple, Optional
from ..dataload.motif._motif import Motif, MinimalMEME, _create_kernel_matrix


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)


def init_conv(
    model,
    weights: torch.tensor,
    module_name: str = "convnet",
    module_number: int = 0,
    kernel_name: str = None,
    kernel_number: int = None,
):
    if kernel_name is None:
        assert module_number is not None
        assert isinstance(
            model.__getattr__(module_name).module[module_number], torch.nn.Conv1d
        )
        model.__getattr__(module_name).module[
            module_number
        ].weight = torch.nn.Parameter(weights)
    else:
        assert kernel_number is not None
        assert isinstance(
            model.__getattr__(module_name).__getattr__(kernel_name)[kernel_number],
            torch.Tensor,
        )
        model.__getattr__(module_name).__getattr__(kernel_name)[
            kernel_number
        ] = torch.nn.Parameter(weights)


def init_from_motifs(
    model,
    motifs: Union[PathLike, Dict[str, Motif]],
    module_name: str,
    module_number: int = None,
    kernel_name: str = None,
    kernel_number: int = None,
    convert_to_pwm=True,
):
    """Initialize a model's convolutional layer from a set of motifs.

    Parameters
    ----------
    model : LightningModule
        The model to initialize.
    module_name : str
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
        assert isinstance(
            model.__getattr__(module_name).module[module_number], torch.nn.Conv1d
        )
        layer_size = model.__getattr__(module_name).module[module_number].weight.size()
    else:
        assert kernel_number is not None
        assert isinstance(
            model.__getattr__(module_name).__getattr__(kernel_name)[kernel_number],
            torch.Tensor,
        )
        layer_size = (
            model.__getattr__(module_name)
            .__getattr__(kernel_name)[kernel_number]
            .size()
        )
    kernel_mtx = _create_kernel_matrix(
        layer_size, motifs, convert_to_pwm=convert_to_pwm
    )
    init_conv(
        model=model,
        weights=kernel_mtx,
        module_name=module_name,
        module_number=module_number,
        kernel_name=kernel_name,
        kernel_number=kernel_number,
    )
