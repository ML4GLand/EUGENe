import torch
import torch.nn as nn
import torch.nn.init as init
from os import PathLike
from typing import Union, Dict
from ..dataload.motif._motif import Motif, MinimalMEME, _create_kernel_matrix

initializer_dict = {
    "uniform": init.uniform_,
    "normal": init.normal_,
    "constant": init.constant_,
    "eye": init.eye_,
    "dirac": init.dirac_,
    "xavier_uniform": init.xavier_uniform_,
    "xavier_normal": init.xavier_normal_,
    "kaiming_uniform": init.kaiming_uniform_,
    "kaiming_normal": init.kaiming_normal_,
    "orthogonal": init.orthogonal_,
    "sparse": init.sparse_,
    "ones": init.ones_,
    "zeros": init.zeros_
}


def _init_weights(
    module, 
    initializer,
    **kwargs
):
    """Initialize the weights of a module.

    Parameters
    ----------
    module : torch.nn.Module
        The module to initialize.
    initializer : str, optional
        The name of the initializer to use, by default "xavier_uniform"
    **kwargs
        Additional arguments to pass to the initializer.
    """
    if initializer in initializer_dict:
        init_func = initializer_dict[initializer]
    elif callable(initializer):
        init_func = initializer
    else:
        raise ValueError(
            f"Initializer {initializer} not found in initializer_dict or is not callable."
        )
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        print(f"Initializing {module} with {initializer}")
        init_func(module.weight)
    elif isinstance(module, nn.ParameterList):
        for param in module:
            if  param.dim() > 1:
                print(f"Initializing {param} with {initializer}")
                init_func(param)


def init_weights(
    model,
    initializer="kaiming_normal",
    **kwargs
):
    """Initialize the weights of a model.

    Parameters
    ----------
    model : LightningModule
        The model to initialize.
    initializer : str, optional
        The name of the initializer to use, by default "kaiming_normal"
    **kwargs
        Additional arguments to pass to the initializer.
    """
    model.apply(lambda m: _init_weights(m, initializer, **kwargs))


def init_conv(
    model,
    weights,
    module_name: str = "convnet",
    module_number: int = 0,
    kernel_name: str = None,
    kernel_number: int = None,
):
    if kernel_name is None:
        assert module_number is not None
        assert isinstance(
            model.__getattr__(module_name).module[module_number], nn.Conv1d
        )
        model.__getattr__(module_name).module[
            module_number
        ].weight = nn.Parameter(weights)
    else:
        assert kernel_number is not None
        assert isinstance(
            model.__getattr__(module_name).__getattr__(kernel_name)[kernel_number],
            torch.Tensor,
        )
        model.__getattr__(module_name).__getattr__(kernel_name)[
            kernel_number
        ] = nn.Parameter(weights)


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
            model.__getattr__(module_name).module[module_number], nn.Conv1d
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
