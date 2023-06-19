import torch
import torch.nn as nn
from typing import Dict
import torch.nn.init as init
from .._utils import get_layer
from motifdata import to_kernel
from motifdata import MotifSet


INITIALIZERS_REGISTRY = {
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
    "zeros": init.zeros_,
}


def _init_weights(module, initializer):
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
    if initializer in INITIALIZERS_REGISTRY:
        init_func = INITIALIZERS_REGISTRY[initializer]
    elif callable(initializer):
        init_func = initializer
    else:
        raise ValueError(
            f"Initializer {initializer} not found in INITIALIZER_REGISTRY or is not callable."
        )
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        init_func(module.weight)
    elif isinstance(module, nn.ParameterList):
        for param in module:
            if param.dim() > 1:
                init_func(param)


def init_weights(model, initializer="kaiming_normal", **kwargs):
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


def init_motif_weights(
    model,
    layer_name,
    motifs,
    list_index=None,
    initializer="kaiming_normal",
    convert_to_pwm=True,
    divide_by_bg=False,
    motif_align="center",
    kernel_align="center",
):
    """Initialize the convolutional kernel of choice using a set of motifs

    This function is designed to initialize the convolutional kernels of a given layer of a model with a set of motifs.
    It has only been tested on nn.Conv1d layers and ParameterList layers that have a shape of (num_kernels, 4, kernel_size).
    Simply use the named module of the layer you want to initialize and pass it to this function. If the layer is a ParameterList,
    you must also pass the index of the kernel you want to initialize. If the layer is a Conv1d layer, you can pass None as the index.

    Parameters
    ----------
    model :
        The model to initialize.
    layer_name : str
        The name of the layer to initialize. You can use the list_available_layers function to get a list of available layers.
    motifs : MotifSet
        A MotifSet object containing the motifs to initialize the kernel with.
    list_index : int, optional
        The index of the kernel to initialize. Only required if the layer is a ParameterList layer, by default None

    Returns
    -------
    None

    Examples
    --------
    >>> from eugene import models
    >>> from motifdata import MotifSet
    >>> model = models.zoo.DeepSTARR(input_len=1000, output_dim=1)
    >>> motifs = MotifSet("path/to/motifs.txt")
    >>> list_available_layers(model)
    >>> init_motif_weights(model, "conv1d_tower.layers.0", motifs)
    """
    layer = get_layer(model, layer_name)
    if list_index is None:
        layer_size = layer.weight.size()
        kernel = torch.zeros(layer_size)
        INITIALIZERS_REGISTRY[initializer](kernel)
        pwms = to_kernel(
            motifs,
            kernel=kernel,
            convert_to_pwm=convert_to_pwm,
            divide_by_bg=divide_by_bg,
            motif_align=motif_align,
            kernel_align=kernel_align,
        )
        layer.weight = nn.Parameter(pwms)
    else:
        layer_size = layer[list_index].size()
        kernel = torch.zeros(layer_size)
        INITIALIZERS_REGISTRY[initializer](kernel)
        pwms = to_kernel(
            motifs,
            kernel=kernel,
            convert_to_pwm=convert_to_pwm,
            divide_by_bg=divide_by_bg,
            motif_align=motif_align,
            kernel_align=kernel_align,
        )
        layer[list_index] = nn.Parameter(pwms)
