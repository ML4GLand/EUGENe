import importlib
from typing import Union
import os
from os import PathLike
import yaml
from .._settings import settings


def list_available_layers(model):
    """List all layers in a model"""
    return [name for name, _ in model.named_modules() if len(name) > 0]


def get_layer(model, layer_name):
    """Get a layer from a model by name"""
    return dict([*model.named_modules()])[layer_name]


def load_config(config_path: Union[str, PathLike], **kwargs):
    """Instantiate a module or architecture from a config file

    This function is used to instantiate a module or architecture from a
    config file. The config file must be a YAML file with parameters from
    the module or architecture as well as the name of the module or
    architecture. For example, to instantiate a CNN within a SequenceModule,
    the config file might look like this:

    ```yaml
    module: SequenceModule
    model:
        model_name: simple_cnn
        arch_name: CNN
        arch:
            input_len: 100
            output_dim: 1
            conv_kwargs:
                input_channels: 4
                conv_channels: [32]
                conv_kernels: [13]
                conv_strides: [1]
                pool_kernels: [2]
                pool_strides: [2]
                dropout_rates: 0.3
                batchnorm: True
                activations: relu
            dense_kwargs:
                hidden_dims: [64]
                dropout_rates: 0.2
                batchnorm: True
    task: regression
    loss_fxn: mse
    optimizer: adam
    optimizer_lr: 0.001
    ```

    The `module` parameter is the name of the LightningModule to instantiate in eugene.models.
    The `arch_name` parameter is the name of the architecture to instantiate in eugene.models.zoo.
    The `arch` parameter contains all the arguments for the CNN class in eugene.models.zoo._basic_models
    The conv_kwargs and dense_kwargs are Conv1DTower and DenseBlock respectively in eugene.models.base
    The parameters task, loss_fxn, optimizer, and optimizer_lr are all arguments for SequenceModule.

    If a "module" parameter is not passed in, this function assumes that we just want to instantiate
    an architecture. For example, to instantiate a CNN, the config file might look like this:

    ```yaml
    model:
        model_name: simple_cnn
        arch_name: CNN
        arch:
            input_len: 100
            output_dim: 1
            conv_kwargs:
                input_channels: 4
                conv_channels: [32]
                conv_kernels: [13]
                conv_strides: [1]
                pool_kernels: [2]
                pool_strides: [2]
                dropout_rates: 0.3
                batchnorm: True
                activations: relu
            dense_kwargs:
                hidden_dims: [64]
                dropout_rates: 0.2
                batchnorm: True
    ```

    where we have removed the module, task, loss_fxn, optimizer, and optimizer_lr parameters.
    This will return an instance of the CNN class in eugene.models.zoo._basic_models as an nn.Module.

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
        return model
    else:
        raise ValueError("Config file must contain either a 'model' or 'module' key")
