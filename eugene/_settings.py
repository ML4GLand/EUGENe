import logging
import os
from pathlib import Path
from typing import Union
from typing import Any, Union, Optional, Iterable, TextIO
from typing import Tuple, List, ContextManager

import pytorch_lightning as pl
import torch
from rich.console import Console
from rich.logging import RichHandler

from ._compat import Literal

eugene_logger = logging.getLogger("eugene")

def _type_check(var: Any, varname: str, types: Union[type, Tuple[type, ...]]):
    if isinstance(var, types):
        return
    if isinstance(types, type):
        possible_types_str = types.__name__
    else:
        type_names = [t.__name__ for t in types]
        possible_types_str = "{} or {}".format(
            ", ".join(type_names[:-1]), type_names[-1]
        )
    raise TypeError(f"{varname} must be of type {possible_types_str}")

class EugeneConfig:
    """
    Config manager for eugene.
    Examples
    --------
    To set the seed
    >>> eugene.settings.seed = 42
    To set the batch size for functions like `eugene.get_latent_representation`
    >>> eugene.settings.batch_size = 1024
    To set the progress bar style, choose one of "rich", "tqdm"
    >>> eugene.settings.progress_bar_style = "rich"
    To set the verbosity
    >>> import logging
    >>> eugene.settings.verbosity = logging.INFO
    To set pin memory for GPU training
    >>> eugene.settings.dl_pin_memory_gpu_training = True
    To set the number of threads PyTorch will use
    >>> eugene.settings.num_threads = 2
    To prevent Jax from preallocating GPU memory on start (default)
    >>> eugene.settings.jax_preallocate_gpu_memory = False
    """

    def __init__(
        self,
        verbosity: int = logging.INFO,
        progress_bar_style: Literal["rich", "tqdm"] = "tqdm",
        batch_size: int = 128,
        seed: int = 13,
        datasetdir = "./data/",
        logging_dir: str = "./eugene_log/",
        dl_num_workers: int = 0,
        dl_pin_memory_gpu_training: bool = False,
        jax_preallocate_gpu_memory: bool = False,
    ):

        self.seed = seed
        self.batch_size = batch_size
        if progress_bar_style not in ["rich", "tqdm"]:
            raise ValueError("Progress bar style must be in ['rich', 'tqdm']")
        self.progress_bar_style = progress_bar_style
        self.datasetdir = datasetdir
        self.logging_dir = logging_dir
        self.dl_num_workers = dl_num_workers
        self.dl_pin_memory_gpu_training = dl_pin_memory_gpu_training
        self._num_threads = None
        self.jax_preallocate_gpu_memory = jax_preallocate_gpu_memory
        self.verbosity = verbosity

    @property
    def batch_size(self) -> int:
        """
        Minibatch size for loading data into the model.
        This is only used after a model is trained. Trainers have specific
        `batch_size` parameters.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """
        Minibatch size for loading data into the model.
        This is only used after a model is trained. Trainers have specific
        `batch_size` parameters.
        """
        self._batch_size = batch_size

    @property
    def dl_num_workers(self) -> int:
        """Number of workers for PyTorch data loaders (Default is 0)."""
        return self._dl_num_workers

    @dl_num_workers.setter
    def dl_num_workers(self, dl_num_workers: int):
        """Number of workers for PyTorch data loaders (Default is 0)."""
        self._dl_num_workers = dl_num_workers

    @property
    def dl_pin_memory_gpu_training(self) -> int:
        """Set `pin_memory` in data loaders when using a GPU for training."""
        return self._dl_pin_memory_gpu_training

    @dl_pin_memory_gpu_training.setter
    def dl_pin_memory_gpu_training(self, dl_pin_memory_gpu_training: int):
        """Set `pin_memory` in data loaders when using a GPU for training."""
        self._dl_pin_memory_gpu_training = dl_pin_memory_gpu_training

    @property
    def logging_dir(self) -> Path:
        """Directory for training logs (default `'./eugene_log/'`)."""
        return self._logging_dir

    @logging_dir.setter
    def logging_dir(self, logging_dir: Union[str, Path]):
        self._logging_dir = Path(logging_dir).resolve()

    @property
    def datasetdir(self) -> Path:
        """Directory for example (default `'./data/'`)."""
        return self._datasetdir

    @datasetdir.setter
    def datasetdir(self, datasetdir: Union[str, Path]):
        _type_check(datasetdir, "datasetdir", (str, Path))
        self._datasetdir = Path(datasetdir).resolve()

    @property
    def num_threads(self) -> None:
        """Number of threads PyTorch will use."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, num: int):
        """Number of threads PyTorch will use."""
        self._num_threads = num
        torch.set_num_threads(num)

    @property
    def progress_bar_style(self) -> str:
        """Library to use for progress bar."""
        return self._pbar_style

    @progress_bar_style.setter
    def progress_bar_style(self, pbar_style: Literal["tqdm", "rich"]):
        """Library to use for progress bar."""
        self._pbar_style = pbar_style

    @property
    def seed(self) -> int:
        """Random seed for torch and numpy."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Random seed for torch and numpy."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pl.utilities.seed.seed_everything(seed)
        self._seed = seed

    @property
    def verbosity(self) -> int:
        """Verbosity level (default `logging.INFO`)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: Union[str, int]):
        """
        Sets logging configuration for eugene based on chosen level of verbosity.
        If "eugene" logger has no StreamHandler, add one.
        Else, set its level to `level`.
        Parameters
        ----------
        level
            Sets "eugene" logging level to `level`
        force_terminal
            Rich logging option, set to False if piping to file output.
        """
        self._verbosity = level
        eugene_logger.setLevel(level)
        if len(eugene_logger.handlers) == 0:
            console = Console(force_terminal=True)
            if console.is_jupyter is True:
                console.is_jupyter = False
            ch = RichHandler(
                level=level, show_path=False, console=console, show_time=False
            )
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            eugene_logger.addHandler(ch)
        else:
            eugene_logger.setLevel(level)

    def reset_logging_handler(self):
        """
        Resets "eugene" log handler to a basic RichHandler().
        This is useful if piping outputs to a file.
        """
        eugene_logger.removeHandler(eugene_logger.handlers[0])
        ch = RichHandler(level=self._verbosity, show_path=False, show_time=False)
        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)
        eugene_logger.addHandler(ch)

    @property
    def jax_preallocate_gpu_memory(self):
        """
        Jax GPU memory allocation settings.
        If False, Jax will ony preallocate GPU memory it needs.
        If float in (0, 1), Jax will preallocate GPU memory to that
        fraction of the GPU memory.
        """
        return self._jax_gpu

    @jax_preallocate_gpu_memory.setter
    def jax_preallocate_gpu_memory(self, value: Union[float, bool]):
        # see https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html#gpu-memory-allocation
        if value is False:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        elif isinstance(value, float):
            if value >= 1 or value <= 0:
                raise ValueError("Need to use a value between 0 and 1")
            # format is ".XX"
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(value)[1:4]
        else:
            raise ValueError("value not understood, need bool or float in (0, 1)")
        self._jax_gpu = value


settings = EugeneConfig()