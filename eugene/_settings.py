import logging
from pathlib import Path
from typing import Union
from typing import Any, Union, Optional, Iterable, TextIO
from typing import Tuple, List, ContextManager

import pytorch_lightning as pl
import torch
from rich.console import Console
from rich.logging import RichHandler

from ._compat import Literal

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPUs: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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
    >>> eugene.settings.seed = 13
    To set the default batch size for all functions
    >>> eugene.settings.batch_size = 1024
    To set the progress bar style, choose one of "rich", "tqdm"
    >>> eugene.settings.progress_bar_style = "rich"
    To set the verbosity
    >>> import logging
    >>> eugene.settings.verbosity = logging.INFO
    To set pin memory for GPU training
    >>> eugene.settings.dl_pin_memory_gpu_training = True
    """

    def __init__(
        self,
        verbosity: int = logging.INFO,
        progress_bar_style: Literal["rich", "tqdm"] = "tqdm",
        batch_size: int = 128,
        seed: int = 13,
        gpus: int = None,
        dataset_dir = "./datasets/",
        logging_dir: str = "./eugene_log/",
        output_dir: str = "./eugene_output/",
        config_dir: str = "./eugene_config/",
        dl_num_workers: int = 0,
        dl_pin_memory_gpu_training: bool = False,
    ):

        self.verbosity = verbosity
        if progress_bar_style not in ["rich", "tqdm"]:
            raise ValueError("Progress bar style must be in ['rich', 'tqdm']")
        self.progress_bar_style = progress_bar_style
        self.batch_size = batch_size
        self.seed = seed
        self.gpus = 1 if torch.cuda.is_available() else 0 if gpus is None else gpus
        self.dataset_dir =dataset_dir
        self.logging_dir = logging_dir
        self.output_dir = output_dir
        self.dl_num_workers = dl_num_workers
        self.dl_pin_memory_gpu_training = dl_pin_memory_gpu_training

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
    def gpus(self) -> int:
        """
        Number of GPUs to use.
        """
        return self._gpus

    @gpus.setter
    def gpus(self, gpus: int):
        """
        Number of GPUs to use.
        """
        self._gpus = gpus

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
    def dataset_dir(self) -> Path:
        """Directory for example (default `'./data/'`)."""
        return self._dataset_dir

    @dataset_dir.setter
    def dataset_dir(self, dataset_dir: Union[str, Path]):
        _type_check(dataset_dir, "dataset_dir", (str, Path))
        self._dataset_dir = Path(dataset_dir).resolve()

    @property
    def config_dir(self) -> Path:
        """Directory for config files (default `'./eugene_config/'`)."""
        return self._config_dir

    @config_dir.setter
    def config_dir(self, config_dir: Union[str, Path]):
        self._config_dir = Path(config_dir).resolve()

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

    @property
    def output_dir(self) -> Path:
        """Directory for saving output (default `'./eugene_output/'`)."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir: Union[str, Path]):
        self._output_dir = Path(output_dir).resolve()

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


settings = EugeneConfig()
