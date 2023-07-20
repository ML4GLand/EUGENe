from pathlib import Path
from typing import Union

import torch


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
        seed: int = 13,
        progress_bar_style: str = "tqdm",
        rc_context: dict = None,
        dpi: int = 300,
        batch_size: int = 128,
        gpus: int = None,
        dl_num_workers: int = 0,
        dl_pin_memory_gpu_training: bool = False,
        dataset_dir: str = "./eugene_data/",
        logging_dir: str = "./eugene_logs/",
        output_dir: str = "./eugene_output/",
        config_dir: str = "./eugene_configs/",
        figure_dir: str = "./eugene_figures/",
    ):
        self.seed = seed
        if progress_bar_style not in ["tqdm"]:
            raise ValueError("Progress bar style must be in ['tqdm']")
        self.progress_bar_style = progress_bar_style
        self.rc_context = rc_context
        self.dpi = dpi
        self.batch_size = batch_size
        self.gpus = 1 if torch.cuda.is_available() else 0 if gpus is None else gpus
        self.dl_num_workers = dl_num_workers
        self.dl_pin_memory_gpu_training = dl_pin_memory_gpu_training
        self.dataset_dir = dataset_dir
        self.logging_dir = logging_dir
        self.output_dir = output_dir
        self.config_dir = config_dir
        self.figure_dir = figure_dir

    @property
    def seed(self) -> int:
        """Random seed for torch and numpy."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Random seed for torch and numpy."""
        self._seed = seed

    @property
    def progress_bar_style(self) -> str:
        """Library to use for progress bar."""
        return self._pbar_style

    @progress_bar_style.setter
    def progress_bar_style(self, pbar_style: str = "tqdm"):
        """Library to use for progress bar."""
        self._pbar_style = pbar_style

    @property
    def rc_context(self) -> dict:
        """Matplotlib rc_context."""
        return self._rc_context

    default_rc_context = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    @rc_context.setter
    def rc_context(self, rc_context: dict):
        """Matplotlib rc_context."""
        if rc_context is None:
            self._rc_context = self.default_rc_context
        else:
            self._rc_context = rc_context

    @property
    def dpi(self) -> int:
        """Matplotlib dpi."""
        return self._dpi

    @dpi.setter
    def dpi(self, dpi: int):
        """Matplotlib dpi."""
        self._dpi = dpi

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
        self._dataset_dir = Path(dataset_dir).resolve()

    @property
    def output_dir(self) -> Path:
        """Directory for saving output (default `'./eugene_output/'`)."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir: Union[str, Path]):
        self._output_dir = Path(output_dir).resolve()

    @property
    def config_dir(self) -> Path:
        """Directory for config files (default `'./eugene_config/'`)."""
        return self._config_dir

    @config_dir.setter
    def config_dir(self, config_dir: Union[str, Path]):
        self._config_dir = Path(config_dir).resolve()

    @property
    def figure_dir(self) -> Path:
        """Directory for saving figures (default `'./eugene_figures/'`)."""
        return self._figure_dir

    @figure_dir.setter
    def figure_dir(self, figure_dir: Union[str, Path]):
        self._figure_dir = Path(figure_dir).resolve()


settings = EugeneConfig()
