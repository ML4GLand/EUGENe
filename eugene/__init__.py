"""eugene"""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

from ._constants import REGISTRY_KEYS
from ._settings import settings

from . import preprocess as pp
from . import dataload as dl
from . import datasets
from . import models
from . import train
from . import evaluate
from . import interpret
from . import plot as pl
from . import external
from . import utils

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "eugene-tools"
__version__ = importlib_metadata.version(package_name)

settings.verbosity = logging.INFO

# This prevents double output.
eugene_logger = logging.getLogger("eugene")
eugene_logger.propagate = False

__all__ = [
    "settings",
    "REGISTRY_KEYS",
    "datasets",
    "dataload",
    "preprocess",
    "models",
    "train",
    "evaluate",
    "interpret",
    "plot",
    "external",
    "utils",
]
