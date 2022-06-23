"""eugene"""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

from ._constants import REGISTRY_KEYS
from ._settings import settings

from . import datasets
from . import dataloading as dl
from . import preprocessing as pp
from . import models
from . import train
from . import predict
from . import interpret
from . import plotting as pl
from . import utils
from . import external

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "eugene"
__version__ = importlib_metadata.version(package_name)

settings.verbosity = logging.INFO

# This prevents double output.
eugene_logger = logging.getLogger("eugene")
eugene_logger.propagate = False

__all__ = ["settings", "REGISTRY_KEYS", "datasets", "dataloading", "preprocessing", "models", "train", "predict", "interpret", "plotting", "external", "utils"]