"""eugene"""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

from ._settings import settings

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "eugene"
__version__ = importlib_metadata.version(package_name)

# This prevents double output.
eugene_logger = logging.getLogger("eugene")
eugene_logger.propagate = False

__all__ = [
    "settings",
    "dataload",
    "preprocess",
    "models",
    "train",
    "evaluate",
    "plot",
    "utils",
]
