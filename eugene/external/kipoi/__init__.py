import os as _os
import sys as _sys

_bin_dir = _os.path.dirname(_sys.executable)
_os.environ["PATH"] += _os.pathsep + _bin_dir

from ._wrappers import get_model_names, get_model
from .kipoi_veff.plot import seqlogo_heatmap