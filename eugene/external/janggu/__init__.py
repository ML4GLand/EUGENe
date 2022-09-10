import os as _os
import sys as _sys

_bin_dir = _os.path.dirname(_sys.executable)
_os.environ["PATH"] += _os.pathsep + _bin_dir
from pybedtools import paths as _paths

_paths._set_bedtools_path(_bin_dir)
