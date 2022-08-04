import os
import sys

bin_dir = os.path.dirname(sys.executable)
os.environ["PATH"] += os.pathsep + bin_dir
from pybedtools import paths

paths._set_bedtools_path(bin_dir)
from . import data
