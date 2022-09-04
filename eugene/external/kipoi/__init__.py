import os
import sys
import kipoi
bin_dir = os.path.dirname(sys.executable)
os.environ["PATH"] += os.pathsep + bin_dir

kipoi_model_list = kipoi.list_models()

from ._wrappers import get_model_names, get_model