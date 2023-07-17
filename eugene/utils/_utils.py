import os
from os import PathLike
import logging


def make_dirs(
    output_dir: PathLike,
    overwrite: bool = False,
):
    """Make a directory if it doesn't exist.

    Parameters
    ----------
    output_dir : PathLike
        The path to the directory to create.
    overwrite : bool, optional
        Whether to overwrite the directory if it already exists, by default False.
    """
    if os.path.exists(output_dir):
        if overwrite:
            logging.info("Overwriting existing directory: {}".format(output_dir))
            os.system("rm -rf {}".format(output_dir))
        else:
            print("Output directory already exists: {}".format(output_dir))
            return
    os.makedirs(output_dir)
