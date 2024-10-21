"""Single run of prep-dataset, given input arguments."""

import argparse
import logging
import os
import sys
import yaml
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import matplotlib
import pandas as pd
import psutil

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # This needs to be after matplotlib.use('Agg')
import seaborn as sns

from eugene.prep_dataset import consts

logger = logging.getLogger("eugene")


def run_prep_dataset(args: argparse.Namespace):
    """The full script for the command line tool prep-dataset.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from command line.

    """
    try:
        # Log the start time.
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Running prep-dataset")
        
        # Load parameters
        logger.info("Loading parameters")
        with open(args.params_file, "r") as f:
            params = yaml.safe_load(f)
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
            
        # Generate main SeqData object
        logger.info("Loading data")

        # Calculate sequence distributions
        logger.info("Calculating sequence distributions")

        # Run baseline motif analysis
        logger.info("Running baseline motif analysis")

        # Split into train and test sets
        logger.info("Splitting data into train and test sets")

        # Log the end time
        logger.info("Completed prep-dataset")
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))

    # The exception allows user to end inference prematurely with CTRL-C.
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Terminated without saving\n")
