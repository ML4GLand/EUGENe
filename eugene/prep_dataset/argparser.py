import argparse

from eugene.prep_dataset import consts

# Define the parent parser with the common arguments
def get_parent_parser() -> argparse.ArgumentParser:
    """Create a parent parser with common arguments."""
    parent_parser = argparse.ArgumentParser(add_help=False)  # Prevent auto-help addition here

    parent_parser.add_argument(
        "--params",
        nargs=None,
        type=str,
        dest="params_file",
        required=False,
        default=None,
        help="Path to a params file. "
        "This is a YAML file that contains parameters for this subcommand. "
        "If not provided, default parameters will be used.",
    )
    parent_parser.add_argument(
        "--path_out",
        nargs=None,
        type=str,
        dest="path_out",
        required=True,
        help="Output directory location. " 
        "If it does not exist, it will be created.",
    )
    parent_parser.add_argument(
        "--random-state",
        nargs=None,
        type=int,
        dest="random_state",
        required=False,
        default=consts.RANDOM_STATE,
        help="Random state to use for random number generators.",
    )
    parent_parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        dest="n_threads",
        help="Number of threads to use when pytorch is run "
        "on CPU. Defaults to the number of logical cores.",
    )
    parent_parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Including the flag --debug will log "
        "extra messages useful for debugging.",
    )

    return parent_parser


def add_subparser_args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add tool-specific arguments for prep-dataset.

    Args:
        subparsers: Parser object before addition of arguments specific to prep-dataset.

    Returns:
        parser: Parser object with additional parameters.
    """

    # Create a subparser for the main "prep-dataset" command
    subparser = subparsers.add_parser(
        "prep-dataset",
        description="Generates training, validation and testing datasets.",
        help="Subcommand to prepare datasets for training. Will also generate a report of the input data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers under "prep-dataset"
    subparser_commands = subparser.add_subparsers(
        title="sub-commands", 
        description="Valid prep-dataset commands are 'tabular', 'regions', and 'binarized_regions'", 
        dest="command",
        required=True,
    )

    # Inherit arguments from the parent parser
    parent_parser = get_parent_parser()

    # Add subcommand-specific parsers that inherit from the parent parser
    subparser_commands.add_parser(
        "tabular",
        description="Prepares a dataset from input tabular files.",
        help="Prepares a dataset from input tabular files. These files should contain sequences and labels.",
        parents=[parent_parser],  # Inherit from the parent parser
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # You can add more subcommands here in a similar way, e.g., "regions", "binarized_regions"
    # subparser_commands.add_parser(...)

    return subparsers
