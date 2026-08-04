"""List the components defined in a preprocessing config."""

from __future__ import annotations

import argparse
from typing import Any

from ftag.cli_utils import HelpFormatter, valid_path

from upp.classes.preprocessing_config import PreprocessingConfig


def parse_args(args: Any) -> argparse.Namespace:
    """Parse the command line arguments.

    Parameters
    ----------
    args : Any
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace with the parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=valid_path,
        help="Path to config file",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Split to load the config for (component names are split-independent)",
    )
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Only print the unique region names",
    )

    return parser.parse_args(args)


def main(args: Any | None = None) -> None:
    """List components as tab-separated `region sample flavour name` rows.

    Parameters
    ----------
    args : Any | None, optional
        Command line arguments, by default None
    """
    args = parse_args(args)

    config = PreprocessingConfig.from_file(
        config_path=args.config,
        split=args.split,
        skip_checks=True,
        skip_config_copy=True,
    )

    if args.regions:
        for region in config.components.regions:
            print(region.name)
        return

    for component in config.components:
        print(
            component.region.name,
            component.sample.name,
            component.flavour.name,
            component.name,
            sep="\t",
        )


if __name__ == "__main__":
    main()
