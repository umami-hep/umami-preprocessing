from __future__ import annotations

import argparse
from typing import Any

from ftag.cli_utils import HelpFormatter, valid_path

from upp.stages.estimate_jets import run_estimate_jets


def parse_args(args: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the maximum number of jets available per component "
                    "and scale regions to match flavour ratios.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=valid_path,
        help="Path to config file",
    )
    return parser.parse_args(args)


def main(args: Any | None = None) -> None:
    parsed_args = parse_args(args)
    run_estimate_jets(parsed_args.config)


if __name__ == "__main__":
    main()
