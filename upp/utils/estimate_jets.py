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
    parser.add_argument(
        "--output",
        default="max_num_jets.log",
        help="Path to output log file",
    )
    parser.add_argument(
        "--no-prep",
        action="store_true",
        help="Do not estimate and write PDFs/create histograms",
    )
    return parser.parse_args(args)


def main(args: Any | None = None) -> None:
    parsed_args = parse_args(args)
    run_estimate_jets(
        parsed_args.config,
        output_path=parsed_args.output,
        prep=not parsed_args.no_prep,
    )


if __name__ == "__main__":
    main()
