from __future__ import annotations

import argparse
import math
import re
from typing import Any

from ftag.cli_utils import HelpFormatter, valid_path
from ftag.hdf5 import H5Reader

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.utils.logger import setup_logger


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
        "--config_path",
        required=True,
        type=valid_path,
        help="Path to config file",
    )
    parser.add_argument(
        "--deviation-factor",
        default=10.0,
        type=float,
        help="""Maximum deviation/spread factor that is allowed in a group without
        triggering an error. By default 10""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the final numbers to the terminal",
    )

    args = parser.parse_args(args)
    return args


def check_within_factor(
    groups: dict[str, dict[str, int]],
    factor: float = 10.0,
) -> None:
    """Check non-zeros/spread/geometric mean of the samples per group.

    Parameters
    ----------
    groups : dict[str, dict[str, int]]
        Dict with the groups and the sample
    factor : float, optional
        Factor that is the maximum allowed deviation, by default 10.0

    Raises
    ------
    ValueError
        If n_jets is zero for a sample
        If the spread factor is too high
        If the geometric mean deviation is too high
    """
    for name, inner in groups.items():
        if not inner:
            continue

        # Check that each sample is not zero
        for inner_key, inner_value in inner.items():
            if inner_value == 0:
                raise ValueError(f"Found zero jets in group {name} / sample {inner_key}!")

        inner_values = {k: v for k, v in inner.items() if v > 0}

        # Quick necessary & sufficient spread check
        min_k = min(inner_values, key=lambda k: inner_values[k])
        max_k = max(inner_values, key=lambda k: inner_values[k])
        spread = inner_values[max_k] / inner_values[min_k]

        if spread > factor:
            raise ValueError(
                f"Found a spread of a factor of {spread} in group {name}! Check the samples!"
            )

        # Per-key check vs geometric mean (nice symmetric criterion)
        geometric_mean = math.exp(
            sum(math.log(v) for v in inner_values.values()) / len(inner_values)
        )

        for key, value in inner_values.items():
            deviation = max(value / geometric_mean, geometric_mean / value)
            if deviation > factor:
                raise ValueError(
                    f"Sample {key} deviates from the geometric mean by a factor of {deviation}"
                )


def run_input_sample_check(
    config: PreprocessingConfig,
    deviation_factor: float,
    verbose: bool = True,
) -> None:
    """Run the input sample checks on the number of the jets.

    Parameters
    ----------
    config : PreprocessingConfig
        Loaded preprocessing config object
    deviation_factor : float
        Maximum deviation factor that is allowed without an error
    verbose : bool, optional
        Decide, if the results are also printed to the terminal, by default True
    """
    # Setup the logger
    log = setup_logger()

    # Define the r-tags into campaigns translation
    rtag_to_campaign_dict = {
        "r13167": "MC20a",
        "r14859": "MC20a",
        "r13144": "MC20d",
        "r14860": "MC20d",
        "r13145": "MC20e",
        "r14861": "MC20e",
        "r14622": "MC23a",
        "r14932": "MC23a",
        "r15540": "MC23a",
        "r15224": "MC23d",
        "r15530": "MC23d",
        "r16083": "MC23e",
        "r14799": "MC23c - REMOVE",
    }

    # Print starting info
    log.info("[bold green]Checking input samples sizes...")

    # Init a dict for the different sample types
    sample_type_dict: dict[str, dict[str, Any]] = {}

    # Get the different
    for config_blocks in config.config:
        if (
            isinstance(config.config[config_blocks], dict)
            and "pattern" in config.config[config_blocks]
        ):
            sample_type_dict[config_blocks] = {}
            sample_type_dict[config_blocks]["pattern"] = config.config[config_blocks]["pattern"]

    # Setup H5Reader for the different samples to read the total number of jets
    for sample_type, sample_list in sample_type_dict.items():
        # Log the status
        if verbose:
            log.info(f"Checking sample {sample_type}...")

        # Extract the patterns and ensure it's a list or a string
        patterns = sample_list.get("pattern", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        elif isinstance(patterns, list):
            pass
        else:
            log.error(f"Unsupported type for 'pattern' in {sample_type}: {type(patterns)}")
            sample_type_dict[sample_type] = {}
            continue

        # Loop over the different samples in the category
        for sample in patterns:
            # Log the status
            if verbose:
                log.info(f"{sample}")

            # Extract the DSID and r-tag from the sample name
            dsid = (d := re.search(r"(?<=\.)\d{6}(?=\.)", sample)) and d.group(0)
            rtag = (p := re.search(r"_(r\d+)", sample)) and p.group(1)

            # Log an error that the dsid couldn't be extracted
            if dsid is None:
                log.error(f"Can't extract DSID from pattern: {sample}")

            # Log an error that the r-tag couldn't be extracted
            if rtag is None:
                log.error(f"Can't extract r-tag from pattern: {sample}")

            # Build the name for the dict
            if dsid and rtag:
                entry_name = f"{dsid} / {rtag_to_campaign_dict.get(rtag)}"

            elif rtag:
                entry_name = f"{sample} / {rtag_to_campaign_dict.get(rtag)}"

            elif dsid:
                entry_name = f"{dsid} / {sample}"

            else:
                entry_name = sample

            # Create the H5 reader for each sample and read the number of jets from it
            sample_list[entry_name] = H5Reader(
                fname=config.ntuple_dir / sample,
                batch_size=config.batch_size,
                jets_name=config.jets_name,
            ).num_jets

        # Drop the pattern
        del sample_list["pattern"]

    # Check that all the samples per group are not too different
    check_within_factor(groups=sample_type_dict, factor=deviation_factor)

    # Printing the dict with all the number of jets to the terminal, if wanted
    if verbose:
        log.info("Available jets in given groups:\n")

        for sample_type, sample_dict in sample_type_dict.items():
            log.info(f"Group: {sample_type}")

            for entry_name, n_jets in sample_dict.items():
                log.info(f"  - Sample: {entry_name}, N_Jets: {n_jets:,}")


def main(args: Any | None = None) -> None:
    args = parse_args(args)

    # Load preprocessing config
    config = PreprocessingConfig.from_file(
        config_path=args.config_path,
        split="train",
        skip_config_copy=True,
    )

    run_input_sample_check(
        config=config,
        deviation_factor=args.deviation_factor,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
