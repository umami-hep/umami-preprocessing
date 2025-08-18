"""
Preprocessing pipeline for jet taggging.

By default all stages for the training split are run.
To run with only specific stages enabled, include the flag for the required stages.
To run without certain stages, include the corresponding negative flag.

Note that all stages are required to run the pipeline. If you want to disable resampling,
you need to set method: none in your config file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from ftag.cli_utils import HelpFormatter, valid_path

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.hist import create_histograms
from upp.stages.merging import Merging
from upp.stages.normalisation import Normalisation
from upp.stages.plot import plot_resampling_dists
from upp.stages.resampling import Resampling
from upp.stages.reweight import Reweight
from upp.stages.rw_merge import RWMerge
from upp.stages.split_containers import SplitContainers
from upp.utils.check_input_samples import run_input_sample_check
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
        "--config",
        required=True,
        type=valid_path,
        help="Path to config file",
    )
    parser.add_argument(
        "--prep",
        action="store_true",
        default=None,
        help="Estimate and write PDFs",
    )
    parser.add_argument(
        "--no-prep",
        dest="prep",
        action="store_false",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        default=None,
        help="Run resampling",
    )
    parser.add_argument(
        "--no-resample",
        dest="resample",
        action="store_false",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=None,
        help="Run merging",
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        default=None,
        help="Compute normalisations",
    )
    parser.add_argument(
        "--no-norm",
        dest="norm",
        action="store_false",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=None,
        help="Plot output distributions",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which file to produce",
    )
    parser.add_argument(
        "--component",
        default=None,
        help="Component which is processed during --prep",
    )
    parser.add_argument(
        "--split-components",
        action="store_true",
        default=False,
        help="Split containers into components",
    )
    parser.add_argument(
        "--reweight", "--rw", action="store_true", default=False, help="Run the reweighting stage"
    )
    parser.add_argument(
        "--rw-merge", "--rwm", action="store_true", default=False, help="Run the reweighting stage"
    )
    parser.add_argument(
        "--rw-merge-idx",
        "--rwm-idx",
        type=str,
        default=None,
        help=(
            "Commar seperated pair of indices representing the range of output "
            "files to create, e.g '0,10' will create files 0 to 9"
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Region which is processed during --resample",
    )
    parser.add_argument(
        "--skip-sample-check",
        action="store_true",
        help="Skip the inital input sample check",
    )
    parser.add_argument(
        "--grid", action="store_true", help="Use when running the split stage on the grid. "
    )
    parser.add_argument(
        "--container",
        default=None,
        type=str,
        help="Container to use during the 'split-containers' stage. "
        "If not specified, all containers in the config will be used.",
    )
    parser.add_argument(
        "--files",
        default=None,
        help="comma-separated list of files to use during the 'split-containers' stage ",
    )

    args = parser.parse_args(args)
    d = vars(args)
    ignore = [
        "config",
        "split",
        "component",
        "region",
        "container",
        "files",
        "grid",
        "split_components",
        "reweight",
        "rw_merge",
        "rw_merge_idx",
    ]
    if not any(v for a, v in d.items() if a not in ignore):
        for v in d:
            if v not in ignore and d[v] is None:
                d[v] = True
    return args


def run_pp(args: argparse.Namespace) -> None:
    """Run the preprocessing.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    """
    log = setup_logger()

    # print start info
    log.info("[bold green]Starting preprocessing...")
    start = datetime.now()
    log.info(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    # load config
    config = PreprocessingConfig.from_file(args.config, args.split, skip_checks=args.grid)

    if args.split_components:
        log.info("Splitting containers...")
        split = SplitContainers(args.config)
        split.run(
            args.container,
            args.files,
            "." if args.grid else None,  # Use current directory if not on grid
        )
        # If we aren't running on the grid, we create the metadata after splitting
        if not args.grid:
            split.create_meta_data()
    if args.reweight:
        log.info("Running reweighting...")
        reweight = Reweight(config)
        reweight.run()
    # create virtual datasets and pdf files
    if args.prep:
        # Check the input samples sizes
        if not args.skip_sample_check:
            run_input_sample_check(
                config=config,
                deviation_factor=10.0,
                verbose=True,
            )

        if args.split == "train":
            create_histograms(
                config=config,
                component_to_run=args.component,
            )

    # run the resampling
    if args.resample:
        resampling = Resampling(config)
        resampling.run(region=args.region, component=args.component)

    # run the merging
    if args.merge:
        merging = Merging(config)
        merging.run()
    if args.rw_merge:
        if args.rw_merge_idx:
            rw_merge_idx = args.rw_merge_idx
            assert "," in rw_merge_idx, "rw-merge-idx must be a comma-separated pair of indices"
            rw_merge_idx = tuple(map(int, rw_merge_idx.split(",")))
            assert len(rw_merge_idx) == 2, "rw-merge-idx must be a pair of indices"
        else:
            rw_merge_idx = None
        rw_merge = RWMerge(config, rw_merge_idx)
        rw_merge.run()

    # run the normalisation
    if args.norm and args.split == "train":
        norm = Normalisation(config)
        norm.run()

    # make plots
    if args.plot:
        title = " Plotting "
        log.info(f"[bold green]{title:-^100}")
        plot_resampling_dists(config=config, stage="initial")
        plot_resampling_dists(config=config, stage=args.split)

    # print end info
    end = datetime.now()
    title = " Finished Preprocessing! "
    log.info(f"[bold green]{title:-^100}")
    log.info(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Elapsed time: {str(end - start).split('.')[0]}")


def main(args: Any | None = None) -> None:
    args = parse_args(args)
    log = setup_logger()

    if args.split == "all":
        d = vars(args)
        for split in ["train", "val", "test"]:
            d["split"] = split
            log.info(f"[bold blue]{'-' * 100}")
            title = f" {args.split} "
            log.info(f"[bold blue]{title:-^100}")
            log.info(f"[bold blue]{'-' * 100}")
            run_pp(args)
    else:
        run_pp(args)


if __name__ == "__main__":
    main()
