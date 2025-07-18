# This stage aims to run the whole pre-processing in only 2 passes, rather than the current 4 or so
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.logger import setup_logger
from upp.stages.hist import create_histograms
from upp.stages.merging import Merging
from upp.stages.normalisation import Normalisation
from upp.stages.plot import plot_initial_resampling_dists, plot_resampled_dists
from upp.stages.resampling import Resampling

class HelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    ...

def parse_args():
    abool = "store_true"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to config file")
    parser.add_argument("--prep", action=abool, default=None, help="Estimate and write PDFs")
    parser.add_argument("--no-prep", dest="prep", action="store_false")
    parser.add_argument("--resample", action=abool, default=None, help="Run resampling")
    parser.add_argument("--no-resample", dest="resample", action="store_false")
    parser.add_argument("--merge", action=abool, default=None, help="Run merging")
    parser.add_argument("--no-merge", dest="merge", action="store_false")
    parser.add_argument("--norm", action=abool, default=None, help="Compute normalisations")
    parser.add_argument("--no-norm", dest="norm", action="store_false")
    parser.add_argument("--plot", action=abool, default=None, help="Plot resampled distributions")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    splits = ["train", "val", "test", "all"]
    parser.add_argument("--split", default="train", choices=splits, help="Which file to produce")

    args = parser.parse_args()
    d = vars(args)
    ignore = ["config", "split"]
    if not any(v for a, v in d.items() if a not in ignore):
        for v in d:
            if v not in ignore and d[v] is None:
                d[v] = True
    return args

class SingleSampleReader:

    def __init__(self, ):

        pass


    # def 

def run_pp(args) -> None:
    log = setup_logger()

    # print start info
    log.info("[bold green]Starting preprocessing...")
    start = datetime.now()
    log.info(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # load config
    config = PreprocessingConfig.from_file(Path(args.config), args.split)

    # run 2-stage preprocessing - much faster this way
    # step 1:
    #   - Create h5 reader for each [sample]
    #          - Create a helper class which wraps a component and a reader, applying all its cuts
    #          - Create a helper class which wraps a component and a writer, applying all its cuts
    #           - this returns just the jets after all the cuts
    #  Step 1.5:
    #  - We setup an h5 writer for each train/test/val
    #       - The train and val file writer we add an additional output for each re-weight, initialised to nan
    #       - We now have some N arrays, which we merge, and then split in train/test/val
    #       - We setup the initial histograms for each re-weight
    #       - We write val and test, then we calcualte the weights before writing the train and add them to the histograms
    #       - Once we hav exhausted all the readers, we calculate the weights and then write it to a histogram file
    #  Step 2: (lol)
    #       - We iterate the train file, and modify it in place, inserting the weights

    #  
    # # create virtual datasets and pdf files
    # if args.prep and args.split == "train":
    #     create_histograms(config)

    # # run the resampling
    # if args.resample:
    #     resampling = Resampling(config)
    #     resampling.run()

    # # run the merging
    # if args.merge:
    #     merging = Merging(config)
    #     merging.run()

    # # run the normalisation
    # if args.norm and args.split == "train":
    #     norm = Normalisation(config)
    #     norm.run()

    # make plots
    if args.plot:
        plot_initial_resampling_dists(config=config)
        plot_resampled_dists(config=config, stage=args.split)

    # print end info
    end = datetime.now()
    title = " Finished Preprocessing! "
    log.info(f"[bold green]{title:-^100}")
    log.info(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Elapsed time: {str(end - start).split('.')[0]}")


def main() -> None:
    args = parse_args()
    log = setup_logger()

    if args.split == "all":
        d = vars(args)
        for split in ["train", "val", "test"]:
            d["split"] = split
            log.info(f"[bold blue]{'-'*100}")
            title = f" {args.split} "
            log.info(f"[bold blue]{title:-^100}")
            log.info(f"[bold blue]{'-'*100}")
            run_pp(args)
    else:
        run_pp(args)


if __name__ == "__main__":
    main()