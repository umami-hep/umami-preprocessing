from __future__ import annotations

import logging as log
from pathlib import Path
from typing import TYPE_CHECKING

from ftag import Cuts
from ftag.hdf5 import H5Reader
from puma import Histogram, HistogramPlot

from upp.utils import path_append

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.preprocessing_config import PreprocessingConfig


def make_hist(
    stage: str,
    flavours: list,
    variable: str,
    in_paths_list: str | list,
    jets_name: str = "jets",
    bins_range: tuple | None = None,
    suffix: str = "",
    num_jets: int | None = -1,
    out_dir: Path | None = None,
    suffixes: list | None = None,
    out_format: str = "png",
    batch_size: int = 10000,
) -> None:
    """Make distribution plots of the reweighting variables.

    Plot the distribution of the given variable
    for multiple different samples (like ttbar, zpext, etc.)
    in one plot.

    Parameters
    ----------
    stage : str
        The stage in which the preprocessing is currently in.
        Mainly used for the ouput name string.
    flavours : list
        List of the flavours that are to be plotted. The list
        needs to contain the Flavour class instances from the
        different flavours.
    variable : str
        Variable that is to be histogrammed and plotted.
    in_paths_list : str | list
        String or list of strings with the paths to the files
        from which the jets are loaded.
    jets_name: str, optional
        Name of the jet dataset / the global objects
        by default "jets"
    bins_range : tuple, optional
        bins_range argument from from puma.HistogramPlot,
        by default None
    suffix : str, optional
        A string suffix which is added to the plot
        output name, by default "".
    num_jets : int, optional
        Number of jets that are to be plotted per flavour,
        by default -1 (all).
    out_dir : Path object, optional
        Special output directoy, by default None
    suffixes : list, optional
        Suffixes to mark the different samples, by default None
    out_format : str, optional
        Format of the output plot, by default "png".
    batch_size : int, optional
        Number of jets used per batch, by default 10000.
    """
    # Get the correct name of the xlabel
    if "pt" in variable:
        xlabel = "Jet $p_\\mathrm{T}$ [GeV]"

    elif "eta" in variable:
        xlabel = "Jet $|\\eta|$"

    else:
        xlabel = variable

    # Setup the histogram
    plot = HistogramPlot(
        ylabel=f"Normalised Number of {jets_name}",
        xlabel=xlabel,
        bins=50,
        y_scale=1.5,
        logy=True,
        norm=True,
        bins_range=bins_range,
        underoverflow=False,
    )

    # Check that the given paths argument is a list
    if not isinstance(in_paths_list[0], list):
        in_paths_list = [in_paths_list]

    # Init dummy suffixes if not provided
    if suffixes is None:
        suffixes = ["" for _ in in_paths_list]

    # Define different linestyles for the different samples
    linestiles = ["-", "--", "-.", ":"]

    # Loop over the different samples
    for i, in_paths in enumerate(in_paths_list):
        # Load jets from the file
        reader = H5Reader(fname=in_paths, batch_size=batch_size, jets_name=jets_name)

        # Loop over the flavours
        for label_value, flavour in enumerate(flavours):
            (f"{flavour.label}jets" if len(flavour.label) == 1 else flavour.label)

            if stage == "initial":
                cuts = flavour.cuts

            else:
                cuts = Cuts.from_list([f"flavour_label == {label_value}"])

            # Get the histo values
            values = reader.load(
                {jets_name: [variable]},
                num_jets=num_jets,
                cuts=cuts,
            )[jets_name][variable]

            # Check for pT values
            if "pt" in variable:
                values /= 1000

            # Add to histogram
            plot.add(
                Histogram(
                    values=values,
                    label=flavour.label + " " + suffixes[i],
                    colour=flavour.colour,
                    linestyle=linestiles[i],
                )
            )

    # Draw plot
    plot.draw()

    # Check if an extra output dir is specified
    if out_dir is None:
        out_dir = Path(in_paths_list[0][0]).parent.parent / "plots"

    # Check that the output dir exists
    out_dir.mkdir(exist_ok=True)

    # Define output name and path and save it
    fname = f"{stage}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.{out_format}"
    plot.savefig(out_path)
    log.info(f"Saved plot {out_path}")


def plot_resampling_dists(config: PreprocessingConfig, stage: str) -> None:
    """Plot initial resampling dist plots.

    Plot the initial distribtions of the resampling variables
    for the given samples.

    Parameters
    ----------
    config : PreprocessingConfig
        PreprocessingConfig object of the current preprocessing.
    """
    # Get the paths/suffixes of the samples
    if stage == "initial":
        paths = [list(sample.path) for sample in config.components.samples]
        suffixes = [sample.name for sample in config.components.samples]

    elif stage != "test" or config.merge_test_samples:
        paths = [[config.out_fname]]
        suffixes = None

    else:
        paths = [path_append(config.out_fname, sample) for sample in config.components.samples]
        suffixes = None

    # Loop over the resamling variables
    for var in config.sampl_cfg.vars:
        make_hist(
            stage=stage,
            flavours=config.components.flavours,
            variable=var,
            in_paths_list=paths,
            jets_name=config.jets_name,
            num_jets=config.num_jets_estimate_plotting,
            out_dir=config.out_dir / "plots",
            suffixes=suffixes,
            batch_size=config.batch_size,
        )
        if "pt" in var:
            make_hist(
                stage=stage,
                flavours=config.components.flavours,
                variable=var,
                in_paths_list=paths,
                jets_name=config.jets_name,
                bins_range=(20, 400),
                suffix="_low",
                num_jets=config.num_jets_estimate_plotting,
                out_dir=config.out_dir / "plots",
                suffixes=suffixes,
                batch_size=config.batch_size,
            )
