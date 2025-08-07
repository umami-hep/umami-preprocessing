from __future__ import annotations

import logging as log
from pathlib import Path
from typing import TYPE_CHECKING

from ftag import Cuts
from ftag.hdf5 import H5Reader
from puma import Histogram, HistogramPlot

from upp.utils.tools import path_append

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.preprocessing_config import PreprocessingConfig


def make_hist(
    stage: str,
    values_dict: dict,
    flavours: list,
    variable: str,
    out_dir: Path,
    jets_name: str = "jets",
    bins_range: tuple | None = None,
    suffix: str = "",
    out_format: str = "png",
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
    values_dict : dict
        Dict with the loaded values.
    flavours : list
        List of the flavours that are to be plotted. The list
        needs to contain the Flavour class instances from the
        different flavours.
    variable : str
        Variable that is to be histogrammed and plotted.
    out_dir : Path
        Output directory to which the plots are written.
    jets_name: str, optional
        Name of the jet dataset / the global objects
        by default "jets"
    bins_range : tuple | None, optional
        bins_range argument from from puma.HistogramPlot,
        by default None
    suffix : str, optional
        A string suffix which is added to the plot
        output name, by default "".
    out_format : str, optional
        Output format of the plot. By default "png"
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
        y_scale=1.5,
        logy=True,
    )

    # Define different linestyles for the different samples
    linestiles = ["-", "--", "-.", ":"]

    for counter, (values_key, values) in enumerate(values_dict.items()):
        # Loop over the flavours
        for label_value, flavour in enumerate(flavours):
            # Define the cuts that are needed to select the flavours
            if stage == "initial":
                cuts = flavour.cuts

            else:
                cuts = Cuts.from_list([f"flavour_label == {label_value}"])

            # Get the histogram object
            histo = Histogram(
                values=(
                    cuts(values).values[variable] / 1e3
                    if "pt" in variable
                    else cuts(values).values[variable]
                ),
                bins=50,
                bins_range=bins_range,
                norm=True,
                label=flavour.label + " " + values_key,
                colour=flavour.colour,
                linestyle=linestiles[counter],
                underoverflow=True,
            )

            # Add to histogram
            plot.add(histogram=histo)

            # Set bin_edges
            if bins_range is None:
                bins_range = (histo.bin_edges[0], histo.bin_edges[-1])

    # Draw plot
    plot.draw()

    # Check that the output dir exists
    out_dir.mkdir(exist_ok=True)

    # Define output name and path and save it
    fname = f"{stage}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.{out_format}"
    plot.savefig(out_path)
    log.info(f"Saved plot to {out_path}")


def plot_resampling_dists(config: PreprocessingConfig, stage: str) -> None:
    """Plot initial resampling dist plots.

    Plot the initial distribtions of the resampling variables
    for the given samples.

    Parameters
    ----------
    config : PreprocessingConfig
        PreprocessingConfig object of the current preprocessing.
    stage : str
        Stage that is to be run.
    """
    log.info("Plotting initial plots for the resampling variables...")
    # Get all the variables that need to be loaded
    vars_to_load = config.sampl_cfg.vars

    # Get the paths/suffixes of the samples
    if stage == "initial":
        paths = [list(sample.path) for sample in config.components.samples]
        suffixes = [sample.name for sample in config.components.samples]
        for iter_flav in config.components.flavours:
            vars_to_load += list(set(iter_flav.cuts.variables))

    elif stage != "test" or config.merge_test_samples:
        paths = [
            [
                (
                    config.out_fname.parent / f"{config.out_fname.stem}*.h5"
                    if config.num_jets_per_output_file
                    else config.out_fname
                )
            ]
        ]
        suffixes = ["" for _ in paths]
        vars_to_load += ["flavour_label"]

    else:
        paths = [path_append(config.out_fname, sample) for sample in config.components.samples]
        suffixes = ["" for _ in paths]
        vars_to_load += ["flavour_label"]

    # Init a values_dict
    values_dict = {}

    # Loop over the different paths
    for counter, in_paths in enumerate(paths):
        values_dict[suffixes[counter]] = H5Reader(
            fname=in_paths,
            batch_size=config.batch_size,
            jets_name=config.jets_name,
            shuffle=False,
            equal_jets=True,
        ).load(
            {config.jets_name: vars_to_load},
            num_jets=config.num_jets_estimate_plotting,
        )[config.jets_name]

    # Loop over the resamling variables
    for var in config.sampl_cfg.vars:
        log.info(f"Plotting {var}")
        make_hist(
            stage=stage,
            values_dict=values_dict,
            jets_name=config.jets_name,
            flavours=config.components.flavours,
            variable=var,
            out_dir=config.out_dir / "plots",
        )

        # For pT, make another plot for the low pT region of ttbar
        if "pt" in var:
            make_hist(
                stage=stage,
                values_dict=values_dict,
                jets_name=config.jets_name,
                flavours=config.components.flavours,
                variable=var,
                bins_range=(20, 400),
                suffix="_low",
                out_dir=config.out_dir / "plots",
            )
