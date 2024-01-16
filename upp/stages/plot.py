from __future__ import annotations

import logging as log
from pathlib import Path

from ftag import Flavours
from ftag.flavour import FlavourContainer
from ftag.hdf5 import H5Reader
from puma import Histogram, HistogramPlot

from upp.utils import path_append


def load_jets(
    paths: str | list,
    variable: str,
    flavour_label="flavour_label",
    jets_name="jets",
) -> dict:
    """
    Load the variables and labels from the jets in a given file(s).

    Parameters
    ----------
    paths : str | list
        Path to the file which is to be loaded. Can be either a string
        or a list. Wildcards are also supported.
    variable : str
        Variable which is to be loaded together with the labels.
    flavour_label : str, optional
        Name of the flavour label variable which is used for the labels,
        by default "flavour_label"
    jets_name: str, optional
        Name of the jet dataset / the global objects
        by default "jets"

    Returns
    -------
    dict
        Dict with the loaded variable and labels.
    """
    variables = {jets_name: [flavour_label, variable]}
    reader = H5Reader(paths, batch_size=1000, jets_name=jets_name)
    df = reader.load(variables, num_jets=10000)[jets_name]
    return df


def make_hist(
    stage: str,
    flavours: list,
    variable: str,
    in_paths: str | list,
    jets_name: str = "jets",
    bins_range: tuple | None = None,
    suffix: str = "",
    flavour_cont: FlavourContainer = Flavours,
) -> None:
    """
    Create and plot the histogram and save it to disk.

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
    in_paths : str
        Path to the files from which the jets are loaded.
    jets_name: str, optional
        Name of the jet dataset / the global objects
        by default "jets"
    bins_range : tuple, optional
        bins_range argument from from puma.HistogramPlot,
        by default None
    suffix : str, optional
        A string suffix which is added to the plot
        output name, by default "".
    """
    # Load the variable from the jets
    df = load_jets(in_paths, variable, jets_name=jets_name)

    # Setup the histogram
    plot = HistogramPlot(
        ylabel=f"Normalised Number of {jets_name}",
        atlas_second_tag="$\\sqrt{s}=13$ TeV",
        xlabel=variable,
        bins=50,
        y_scale=1.5,
        logy=True,
        norm=True,
        bins_range=bins_range,
    )

    # Loop over the flavours and add them to the histogram
    for label_value, label_string in enumerate([f.name for f in flavours]):
        f"{label_string}jets" if len(label_string) == 1 else label_string

        # Add the flavour with its label and colour to the histogram
        plot.add(
            Histogram(
                df[df["flavour_label"] == label_value][variable],
                label=flavour_cont[label_string].label,
                colour=flavour_cont[label_string].colour,
            )
        )

    # Draw the histogram
    plot.draw()

    # Set outdir and check that it exists
    out_dir = Path(in_paths[0]).parent.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    # Define output name and path and save the plot
    fname = f"{stage}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.png"
    plot.savefig(out_path)
    log.info(f"Saved plot {out_path}")


def make_hist_initial(
    stage: str,
    flavours: list,
    variable: str,
    in_paths_list: str | list,
    jets_name: str = "jets",
    bins_range: tuple | None = None,
    suffix: str = "",
    jets_to_plot: int = -1,
    out_dir: Path | None = None,
    suffixes: list | None = None,
    out_format: str = "png",
) -> None:
    """Make initial distribution plots.

    Plot the initial distribution of the given variable
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
    jets_to_plot : int, optional
        Number of jets that are to be plotted per flavour,
        by default -1 (all).
    out_dir : Path object, optional
        Special output directoy, by default None
    suffixes : list, optional
        Suffixes to mark the different samples, by default None
    out_format : str, optional
        Format of the output plot, by default "png".
    """
    # Setup the histogram
    plot = HistogramPlot(
        ylabel=f"Normalised Number of {jets_name}",
        atlas_second_tag="$\\sqrt{s}=13$ TeV",
        xlabel=variable,
        bins=100,
        y_scale=1.5,
        logy=True,
        norm=True,
        bins_range=bins_range,
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
        reader = H5Reader(in_paths, batch_size=10000, jets_name=jets_name)

        # Loop over the flavours
        for flavour in flavours:
            (f"{flavour.label}jets" if len(flavour.label) == 1 else flavour.label)

            # Add to histogram
            plot.add(
                Histogram(
                    reader.load(
                        {jets_name: [variable]},
                        num_jets=jets_to_plot,
                        cuts=flavour.cuts,
                    )[jets_name][variable],
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


def plot_initial_resampling_dists(config) -> None:
    """Plot initial resampling dist plots.

    Plot the initial distribtions of the resampling variables
    for the given samples.

    Parameters
    ----------
    config : PreprocessingConfig object
        PreprocessingConfig object of the current preprocessing.
    """
    # Get the paths of the samples
    paths = [list(sample.path) for sample in config.components.samples]

    # Get the suffixes of the samples
    suffixes = [sample.name for sample in config.components.samples]

    # Loop over the resamling variables
    for var in config.sampl_cfg.vars:
        make_hist_initial(
            stage="initial",
            flavours=config.components.flavours,
            variable=var,
            in_paths_list=paths,
            jets_name=config.jets_name,
            jets_to_plot=100000,
            out_dir=config.out_dir / "plots",
            suffixes=suffixes,
        )
        if "pt" in var:
            make_hist_initial(
                stage="initial",
                flavours=config.components.flavours,
                variable=var,
                in_paths_list=paths,
                jets_name=config.jets_name,
                bins_range=(0, 500e3),
                suffix="low",
                jets_to_plot=100000,
                out_dir=config.out_dir / "plots",
                suffixes=suffixes,
            )


def plot_resampled_dists(config, stage: str) -> None:
    """Plot resampled variable distributions.

    Plot the histograms of different variables at the
    various stages of the preprocessing.

    Parameters
    ----------
    config : PreprocessingConfig object
        PreprocessingConfig object of the current preprocessing.
    stage : str
        Current stage of the preprocessing for which the plot is
        made.
    """
    # Define special output names for the merge and test stage
    if stage != "test" or config.merge_test_samples:
        paths = [config.out_fname]
    else:
        paths = [path_append(config.out_fname, sample) for sample in config.components.samples]

    # Loop over the variables to plot them
    for var in config.sampl_cfg.vars:
        make_hist(
            stage=stage,
            flavours=config.components.flavours,
            flavour_cont=config.flavour_cont,
            variable=var,
            in_paths=paths,
            jets_name=config.jets_name,
        )
        if "pt" in var:
            make_hist(
                stage=stage,
                flavours=config.components.flavours,
                flavour_cont=config.flavour_cont,
                variable=var,
                in_paths=paths,
                jets_name=config.jets_name,
                bins_range=(0, 500e3),
                suffix="low",
            )
