from __future__ import annotations

import logging as log
from pathlib import Path

from ftag import Flavours
from ftag.hdf5 import H5Reader
from puma import Histogram, HistogramPlot

from upp.utils import path_append


def load_jets(paths, variable, flavour_label="flavour_label", jets_name="muons"):
    variables = {jets_name: [flavour_label, variable]}
    reader = H5Reader(paths, batch_size=1000, jets_name=jets_name)
    df = reader.load(variables, num_jets=10000)[jets_name]
    return df


def make_hist(stage, flavours, variable, in_paths, jets_name="muons", bins_range=None, suffix=""):
    df = load_jets(in_paths, variable, jets_name=jets_name)

    plot = HistogramPlot(
        ylabel="Normalised Number of jets",
        atlas_second_tag="$\\sqrt{s}=13$ TeV",
        xlabel=variable,
        bins=50,
        y_scale=1.5,
        logy=True,
        norm=True,
        bins_range=bins_range,
    )

    for label_value, label_string in enumerate([f.name for f in flavours]):
        puma_flavour = f"{label_string}jets" if len(label_string) == 1 else label_string
        if puma_flavour == "qcd":
            puma_flavour = "dijets"
        plot.add(
            Histogram(
                df[df["flavour_label"] == label_value][variable],
                label=Flavours[label_string].label,
                colour=Flavours[label_string].colour,
            )
        )

    plot.draw()
    out_dir = Path(in_paths[0]).parent.parent / "plots"
    out_dir.mkdir(exist_ok=True)
    fname = f"{stage}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.png"
    plot.savefig(out_path)
    log.info(f"Saved plot {out_path}")


def make_hist_initial(
    stage,
    flavours,
    variable,
    in_paths_list,
    jets_name="muons",
    bins_range=None,
    suffix="",
    jets_to_plot=-1,
    out_dir=None,
    suffixes=None,
):
    # df = load_jets(in_paths, variable)

    plot = HistogramPlot(
        ylabel="Normalised Number of jets",
        atlas_second_tag="$\\sqrt{s}=13$ TeV",
        xlabel=variable,
        bins=100,
        y_scale=1.5,
        logy=True,
        norm=True,
        bins_range=bins_range,
    )
    if not isinstance(in_paths_list[0], list):
        in_paths_list = [in_paths_list]
    if suffixes is None:
        suffixes = ["" for _ in in_paths_list]

    linestiles = ["-", "--", "-.", ":"]
    for i, in_paths in enumerate(in_paths_list):
        reader = H5Reader(in_paths, batch_size=10000, jets_name=jets_name)
        for flavour in flavours:
            puma_flavour = f"{flavour.label}jets" if len(flavour.label) == 1 else flavour.label
            if puma_flavour == "qcd":
                puma_flavour = "dijets"
            plot.add(
                Histogram(
                    reader.load(
                        {jets_name: [variable]},
                        num_jets=jets_to_plot,
                        cuts=flavour.cuts,
                    )[
                        jets_name
                    ][variable],
                    label=flavour.label + " " + suffixes[i],
                    colour=flavour.colour,
                    linestyle=linestiles[i],
                )
            )

    plot.draw()
    if out_dir is None:
        out_dir = Path(in_paths_list[0][0]).parent.parent / "plots"
    out_dir.mkdir(exist_ok=True)
    fname = f"{stage}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.png"
    plot.savefig(out_path)
    log.info(f"Saved plot {out_path}")


def plot_initial(config):
    paths = [list(sample.path) for sample in config.components.samples]
    suffixes = [sample.name for sample in config.components.samples]
    for var in config.sampl_cfg.vars:
        make_hist_initial(
            "initial",
            config.components.flavours,
            var,
            paths,
            "muons",
            jets_to_plot=100000,
            out_dir=config.out_dir / "plots",
            suffixes=suffixes,
        )
        if "pt" in var:
            make_hist_initial(
                "initial",
                config.components.flavours,
                var,
                paths,
                "muons",
                (0, 500e3),
                "low",
                jets_to_plot=100000,
                out_dir=config.out_dir / "plots",
                suffixes=suffixes,
            )


def main(config, stage):
    if stage != "test" or config.merge_test_samples:
        paths = [config.out_fname]
    else:
        paths = [path_append(config.out_fname, sample) for sample in config.components.samples]

    for var in config.sampl_cfg.vars:
        make_hist(stage, config.components.flavours, var, paths, "muons")
        if "pt" in var:
            make_hist(stage, config.components.flavours, var, paths, "muons", (0, 500e3), "low")
