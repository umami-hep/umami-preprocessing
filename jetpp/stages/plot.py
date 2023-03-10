import logging as log
from pathlib import Path

import h5py
import pandas as pd
from puma import Histogram, HistogramPlot
from puma.utils import global_config

from jetpp.utils import path_append

flav_cfg = global_config["flavour_categories"]


def load_jets(paths, variable):
    variables = ["flavour_label", variable]
    df = pd.DataFrame(columns=variables)
    for path in paths:
        with h5py.File(path) as f:
            df = pd.concat([df, pd.DataFrame(f["jets"].fields(variables)[: int(1e6)])])
    return df


def make_hist(stage, flavours, variable, in_paths, x_range=None):
    df = load_jets(in_paths, variable)

    if x_range:
        x_min, x_max = x_range
    else:
        x_min, x_max = None, None

    plot = HistogramPlot(
        ylabel="Number of jets",
        atlas_second_tag="$\\sqrt{s}=13$ TeV",
        xlabel=variable,
        bins=100,
        y_scale=1.5,
        logy=True,
        norm=True,
        xmin=x_min,
        xmax=x_max,
    )

    for label_value, label_string in enumerate([f.name for f in flavours]):
        puma_flavour = f"{label_string}jets" if len(label_string) == 1 else label_string
        if puma_flavour == "qcd":
            puma_flavour = "dijets"
        plot.add(
            Histogram(
                df[df["flavour_label"] == label_value][variable],
                label=flav_cfg[puma_flavour]["legend_label"],
                colour=flav_cfg[puma_flavour]["colour"],
            )
        )

    plot.draw()
    out_dir = Path(in_paths[0]).parent.parent / "plots"
    out_dir.mkdir(exist_ok=True)
    fname = f"{stage}_{variable}"
    out_path = out_dir / f"{fname}.png"
    plot.savefig(out_path)
    log.info(f"Saved plot {out_path}")


def main(config, stage):
    if stage != "test" or config.merge_test_samples:
        paths = [config.out_fname]
    else:
        paths = [path_append(config.out_fname, sample) for sample in config.components.samples]

    for var in config.sampl_cfg.vars:
        make_hist(stage, config.components.flavours, var, paths)
