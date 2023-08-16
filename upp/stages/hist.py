import functools
import logging as log
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from scipy.stats import binned_statistic_dd

from upp.logger import setup_logger


def bin_jets(array, bins) -> np.array:
    hist, _, bins = binned_statistic_dd(s2u(array), None, "count", bins, expand_binnumbers=True)
    bins -= 1
    return hist, bins


@dataclass
class Hist:
    path: Path

    def write_hist(self, jets, resampling_vars, bins) -> None:
        # make parent dir
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # bin jets
        hist = bin_jets(jets[resampling_vars], bins)[0]
        pdf = hist / len(jets)
        if not math.isclose(pdf.sum(), 1, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"{pdf.sum()} != 1, check cuts and binning")

        with h5py.File(self.path, "w") as f:
            f.create_dataset("pdf", data=pdf)
            f.create_dataset("hist", data=hist)
            f.attrs.create("num_jets", len(jets))
            f.attrs.create("resampling_vars", resampling_vars)
            for i, v in enumerate(resampling_vars):
                f.attrs.create(f"bins_{v}", bins[i])

    @functools.cached_property
    def hist(self) -> np.array:
        with h5py.File(self.path) as f:
            return f["hist"][:]

    @functools.cached_property
    def pdf(self) -> np.array:
        with h5py.File(self.path) as f:
            return f["pdf"][:]


"""
    @functools.cached_property
    def upscaled_pdf(self):
        upscale = 1
        # get bins
        xs = []
        with h5py.File(self.path) as f:
            attrs = dict(f.attrs)

        for var in attrs["resampling_vars"]:
            var_bins = attrs[f"bins_{var}"]
            n_bins = len(var_bins) - 1
            points = np.linspace(0, n_bins - 1 , n_bins * upscale)
            xs.append(points)

        # return the smoothed pdf
        xy = np.meshgrid(*xs, indexing="ij")
        smoothed = ndimage.map_coordinates(self.pdf, xy, order=1)
        return smoothed / smoothed.sum()

def smooth_weights(weights, path):
    upscale = 1
    print(upscale)
    # get bins
    xs = []
    with h5py.File(path) as f:
        attrs = dict(f.attrs)

    for var in attrs["resampling_vars"]:
        var_bins = attrs[f"bins_{var}"]
        n_bins = len(var_bins) - 1
        points = np.linspace(0, n_bins - 1 , n_bins * upscale)
        xs.append(points)

    # return the smoothed pdf
    xy = np.meshgrid(*xs, indexing="ij")
    smoothed = ndimage.map_coordinates(weights, xy, order=1)
    return smoothed# / smoothed.sum()
"""


def main(config=None):
    setup_logger()

    title = " Writing PDFs "
    log.info(f"[bold green]{title:-^100}")

    log.info(f"[bold green]Estimating PDFs using {config.num_jets_estimate:,} jets...")
    sampl_vars = config.sampl_cfg.vars
    for c in config.components:
        log.info(f"Estimating PDF for {c}")
        c.setup_reader(config.batch_size)
        cuts_no_split = c.cuts.ignore(["eventNumber"])
        c.check_num_jets(
            config.num_jets_estimate, cuts=cuts_no_split, silent=False, raise_error=False
        )
        jets = c.get_jets(sampl_vars, config.num_jets_estimate, cuts_no_split)
        c.hist.write_hist(jets, sampl_vars, config.sampl_cfg.flat_bins)

    log.info(f"[bold green]Saved to {config.components[0].hist.path.parent}/")
