from __future__ import annotations

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


def bin_jets(array: dict, bins: list) -> np.ndarray:
    """Create the histogram and bins for the given resampling variables.

    Parameters
    ----------
    array : dict
        Dict with the loaded jets and the resampling
        variables.
    bins : list
        Flat list with the bins which are to be used.

    Returns
    -------
    hist : np.ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    out_bins : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.
    """
    hist, _, out_bins = binned_statistic_dd(
        sample=s2u(array),
        values=None,
        statistic="count",
        bins=bins,
        expand_binnumbers=True,
    )
    out_bins -= 1
    return hist, out_bins


@dataclass
class Hist:
    """Histogram data class for the preprocessing."""

    path: Path

    def write_hist(
        self,
        jets: dict,
        resampling_vars: list,
        bins: list,
    ) -> None:
        """
        Write the histogram to file.

        Parameters
        ----------
        jets : dict
            Dict with the loaded jets.
        resampling_vars : list
            List of the resampling variables.
        bins : list
            Flat list with the bins.

        Raises
        ------
        ValueError
            If the given binning and cuts don't match.
        """
        # make parent dir
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # bin jets
        hist = bin_jets(jets[resampling_vars], bins)[0]
        pbin = hist / len(jets)  # probability (rate) of each bin
        if not math.isclose(pbin.sum(), 1, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"{pbin.sum()} != 1, check cuts and binning")

        with h5py.File(self.path, "w") as f:
            f.create_dataset("pbin", data=pbin)
            f.create_dataset("hist", data=hist)
            f.attrs.create("num_jets", len(jets))
            f.attrs.create("resampling_vars", resampling_vars)
            for i, v in enumerate(resampling_vars):
                f.attrs.create(f"bins_{v}", bins[i])

    @functools.cached_property
    def hist(self) -> np.array:
        """Return the histogram completely.

        Returns
        -------
        np.array
            Full histogram
        """
        with h5py.File(self.path) as f:
            return f["hist"][:]

    @functools.cached_property
    def pbin(self) -> np.array:
        """Return the probability rate for each bin.

        Returns
        -------
        np.array
            Probability rates for the different bins.
        """
        # probability (rate) of each bin
        with h5py.File(self.path) as f:
            return f["pbin"][:]


def create_histograms(config) -> None:
    """Create the virtual datasets and pdf files.

    Parameters
    ----------
    config : PreprocessingConfig object
        PreprocessingConfig object of the current preprocessing.
    """
    setup_logger()

    title = " Writing PDFs "
    log.info(f"[bold green]{title:-^100}")

    log.info(f"[bold green]Estimating PDFs using {config.num_jets_estimate_hist:,} jets...")
    sampl_vars = config.sampl_cfg.vars
    for c in config.components:
        log.info(f"Estimating {c} PDF using {config.num_jets_estimate_hist:,} samples...")
        c.setup_reader(config.batch_size, config.jets_name)
        cuts_no_split = c.cuts.ignore(["eventNumber"])

        ###
        # TODO: return the number of jets here and pass to the next function to get started
        ###
        c.check_num_jets(
            config.num_jets_estimate_hist,
            cuts=cuts_no_split,
            silent=False,
            raise_error=False,
        )
        jets = c.get_jets(sampl_vars, config.num_jets_estimate_hist, cuts_no_split)
        c.hist.write_hist(jets, sampl_vars, config.sampl_cfg.flat_bins)

    log.info(f"[bold green]Saved to {config.components[0].hist.path.parent}/")
