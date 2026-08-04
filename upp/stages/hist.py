from __future__ import annotations

import functools
import logging as log
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from scipy.stats import binned_statistic_dd

from upp.utils.logger import setup_logger

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.preprocessing_config import PreprocessingConfig


def bin_global_objects(array: dict, bins: list) -> tuple[np.ndarray, np.ndarray]:
    """Create the histogram and bins for the given resampling variables.

    Parameters
    ----------
    array : dict
        Dict with the loaded objects and the resampling
        variables.
    bins : list
        Flat list with the bins which are to be used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Returns a tuple of two np.ndarrays, which are the following:

        hist : np.ndarray, shape(nx1, nx2, nx3,...)
            The values of the selected statistic in each two-dimensional bin.
        out_bins : (N,) array of ints or (D,N) ndarray of ints
            This assigns to each element of `sample` an integer that represents the
            bin in which this observation falls.  The representation depends on the
            `expand_binnumbers` argument.  See `Notes` for details.
    """
    sample = s2u(array).astype(np.float64, copy=False)
    hist, _, out_bins = binned_statistic_dd(
        sample=sample,
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
        global_objects: dict,
        resampling_vars: list,
        bins: list,
        global_name: str = "jets",
    ) -> None:
        """
        Write the histogram to file.

        Parameters
        ----------
        global_objects : dict
            Dict with the loaded objects.
        resampling_vars : list
            List of the resampling variables.
        bins : list
            Flat list with the bins.
        global_name : str, optional
            Name of the global object, used for the count attribute, by default "jets".

        Raises
        ------
        ValueError
            If the given binning and cuts don't match.
        """
        # make parent dir
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # bin objects
        hist = bin_global_objects(global_objects[resampling_vars], bins)[0]
        pbin = hist / len(global_objects)  # probability (rate) of each bin
        if not math.isclose(pbin.sum(), 1, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"{pbin.sum()} != 1, check cuts and binning")

        with h5py.File(self.path, "w") as f:
            f.create_dataset("pbin", data=pbin)
            f.create_dataset("hist", data=hist)
            f.attrs.create(f"num_{global_name}", len(global_objects))
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


def create_histograms(
    config: PreprocessingConfig,
    component_to_run: str | None = None,
) -> None:
    """Create the virtual datasets and pdf files.

    Parameters
    ----------
    config : PreprocessingConfig
        PreprocessingConfig object of the current preprocessing.
    component_to_run : str | None
        Component which should be run. By default (None), all components
        are processed sequentially.

    Raises
    ------
    ValueError
        If no region was processed during histogram creation
    """
    # Setup the logger and load the variables used for resampling
    setup_logger()
    if config.skip_resampling:
        log.info("Resampling is disabled - skipping histogram/PDF creation.")
        return
    sampl_vars = config.sampl_cfg.vars

    title = " Writing PDFs "
    log.info(f"[bold green]{title:-^100}")
    log.info(
        f"[bold green]Estimating PDFs using {config.num_global_objects_estimate_hist:,} objects..."
    )

    # Create check variable to ensure at least one component was processed
    component_processed = not component_to_run

    # Process the different components
    for component in config.components:
        # Check if only one component should be processed
        if isinstance(component_to_run, str) and component_to_run != component.name:
            continue

        log.info(
            f"Estimating {component} PDF using "
            f"{config.num_global_objects_estimate_hist:,} samples..."
        )
        component.setup_reader(batch_size=config.batch_size, global_name=config.global_name)
        cuts_no_split = component.cuts.ignore(["eventNumber"])

        ###
        # TODO: return the number of objects here and pass to the next function to get started
        ###
        component.check_num_global_objects(
            config.num_global_objects_estimate_hist,
            cuts=cuts_no_split,
            silent=False,
            raise_error=False,
        )

        # Load the objects from file used for resampling
        global_objects = component.get_global_objects(
            variables=sampl_vars,
            num_global_objects=config.num_global_objects_estimate_hist,
            cuts=cuts_no_split,
        )

        # Write out the hist used for resampling
        component.hist.write_hist(
            global_objects=global_objects,
            resampling_vars=sampl_vars,
            bins=config.sampl_cfg.flat_bins,
            global_name=config.global_name,
        )

        # Set the check variable to true
        component_processed = True

    # Raise error of no region was processed
    if component_processed is False:
        raise ValueError(
            "No component processed during resampling! Check that you correctly spelled "
            "the component name when running with --component!"
        )

    log.info(f"[bold green]Saved to {config.components[0].hist.path.parent}/")
