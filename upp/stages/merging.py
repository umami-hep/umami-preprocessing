from __future__ import annotations

import json
import logging as log
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from ftag.hdf5 import H5Writer, join_structured_arrays

from upp.logger import ProgressBar
from upp.utils import path_append

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.components import Component, Components
    from upp.classes.preprocessing_config import PreprocessingConfig


class Merging:
    """Merging Class to merge different components/regions."""

    def __init__(self, config: PreprocessingConfig):
        """Init the Merging class instance.

        Parameters
        ----------
        config : PreprocessingConfig
            Loaded preprocessing config as a PreprocessingConfig instance
        """
        self.config = config
        self.components = config.components
        self.variables = config.variables
        self.batch_size = config.batch_size
        self.jets_name = config.jets_name
        self.rng = np.random.default_rng(42)
        self.flavours = self.components.flavours

    def add_jet_flavour_label(self, jets: np.ndarray, component: Component) -> np.ndarray:
        """Add the jet flavour label to the jets.

        If already present, jets will be returned without any changes.

        Parameters
        ----------
        jets : np.ndarray
            Structured array of with the jets and their variables
        component : Component
            Component instance of the

        Returns
        -------
        np.ndarray
            Structured array of the jets and their variables with the
            "flavour_label" added.
        """
        if "flavour_label" in jets.dtype.names:
            return jets
        int_label = self.flavours.index(component.flavour)
        label_array = np.full(len(jets), int_label, dtype=[("flavour_label", "i4")])

        return join_structured_arrays([jets, label_array])

    def write_chunk(self, components: Components) -> int:
        """Write chunk to file.

        Parameters
        ----------
        components : Components
            Components instance of all components that are written

        Returns
        -------
        int
            Number of jets written to file
        """
        merged = {}
        for component in components:
            try:
                # Shallow copy is needed since we add a variable
                batch = copy(next(component.stream))
                batch[self.jets_name] = self.add_jet_flavour_label(
                    jets=batch[self.jets_name],
                    component=component,
                )

            except StopIteration:
                component.complete = True

            if component.complete:
                continue

            # Merge components
            for name, array in batch.items():
                if name not in merged:
                    merged[name] = array
                else:
                    merged[name] = np.concatenate([merged[name], array])

        if all(component.complete for component in components):
            return False

        # Apply track selections
        for name in self.variables.variables:
            if name == self.jets_name:
                continue
            if selector := self.variables.selectors.get(name):
                merged[name] = selector(merged[name])

        # Write
        self.writer.write(merged)

        return len(merged[self.jets_name])

    def write_components(self, sample: str, components: Components) -> None:
        # setup inputs
        for component in components:
            batch_size = self.batch_size * component.num_jets // components.num_jets + 1
            component.setup_reader(
                batch_size,
                fname=component.out_path,
                jets_name=self.jets_name,
            )
            component.stream = component.reader.stream(
                self.variables.combined(),
                component.reader.num_jets,
            )
            component.complete = False

        # setup outputs
        fname = self.config.out_fname
        if sample:
            fname = path_append(fname, sample)
        self.writer = H5Writer(
            fname,
            components[0].reader.dtypes(self.variables.combined()),
            components[0].reader.shapes(components.num_jets, self.variables.keys()),
            add_flavour_label=self.jets_name,
            jets_name=self.jets_name,
        )
        self.writer.add_attr("flavour_label", [f.name for f in self.flavours], self.jets_name)
        self.writer.add_attr("unique_jets", components.unique_jets)
        self.writer.add_attr("jet_counts", json.dumps(components.jet_counts))
        self.writer.add_attr("dsids", str(components.dsids))
        self.writer.add_attr("config", json.dumps(self.config.config))
        self.writer.add_attr("upp_hash", self.config.git_hash)
        log.debug(f"Setup merge output at {self.writer.dst}")

        # Write
        with ProgressBar() as progress:
            task = progress.add_task(
                f"[green]Merging {components.num_jets:,} jets...",
                total=components.num_jets,
            )
            while True:
                n = self.write_chunk(components)
                if not n:
                    break
                progress.update(task, advance=n)

        self.writer.close()
        sample = "merged" if sample is None else sample
        log.info(f"[bold green]Finished merging {components.num_jets:,} {sample} jets!")
        log.info(f"[bold green]Saved to {fname}")

    def run(self):
        """Run merging of the components."""
        title = " Running Merging "
        log.info(f"[bold green]{title:-^100}")

        if not self.config.is_test or self.config.merge_test_samples:
            components = [(None, self.components)]
        else:
            components = self.components.groupby_sample()

        for sample, comps in components:
            self.write_components(sample, comps)
