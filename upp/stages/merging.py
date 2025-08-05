from __future__ import annotations

import json
import logging as log
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from ftag.hdf5 import H5Writer, join_structured_arrays

from upp.utils.logger import ProgressBar
from upp.utils.tools import path_append

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.components import Component, Components
    from upp.classes.preprocessing_config import PreprocessingConfig


class Merging:
    """Merging Class to merge different components/regions."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.components = config.components
        self.variables = config.variables
        self.batch_size = config.batch_size
        self.jets_name = config.jets_name
        self.rng = np.random.default_rng(42)
        self.flavours = self.components.flavours
        self.num_jets_per_output_file = config.num_jets_per_output_file
        self.file_tag = "split"

        # perfectly valid, no union necessary
        self.dtypes: dict[str, np.dtype] = {}
        self.base_shapes: dict[str, tuple[int, ...]] = {}

        # Setup all the jet counters
        self._file_idx: int = 0
        self.total_jets: int = 0
        self.jets_written: int = 0

        # Setup the sample string
        self._sample: str | None = None

        # Use cast because we cannot init Components/H5Writer
        self.current_components = cast("Components", None)
        self.writer = cast(H5Writer, None)

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

    def _open_writer(
        self,
        sample: str | None,
        jets_in_file: int,
        file_idx: int,
        components: Components,
    ) -> None:
        """Create `self.writer` for the next output file and attach all static attributes.

        Parameters
        ----------
        sample : str | None
            Sample name (``None`` for the "train/val test" merge).
        jets_in_file : int
            Capacity of the new file (= leading dimension of every dataset).
        file_idx : int
            Running part index (0, 1, 2, …); used only for the filename suffix.
        components : Components
            The `Components` object we are currently merging needed for `jet_counts`, etc.
        """
        # Construct the filename
        fname = Path(self.config.out_fname)

        if sample:
            fname = path_append(fname, sample)

        if self.num_jets_per_output_file is not None:
            suffix = f"{self.file_tag}_{file_idx:03d}"
            fname = fname.with_name(f"{fname.stem}_{suffix}{fname.suffix}")

        # Adjust shapes to the capacity of this file
        shapes = {name: (jets_in_file,) + shape[1:] for name, shape in self.base_shapes.items()}

        # Instantiate an H5Writer
        self.writer = H5Writer(
            fname,
            self.dtypes,
            shapes,
            add_flavour_label=self.jets_name,
            jets_name=self.jets_name,
            num_jets=jets_in_file,
        )

        # Copy the metadata attributes
        self.writer.add_attr(
            "flavour_label",
            [f.name for f in self.flavours],
            self.jets_name,
        )
        self.writer.add_attr("unique_jets", components.unique_jets)
        self.writer.add_attr("jet_counts", json.dumps(components.jet_counts))
        self.writer.add_attr("dsids", str(components.dsids))
        self.writer.add_attr("config", json.dumps(self.config.config))
        self.writer.add_attr("upp_hash", self.config.git_hash)

        # Log for debugging
        log.debug(f"Setup merge output at {self.writer.dst}")

    def write_chunk(self, components: Components) -> int:
        """Read one chunk, merge and write it to disk.

        Read one batch from every active component, merge them and write
        them to disk. If the batch does not fit into the current file it is
        split across files transparently.

        Parameters
        ----------
        components : Components
            Components that are to be written.

        Returns
        -------
        int
            The number of jets that were consumed from the components
            (== written to disk).  When all components are exhausted the
            function returns 0 so that the caller can stop its loop.
        """
        # Init a merged dict
        merged: dict[str, np.ndarray] = {}

        # Loop over components
        for component in components:
            try:
                # shallow copy because we will add a field
                batch = copy(next(component.stream))
                batch[self.jets_name] = self.add_jet_flavour_label(
                    jets=batch[self.jets_name], component=component
                )
            except StopIteration:
                component.complete = True

            if component.complete:
                continue

            # Merge this component's arrays into the running dict
            for name, array in batch.items():
                if name not in merged:
                    merged[name] = array
                else:
                    merged[name] = np.concatenate([merged[name], array])

        # Stop if there is nothing more to read
        if all(c.complete for c in components):
            return 0

        # Apply track selections
        for name in self.variables.variables:
            if name == self.jets_name:
                continue
            if selector := self.variables.selectors.get(name):
                merged[name] = selector(merged[name])

        # Get the total length of jets from the batch and how much
        # capacity is left in the file
        merged_len = len(merged[self.jets_name])
        capacity_left = self.writer.num_jets - self.writer.num_written

        # Check if the capacity of the given file is already zero
        if capacity_left == 0:
            # close the filled file
            self.writer.close()

            # open the next one
            self._file_idx += 1
            remaining_total = self.total_jets - self.jets_written

            # Quit writing when no jets are left to write
            if remaining_total == 0:
                return 0

            next_file_size = (
                min(self.num_jets_per_output_file, remaining_total)
                if self.num_jets_per_output_file
                else remaining_total
            )
            self._open_writer(
                self._sample,
                next_file_size,
                self._file_idx,
                self.current_components,
            )

            # Recompute free space in the freshly-opened file
            capacity_left = self.writer.num_jets - self.writer.num_written

        # Check if the whole batch fits into the file
        if merged_len <= capacity_left:
            # whole batch fits
            self.writer.write(merged)

        else:
            # Write the *head* that still fits into the present file
            head = {n: a[:capacity_left] for n, a in merged.items()}
            self.writer.write(head)
            self.writer.close()

            # Open a fresh file sized for the remaining jets
            self._file_idx += 1
            remaining_total = self.total_jets - (self.jets_written + capacity_left)
            next_file_size = (
                min(self.num_jets_per_output_file, remaining_total)
                if self.num_jets_per_output_file
                else remaining_total
            )
            self._open_writer(self._sample, next_file_size, self._file_idx, self.current_components)

            # Write the *tail* that goes into the new file
            tail = {n: a[capacity_left:] for n, a in merged.items()}
            self.writer.write(tail)

        # Updating the progress-bar
        self.jets_written += merged_len
        return merged_len

    def write_components(self, sample: str | None, components: Components) -> None:
        """
        Merge *components* into one or more HDF5 files.

        If ``self.num_jets_per_output_file`` is ``None`` the behaviour is identical to the
        original implementation (exactly one output file).  Otherwise the function
        keeps opening new `H5Writer`s whenever the current file reaches that jet
        limit.  All heavy work (splitting batches, rolling files) is handled in
        ``self.write_chunk``.
        """
        # Prepare every Component's reader
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

        # Cache dtype / base shapes once (re-used for every new file)
        self.dtypes = components[0].reader.dtypes(self.variables.combined())
        self.base_shapes = components[0].reader.shapes(components.num_jets, self.variables.keys())

        # Bookkeeping shared with write_chunk
        self.total_jets = components.num_jets
        self.jets_written = 0
        self._file_idx = 0
        self._sample = sample
        self.current_components = components

        # decide capacity of the first file
        first_file_size = (
            min(self.num_jets_per_output_file, self.total_jets)
            if self.num_jets_per_output_file
            else self.total_jets
        )

        # Open the first output file
        self._open_writer(sample, first_file_size, self._file_idx, components)

        # Main merge loop (progress bar unchanged)
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

        # Close Writer
        self.writer.close()
        label = "merged" if sample is None else sample
        log.info(f"[bold green]Finished merging {components.num_jets:,} {label} jets!")

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
