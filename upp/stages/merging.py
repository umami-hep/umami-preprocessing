from __future__ import annotations

import json
import logging as log
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
from ftag.hdf5 import H5Writer, join_structured_arrays

from upp.utils.logger import ProgressBar
from upp.utils.tools import path_append

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.components import Component, Components
    from upp.classes.preprocessing_config import PreprocessingConfig


class Merging:
    """Merging Classto merge different components/regions."""

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

        # Auto-resume toggle (make configurable if you prefer opt-in)
        self.resume = True

        # Auto-delete a corrupted existing part before resuming
        self.auto_fix_parts = True

        # Internal state, guard to keep fast-forward from opening files
        self._fast_forwarding: bool = False

        # Pending tail (used only across the fast-forward boundary)
        self._ff_pending: dict[str, np.ndarray] | None = None

        # perfectly valid, no union necessary
        self.dtypes: dict[str, np.dtype] = {}
        self.base_shapes: dict[str, tuple[int, ...]] = {}

        # Setup all the jet counters
        self._file_idx: int = 0
        self.total_jets: int = 0
        self.jets_written: int = 0

        # Setup the sample string
        self._sample: str | None = None

        # Use cast because we cannot init Components/H5Writer here
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

    def _part_fname(self, sample: str | None, file_idx: int) -> Path:
        """Construct the exact output filename for a given part index.

        Parameters
        ----------
        sample : str | None
            Name of the output sample
        file_idx : int
            Iterator number of the output file

        Returns
        -------
        Path
            Final path to the output file
        """
        # Get base path of the output
        fname = Path(self.config.out_fname)

        # Append the sample name to the file name
        if sample:
            fname = path_append(fname, sample)

        # Define the suffix for the file (including the iterator number)
        suffix = f"{self.file_tag}_{file_idx:03d}"

        # Return the final path
        return fname.with_name(f"{fname.stem}_{suffix}{fname.suffix}")

    def _expected_rows_for_part(self, part_idx: int) -> int:
        """Return the expected number of rows for part `part_idx` given total_jets and split size.

        Parameters
        ----------
        part_idx : int
            Iterator number of the file

        Returns
        -------
        int
            Expected number of rows for the given partial file
        """
        # Assert that the final output file will be splitted
        assert self.num_jets_per_output_file is not None

        # Remaining jets starting at this part
        start = part_idx * int(self.num_jets_per_output_file)
        remaining = max(0, self.total_jets - start)

        return min(int(self.num_jets_per_output_file), remaining)

    def _is_part_valid(self, sample: str | None, part_idx: int) -> bool:
        """Heuristically validate that a part file is complete and consistent.

        Checks:
          - File can be opened.
          - All expected datasets exist (based on self.base_shapes keys).
          - All datasets share the same first-dimension length.
          - First-dimension equals the expected rows for this part.

        Parameters
        ----------
        sample : str | None
            Name of the sample
        part_idx : int
            Iterator number of the file

        Returns
        -------
        bool
            Check that the partial file is complete and valid.
        """
        # Get the file path
        fname = self._part_fname(sample, part_idx)

        # Try to open the h5 file
        try:
            with h5py.File(fname, "r") as f:
                # Collect expected dataset names from base_shapes (already computed)
                expected_names = list(self.base_shapes.keys())

                # Tolerate missing optional groups, but require the jet dataset at least
                if self.jets_name not in f:
                    log.warning(f"Missing dataset '{self.jets_name}' in {fname}")
                    return False

                # Determine observed length from anchor (jets_name) or first dataset
                anchor = self.jets_name if self.jets_name in f else expected_names[0]
                if anchor not in f:
                    # if jets_name wasn't found, try any expected dataset that exists
                    for nm in expected_names:
                        if nm in f:
                            anchor = nm
                            break

                if anchor not in f:
                    log.warning(f"No expected datasets found in {fname}")
                    return False

                obs_len = f[anchor].shape[0]

                # All expected datasets that are present should match obs_len
                for nm in expected_names:
                    if nm in f and f[nm].shape[0] != obs_len:
                        log.warning(
                            f"Dataset '{nm}' len={f[nm].shape[0]} " f"!= {obs_len} in {fname}"
                        )
                        return False

                # Compare with expected rows for this part (if split mode)
                if self.num_jets_per_output_file is not None:
                    exp_len = self._expected_rows_for_part(part_idx)
                    if obs_len != exp_len:
                        log.warning(
                            f"Part {part_idx:03d} in {fname} has {obs_len} rows, "
                            f"expected {exp_len}."
                        )
                        return False

                return True

        # Except the file is broken
        except OSError as e:
            # Typical for truncated/half-written files
            log.warning(f"Failed to open {fname}: {e}")
            return False

    def _detect_and_clean_completed_parts(self, sample: str | None) -> int:
        """Detect valid and invalid parts and remove the invalid path.

        Count contiguous **valid** parts; if the first invalid part is found and
        `auto_fix_parts` is enabled, delete it so resume can overwrite it.

        Parameters
        ----------
        sample : str | None
            Name of the sample to use

        Returns
        -------
        int
            The index of the first missing/invalid part.
        """
        # Check that multiple output files should be created
        if self.num_jets_per_output_file is None:
            return 0

        # Define a counter
        idx = 0

        # Loop over the files
        while True:
            # Get the name of the file
            fname = self._part_fname(sample, idx)

            # If the file doesn't exist, stop the loop
            if not fname.exists():
                break

            # Validate the existing file
            if not self._is_part_valid(sample, idx):
                if self.auto_fix_parts:
                    try:
                        fname.unlink()
                        log.warning(
                            f"[bold yellow]Deleted corrupted part: {fname.name} "
                            f"(will be re-written)."
                        )
                    except OSError as e:
                        log.error(f"Could not delete corrupted part {fname}: {e}")

                # Stop at the first invalid file (deleted or left as-is)
                break

            # Go to next file
            idx += 1

        # Return the idx number of the new file
        return idx

    class _NullWriter:
        """A minimal writer that discards data while tracking how much would be written."""

        def __init__(self, capacity: int):
            self.num_jets = capacity
            self.num_written = 0

        def write(self, batch: dict[str, np.ndarray]) -> None:
            """Count the number of jets that would be written.

            Parameters
            ----------
            batch : dict[str, np.ndarray]
                Dict with the batches
            """
            # advance by the leading dimension of any array (they are aligned)
            if not batch:
                return
            any_arr = next(iter(batch.values()))
            k = len(any_arr)
            self.num_written = min(self.num_written + k, self.num_jets)

        def add_attr(self, *args, **kwargs):
            """Skip the attribute addition."""
            pass

        def close(self):
            """Skip the close."""
            pass

    def _open_writer(
        self,
        sample: str | None,
        jets_in_file: int,
        file_idx: int,
        components: Components,
    ) -> None:
        """Create `self.writer` for the next output file and attach static attributes.

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
        """Read one chunk, merge and write it to disk (or discard in fast-forward).

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

        # 1) Use pending tail first (only set across fast-forward boundary)
        if self._ff_pending is not None:
            merged = self._ff_pending
            self._ff_pending = None
        else:
            # 2) Otherwise, pull one batch from every active component
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

        # If nothing merged and all components are exhausted -> stop
        if not merged and all(c.complete for c in components):
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

        if self._fast_forwarding:
            # Limit consumption to the remaining discard quota
            if merged_len <= capacity_left:
                self.writer.write(merged)
                self.jets_written += merged_len
                return merged_len
            else:
                head = {n: a[:capacity_left] for n, a in merged.items()}
                tail = {n: a[capacity_left:] for n, a in merged.items()}
                self.writer.write(head)
                self._ff_pending = tail  # keep remainder for next iteration
                self.jets_written += capacity_left
                return capacity_left

        # If current file is full (and not fast-forwarding), roll to next file
        if capacity_left == 0 and not self._fast_forwarding:
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

        # Write (or discard) the batch
        if merged_len <= capacity_left or self._fast_forwarding:
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
        """Merge *components* into one or more HDF5 files.

        If ``self.num_jets_per_output_file`` is ``None`` the behaviour is identical to the
        original implementation (exactly one output file).  Otherwise the function
        keeps opening new `H5Writer`s whenever the current file reaches that jet
        limit.  All heavy work (splitting batches, rolling files) is handled in
        ``self.write_chunk``.

        Parameters
        ----------
        sample : str | None
            Name of the sample
        components : Components
            Components that are to be written
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

        # Auto-resume: detect contiguous valid parts; delete a corrupt last part if found
        resume_parts = 0
        if self.resume and isinstance(self.num_jets_per_output_file, int):
            resume_parts = self._detect_and_clean_completed_parts(sample)

        if resume_parts and isinstance(self.num_jets_per_output_file, int):
            to_discard = resume_parts * int(self.num_jets_per_output_file)
            log.info(
                f"[bold yellow]Resuming merge: found {resume_parts} completed part(s); "
                f"skipping first {to_discard:,} jets."
            )
            # Use a NullWriter to pre-consume data via the exact same logic
            self._fast_forwarding = True
            self.writer = self._NullWriter(to_discard)
            while self.jets_written < to_discard:
                consumed = self.write_chunk(components)
                if consumed == 0:
                    break
            self.writer.close()
            self._fast_forwarding = False

            # Align counters with the next missing part
            self._file_idx = resume_parts
            self.jets_written = to_discard

        # Decide capacity of the first real file
        remaining_total = self.total_jets - self.jets_written
        first_file_size = (
            min(self.num_jets_per_output_file, remaining_total)
            if self.num_jets_per_output_file
            else remaining_total
        )

        # Open the first output file
        self._open_writer(sample, first_file_size, self._file_idx, components)

        # Main merge loop with progress
        with ProgressBar() as progress:
            task = progress.add_task(
                f"[green]Merging {components.num_jets:,} jets...",
                total=components.num_jets,
            )
            if self.jets_written:
                progress.update(task, advance=self.jets_written)

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
