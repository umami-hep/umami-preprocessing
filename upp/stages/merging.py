from __future__ import annotations

import json
import logging as log
from copy import copy

import numpy as np
from ftag.hdf5 import H5Writer, join_structured_arrays

from upp.logger import ProgressBar
from upp.utils import path_append


class Merging:
    def __init__(self, config):
        self.ppc = config
        self.components = config.components
        self.variables = config.variables
        self.batch_size = config.batch_size
        self.jets_name = config.jets_name
        self.rng = np.random.default_rng(42)
        self.flavours = self.components.flavours

    def add_jet_flavour_label(self, jets, component):
        if "flavour_label" in jets.dtype.names:
            return jets
        int_label = self.flavours.index(component.flavour)
        label_array = np.full(len(jets), int_label, dtype=[("flavour_label", "i4")])
        return join_structured_arrays([jets, label_array])

    def write_chunk(self, components):
        merged = {}
        for c in components:
            try:
                batch = copy(next(c.stream))  # shallow copy is needed since we add a variable
                batch[self.jets_name] = self.add_jet_flavour_label(batch[self.jets_name], c)

            except StopIteration:
                c.complete = True

            if c.complete:
                continue

            # merge components
            for name, array in batch.items():
                if name not in merged:
                    merged[name] = array
                else:
                    merged[name] = np.concatenate([merged[name], array])

        if all(c.complete for c in components):
            return False

        # apply track selections
        for name in self.variables.variables:
            if name == self.jets_name:
                continue
            if selector := self.variables.selectors.get(name):
                merged[name] = selector(merged[name])

        # write
        self.writer.write(merged)
        return len(merged[self.jets_name])

    def write_components(self, sample, components):
        # setup inputs
        for c in components:
            batch_size = self.batch_size * c.num_jets // components.num_jets + 1
            c.setup_reader(batch_size, fname=c.out_path, jets_name=self.jets_name)
            c.stream = c.reader.stream(self.variables.combined(), c.reader.num_jets)
            c.complete = False

        # setup outputs
        fname = self.ppc.out_fname
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
        self.writer.add_attr("config", json.dumps(self.ppc.config))
        self.writer.add_attr("upp_hash", self.ppc.git_hash)
        log.debug(f"Setup merge output at {self.writer.dst}")

        # write
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
        title = " Running Merging "
        log.info(f"[bold green]{title:-^100}")

        if not self.ppc.is_test or self.ppc.merge_test_samples:
            components = [(None, self.components)]
        else:
            components = self.components.groupby_sample()

        for sample, comps in components:
            self.write_components(sample, comps)
