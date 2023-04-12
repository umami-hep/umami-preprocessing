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
        self.jets_name = self.ppc.jets_name
        self.rng = np.random.default_rng(42)
        self.flavours = self.components.flavours

    def add_jet_flavour_label(self, jets, component):
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

        # write
        self.writer.write(merged)
        return len(merged[self.jets_name])

    def write_components(self, sample, components):
        # setup inputs
        for c in components:
            batch_size = self.batch_size * c.num_jets // components.num_jets + 1
            c.setup_reader(batch_size, fname=c.out_path)
            c.stream = c.reader.stream(self.variables.combined(), c.reader.num_jets)
            c.complete = False

        # setup outputs
        fname = self.ppc.out_fname
        if sample:
            fname = path_append(fname, sample)
        self.writer = H5Writer(
            components[0].reader.files[0],
            fname,
            self.variables.combined(),
            components.num_jets,
            add_flavour_label=self.jets_name,
        )
        self.writer.add_attr("flavour_label", [f.name for f in self.flavours], self.jets_name)
        self.writer.add_attr("unique_jets", components.unique_jets)
        for c in components:
            self.writer.add_attr(f"num_{c}", c.num_jets)
            self.writer.add_attr(f"num_unique_{c}", c.unique_jets)
        dsids = str(list(set(sum([c.sample.dsid for c in components], []))))
        self.writer.add_attr("dsids", dsids)
        sample_ids = str(list(set(sum([c.sample.sample_id for c in components], []))))
        self.writer.add_attr("sample_id", sample_ids)
        self.writer.add_attr("config", str(self.ppc.config))
        self.writer.add_attr("pp_hash", self.ppc.git_hash)
        self.writer.add_attr("sampling", self.ppc.sampl_cfg.method)
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
