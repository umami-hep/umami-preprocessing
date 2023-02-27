import logging as log
import random
from pathlib import Path

import numpy as np
import yaml
from yamlinclude import YamlIncludeConstructor

from jetpp.hdf5 import H5Reader
from jetpp.logger import ProgressBar
from jetpp.stages.hist import bin_jets

random.seed(42)

# support inclusion of yaml files in the config dir
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.SafeLoader, base_dir=Path(__file__).parent.parent / "configs"
)


def select_batch(batch, idx):
    batch_out = {}
    for name, array in batch.items():
        batch_out[name] = array[idx]
    return batch_out


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


class Resampling:
    def __init__(self, config):
        self.config = config.sampl_cfg
        self.components = config.components
        self.variables = config.variables
        self.batch_size = config.batch_size
        self.is_test = config.is_test
        self.num_jets_estimate = config.num_jets_estimate
        if self.config.method == "pdf":
            self.select_func = self.pdf_select_func
        elif self.config.method == "countup":
            self.select_func = self.countup_select_func
        else:
            raise ValueError(f"Unsupported resampling method {self.config.method}")
        self.rng = np.random.default_rng(42)

    def countup_select_func(self, jets, component):
        num_jets = int(len(jets) * self.config.sampling_fraction)
        target_pdf = self.target.hist.pdf
        target_hist = target_pdf * num_jets
        target_hist = (np.floor(target_hist + self.rng.random(target_pdf.shape))).astype(int)
        _hist, binnumbers = bin_jets(jets[self.config.vars], self.config.flat_bins)
        assert target_pdf.shape == _hist.shape

        # loop over bins and select relevant jets
        all_idx = []
        for bin_id in np.ndindex(*target_hist.shape):
            idx = np.where((binnumbers.T == bin_id).all(axis=-1))[0][: target_hist[bin_id]]
            if len(idx) and len(idx) < target_hist[bin_id]:
                idx = np.concatenate([idx, self.rng.choice(idx, target_hist[bin_id] - len(idx))])
            all_idx.append(idx)
        idx = np.concatenate(all_idx).astype(int)
        if len(idx) < num_jets:
            idx = np.concatenate([idx, self.rng.choice(idx, num_jets - len(idx))])
        self.rng.shuffle(idx)
        # log.info(f"final output is {len(idx):,}/{num_jets:,} jets, or {len(idx)/num_jets:.3%}")
        self.track_upsampling_stats(idx, component)
        return idx

    def pdf_select_func(self, jets, component):
        # bin jets
        _hist, binnumbers = bin_jets(jets[self.config.vars], self.config.flat_bins)
        assert self.target.hist.pdf.shape == _hist.shape
        if binnumbers.ndim > 1:
            binnumbers = tuple(binnumbers[i] for i in range(len(binnumbers)))

        # importance sample with replacement
        num_samples = int(len(jets) * self.config.sampling_fraction)
        probs = safe_divide(self.target.hist.pdf, component.hist.pdf)[binnumbers]
        idx = random.choices(np.arange(len(jets)), weights=probs, k=num_samples)
        self.track_upsampling_stats(idx, component)

        return idx

    def track_upsampling_stats(self, idx, component):
        ups_counts = np.unique(idx, return_counts=True)[1]
        mean_ups = ups_counts.mean()
        max_ups = ups_counts.max()
        num_written = component.writer._num_written
        component._ups_ratio = (mean_ups * len(idx) + component._ups_ratio * num_written) / (
            len(idx) + num_written
        )
        component._ups_max = max_ups if max_ups > component._ups_max else component._ups_max

    def sample(self, components, stream, progress):
        # loop through input file
        for batch in stream:
            for c in components:
                if c._complete:
                    continue

                # apply selections
                comp_idx, _ = c.flavour.cuts(batch[self.variables.jets_name])
                batch_out = select_batch(batch, comp_idx)

                # apply sampling
                idx = np.arange(len(batch_out[self.variables.jets_name]))
                if c != self.target and not self.is_test:
                    idx = self.select_func(batch_out[self.variables.jets_name], c)
                    batch_out = select_batch(batch_out, idx)

                # check for completion
                if c.writer._num_written + len(idx) >= c.num_jets:
                    keep = c.num_jets - c.writer._num_written
                    for name, array in batch_out.items():
                        batch_out[name] = array[:keep]
                    c._complete = True

                # write
                c.writer.write(batch_out)
                progress.update(c.pbar, advance=len(idx))
                if c._complete:
                    c.writer.add_attr("upsampling_ratio", c._ups_ratio)
                    c.writer.add_attr("unique_jets", int(c.num_jets / c._ups_ratio))
                    c.writer.close()

            # check for completion
            if all(c._complete for c in components):
                break

        for c in components:
            if not c._complete:
                raise ValueError(f"Ran out of {c} jets after writing {c.writer._num_written:,}")

    def run_on_region(self, components, region):
        # compute the target pdf
        target = [c for c in components if c.is_target(self.config.target)]
        assert len(target) == 1, "Should have 1 target component per region"
        self.target = target[0]

        # groupby samples
        for s, cs in components.groupby_sample():
            # setup input stream
            variables = self.variables.add_jet_vars(cs.cuts.variables)
            reader = H5Reader(s.vds_path, self.batch_size, self.variables.jets_name)
            stream = reader.stream(variables, reader.num_jets, region.cuts)

            # run with progress
            with ProgressBar() as progress:
                for c in cs:
                    c._complete = False
                    c._ups_max = c._ups_ratio = 1.0

                for c in cs:
                    c.pbar = progress.add_task(
                        f"[green]Sampling {c.num_jets:,} jets from {c}...", total=c.num_jets
                    )

                # run sampling
                self.sample(cs, stream, progress)

        # print upsampling factors
        for c in components:
            log.info(
                f"{c} usampling ratio is {np.mean(c._ups_ratio):.3f} for an estimated"
                f" {c.num_jets/np.mean(c._ups_ratio):,.0f}/{c.num_jets:,} unique jets."
                f" Jets are upsampled at most {np.max(c._ups_max):.0f} times"
            )

    def run(self):
        title = " Running resampling "
        log.info(f"[bold green]{title:-^100}")
        log.info(f"Resampling method: {self.config.method}")

        # setup i/o
        for c in self.components:
            c.setup_reader(self.variables, self.batch_size)
            c.setup_writer(self.variables)

        # check samples
        log.info(
            "[bold green]Checking requested num_jets based on a sampling fraction of"
            f" {self.config.sampling_fraction}..."
        )
        for c in self.components:
            sampling_frac = 1 if c.is_target(self.config.target) else self.config.sampling_fraction
            c.check_num_jets(c.num_jets, sampling_frac=sampling_frac)

        # run resampling
        for region, components in self.components.groupby_region():
            log.info(f"[bold green]Running over region {region}...")
            self.run_on_region(components, region)

        # finalise
        unique = sum(c.writer.get_attr("unique_jets") for c in self.components)
        log.info(f"[bold green]Finished resampling a total of {self.components.num_jets:,} jets!")
        log.info(f"[bold green]Estimated unqiue jets: {unique:,.0f}")
        log.info(f"[bold green]Saved to {self.components.out_dir}/")
