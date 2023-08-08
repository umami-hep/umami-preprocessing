import logging as log
import random
from pathlib import Path

import numpy as np
import yaml
from ftag.hdf5 import H5Reader
from yamlinclude import YamlIncludeConstructor

from upp.logger import ProgressBar
from upp.stages.hist import bin_jets

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
        elif not self.config.method or self.config.method == "none":
            self.select_func = None
        else:
            raise ValueError(f"Unsupported resampling method {self.config.method}")
        self.rng = np.random.default_rng(42)

    def countup_select_func(self, jets, component):  # noqa: ARG002
        num_jets = int(len(jets) * self.config.sampling_fraction)
        target_pdf = self.target.hist.pdf
        target_hist = target_pdf * num_jets
        target_hist = (np.floor(target_hist + self.rng.random(target_pdf.shape))).astype(int)
        _hist, binnumbers = bin_jets(jets[self.config.vars], self.config.flat_bins)
        assert target_pdf.shape == _hist.shape

        # loop over bins and select relevant jets
        all_idx = []
        for bin_id in np.ndindex(*target_hist.shape):
            idx = np.where((bin_id == binnumbers.T).all(axis=-1))[0][: target_hist[bin_id]]
            if len(idx) and len(idx) < target_hist[bin_id]:
                idx = np.concatenate([idx, self.rng.choice(idx, target_hist[bin_id] - len(idx))])
            all_idx.append(idx)
        idx = np.concatenate(all_idx).astype(int)
        if len(idx) < num_jets:
            idx = np.concatenate([idx, self.rng.choice(idx, num_jets - len(idx))])
        self.rng.shuffle(idx)
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
        return idx

    def track_upsampling_stats(self, idx, component):
        unique, ups_counts = np.unique(idx, return_counts=True)
        component._unique_jets += len(unique)
        max_ups = ups_counts.max()
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
                if c != self.target and not self.is_test and self.select_func:
                    idx = self.select_func(batch_out[self.variables.jets_name], c)
                    batch_out = select_batch(batch_out, idx)

                # check for completion
                if c.writer.num_written + len(idx) >= c.num_jets:
                    keep = c.num_jets - c.writer.num_written
                    idx = idx[:keep]
                    for name, array in batch_out.items():
                        batch_out[name] = array[:keep]
                    c._complete = True

                # track upsampling stats
                self.track_upsampling_stats(idx, c)

                # write
                c.writer.write(batch_out)
                progress.update(c.pbar, advance=len(idx))
                if c._complete:
                    c._ups_ratio = c.writer.num_written / c._unique_jets
                    c.writer.add_attr("upsampling_ratio", c._ups_ratio)
                    c.writer.add_attr("unique_jets", c._unique_jets)
                    c.writer.add_attr("dsid", str(c.sample.dsid))
                    c.writer.close()

            # check for completion
            if all(c._complete for c in components):
                break

        for c in components:
            if not c._complete:
                raise ValueError(f"Ran out of {c} jets after writing {c.writer.num_written:,}")

    def run_on_region(self, components, region):
        # compute the target pdf
        target = [c for c in components if c.is_target(self.config.target)]
        assert len(target) == 1, "Should have 1 target component per region"
        self.target = target[0]

        # groupby samples
        for sample, cs in components.groupby_sample():
            # make sure all tags equal_jets are the same
            equal_jets_flags = [c.equal_jets for c in cs]
            if len(set(equal_jets_flags)) != 1:
                raise ValueError("equal_jets must be the same for all components in a sample")
            equal_jets_flag = equal_jets_flags[0]

            # setup input stream
            variables = self.variables.add_jet_vars(cs.cuts.variables)
            reader = H5Reader(sample.path, self.batch_size, equal_jets=equal_jets_flag)
            stream = reader.stream(variables.combined(), reader.num_jets, region.cuts)

            # run with progress
            with ProgressBar() as progress:
                for c in cs:
                    c._complete = False
                    c._ups_max = 1.0
                    c._unique_jets = 0

                for c in cs:
                    c.pbar = progress.add_task(
                        f"[green]Sampling {c.num_jets:,} jets from {c}...",
                        total=c.num_jets,
                    )

                # run sampling
                self.sample(cs, stream, progress)

        # print upsampling factors
        for c in components:
            log.info(
                f"{c} usampling ratio is {np.mean(c._ups_ratio):.3f}, with"
                f" {c.num_jets/np.mean(c._ups_ratio):,.0f}/{c.num_jets:,} unique jets."
                f" Jets are upsampled at most {np.max(c._ups_max):.0f} times"
            )

    def set_auto_sampling_fraction(self):
        optimal_frac_list = []
        for c in self.components:
            if not c.is_target(self.config.target):
                optimal_frac_list.append(c.get_auto_sampling_frac(c.num_jets, cuts=c.cuts))
        optimal_frac = np.max(optimal_frac_list)
        optimal_frac = max(optimal_frac, 0.1)
        log.info("[bold green]Auto sampling fraction chosen")
        if optimal_frac > 1:
            if self.config.method == "countup":
                raise ValueError(
                    f"Sampling fraction of {optimal_frac:.3f}>1 is needed for one"
                    " or more components. This is not supported for countup"
                    " method."
                )
            else:
                log.warning(
                    f"[bold yellow]sampling fraction of {optimal_frac:.3f}>1 is"
                    " needed for one or more components."
                )
        log.info(f"[bold green]setting sampling fraction to {optimal_frac:.3f}...")
        self.config.sampling_fraction = optimal_frac

    def run(self):
        title = " Running resampling "
        log.info(f"[bold green]{title:-^100}")
        log.info(f"Resampling method: {self.config.method}")

        # setup i/o
        for c in self.components:
            c.setup_reader(self.batch_size)
            c.setup_writer(self.variables)

        # set samplig fraction if needed
        if self.config.sampling_fraction == "auto" or self.config.sampling_fraction is None:
            self.set_auto_sampling_fraction()

        # check samples
        log.info(
            "[bold green]Checking requested num_jets based on a sampling fraction of"
            f" {self.config.sampling_fraction}..."
        )
        for c in self.components:
            sampling_frac = 1 if c.is_target(self.config.target) else self.config.sampling_fraction
            c.check_num_jets(c.num_jets, sampling_frac=sampling_frac, cuts=c.cuts)

        # run resampling
        for region, components in self.components.groupby_region():
            log.info(f"[bold green]Running over region {region}...")
            self.run_on_region(components, region)

        # finalise
        unique = sum(c.writer.get_attr("unique_jets") for c in self.components)
        log.info(f"[bold green]Finished resampling a total of {self.components.num_jets:,} jets!")
        log.info(f"[bold green]Estimated unqiue jets: {unique:,.0f}")
        log.info(f"[bold green]Saved to {self.components.out_dir}/")
