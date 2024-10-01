from __future__ import annotations

import logging as log
import random
from pathlib import Path

import numpy as np
import yaml
from ftag.hdf5 import H5Reader
from yamlinclude import YamlIncludeConstructor

from upp.logger import ProgressBar
from upp.stages.hist import bin_jets
from upp.stages.interpolation import subdivide_bins, upscale_array_regionally

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
        self.jets_name = config.jets_name
        self.upscale_pdf = config.sampl_cfg.upscale_pdf or 1
        self.num_bins = self.get_num_bins_from_config()
        self.methods_map = {
            "pdf": self.pdf_select_func,
            "countup": self.countup_select_func,
            "none": None,
            None: None,
        }
        if self.config.method not in self.methods_map:
            raise ValueError(
                f"Unsupported resampling method {self.config.method}, choose from"
                f" {self.methods_map.keys()}"
            )
        self.select_func = self.methods_map[self.config.method]
        self.transform = config.transform

        self.rng = np.random.default_rng(42)

    def countup_select_func(self, jets, component):
        if self.upscale_pdf != 1:
            raise ValueError("Upscaling of histogrms is not supported for countup method")
        num_jets = int(len(jets) * component.sampling_fraction)
        target_pdf = self.target.hist.pbin
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
        if self.upscale_pdf > 1:
            bins = [subdivide_bins(bins, self.upscale_pdf) for bins in self.config.flat_bins]
        else:
            bins = self.config.flat_bins

        _hist, binnumbers = bin_jets(jets[self.config.vars], bins)
        # assert target_shape == _hist.shape
        if binnumbers.ndim > 1:
            binnumbers = tuple(binnumbers[i] for i in range(len(binnumbers)))

        # importance sample with replacement
        num_samples = int(len(jets) * component.sampling_fraction)
        ratios = safe_divide(self.target.hist.pbin, component.hist.pbin)
        if self.upscale_pdf > 1:
            ratios = upscale_array_regionally(ratios, self.upscale_pdf, self.num_bins)
        probs = ratios[binnumbers]
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
                if len(comp_idx) == 0:
                    continue
                batch_out = select_batch(batch, comp_idx)

                # apply sampling
                idx = np.arange(len(batch_out[self.variables.jets_name]))
                if c != self.target and self.select_func:
                    idx = self.select_func(batch_out[self.variables.jets_name], c)
                    if len(idx) == 0:
                        continue
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
            reader = H5Reader(
                sample.path,
                self.batch_size,
                jets_name=self.jets_name,
                equal_jets=equal_jets_flag,
                transform=self.transform,
            )
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

    def set_component_sampling_fractions(self):
        if self.config.sampling_fraction == "auto" or self.config.sampling_fraction is None:
            log.info("[bold green]Sampling fraction chosen for each component automatically...")
            for c in self.components:
                if c.is_target(self.config.target):
                    c.sampling_fraction = 1
                else:
                    sam_frac = c.get_auto_sampling_frac(c.num_jets, cuts=c.cuts)
                    if sam_frac > 1:
                        if self.config.method == "countup":
                            raise ValueError(
                                f"[bold red]Sampling fraction of {sam_frac:.3f}>1 is"
                                f" needed for component {c} This is not supported for"
                                " countup method."
                            )
                        else:
                            log.warning(
                                f"[bold yellow]sampling fraction of {sam_frac:.3f}>1 is"
                                f" needed for component {c}"
                            )
                    c.sampling_fraction = max(sam_frac, 0.1)
        else:
            for c in self.components:
                if c.is_target(self.config.target):
                    c.sampling_fraction = 1
                else:
                    c.sampling_fraction = self.config.sampling_fraction

    def run(self):
        title = " Running resampling "
        log.info(f"[bold green]{title:-^100}")
        log.info(f"Resampling method: {self.config.method}")

        # setup i/o
        for c in self.components:
            # just used for the writer configuration
            c.setup_reader(self.batch_size, jets_name=self.jets_name, transform=self.transform)
            c.setup_writer(self.variables, jets_name=self.jets_name)

        # set samplig fraction if needed
        self.set_component_sampling_fractions()

        # check samples
        log.info(
            "[bold green]Checking requested num_jets based on a sampling fraction of"
            f" {self.config.sampling_fraction}..."
        )
        for c in self.components:
            frac = c.sampling_fraction if self.select_func else 1
            c.check_num_jets(c.num_jets, sampling_frac=frac, cuts=c.cuts)

        # run resampling
        for region, components in self.components.groupby_region():
            log.info(f"[bold green]Running over region {region}...")
            self.run_on_region(components, region)

        # finalise
        unique = sum(c.writer.get_attr("unique_jets") for c in self.components)
        log.info(f"[bold green]Finished resampling a total of {self.components.num_jets:,} jets!")
        log.info(f"[bold green]Estimated unqiue jets: {unique:,.0f}")
        log.info(f"[bold green]Saved to {self.components.out_dir}/")

    def get_num_bins_from_config(self) -> list[list[int]]:
        """Get the lengths of the binning regions in each variable from the config.

        Returns
        -------
        typing.List[typing.List[int]]
            lengths of the binning regions in each variable from the config
        """
        num_bins = []
        for row in self.config.bins.values():
            num_bins.append([sub[-1] for sub in row])
        return num_bins
