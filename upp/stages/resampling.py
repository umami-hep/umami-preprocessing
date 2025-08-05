from __future__ import annotations

import logging as log
import random
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import yaml
from ftag.hdf5 import H5Reader
from yamlinclude import YamlIncludeConstructor

from upp.stages.hist import bin_jets
from upp.stages.interpolation import subdivide_bins, upscale_array_regionally
from upp.utils.logger import ProgressBar

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Generator

    from rich.progress import Progress

    from upp.classes.components import Component, Components
    from upp.classes.preprocessing_config import PreprocessingConfig
    from upp.classes.region import Region

random.seed(42)

# support inclusion of yaml files in the config dir
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.SafeLoader, base_dir=Path(__file__).parent.parent / "configs"
)


def select_batch(batch: dict, idx) -> dict:
    batch_out = {}
    for name, array in batch.items():
        batch_out[name] = array[idx]
    return batch_out


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


class Resampling:
    """Class for resampling of the different flavours/classes."""

    def __init__(self, config: PreprocessingConfig):
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

        # Define what type self.target will be
        self.target = cast("Component", None)

    def countup_select_func(self, jets: dict, component: Component) -> np.ndarray:
        """Countup resampling function.

        Parameters
        ----------
        jets : dict
            Dict with the jets which are to be resampled.
        component : Component
            Component instance for a given flavour/class.

        Returns
        -------
        np.ndarray
            Numpy array with the index numbers of the jets that are to be used.

        Raises
        ------
        ValueError
            If the upscale factor is unequal to one. Upscaling is only supported
            for the PDF resampling method.
        """
        # Check that upscaling is not set
        if self.upscale_pdf != 1:
            raise ValueError("Upscaling of histograms is not supported for countup method")

        # Get the target number of jets and target PDF values
        num_jets = int(len(jets) * component.sampling_fraction)
        target_pdf = self.target.hist.pbin

        # Get the target histograms
        target_hist = target_pdf * num_jets
        target_hist = (np.floor(target_hist + self.rng.random(target_pdf.shape))).astype(int)

        # Create histogram and bins for the given resampling variables
        _hist, binnumbers = bin_jets(
            array=jets[self.config.vars],
            bins=self.config.flat_bins,
        )
        assert target_pdf.shape == _hist.shape

        # Loop over bins and select relevant jets (indicies)
        all_idx = []
        for bin_id in np.ndindex(*target_hist.shape):
            idx = np.where((bin_id == binnumbers.T).all(axis=-1))[0][: target_hist[bin_id]]
            if len(idx) and len(idx) < target_hist[bin_id]:
                idx = np.concatenate([idx, self.rng.choice(idx, target_hist[bin_id] - len(idx))])
            all_idx.append(idx)
        idx = np.concatenate(all_idx).astype(int)

        # If not enough jets are found, re-use randomly some of them
        if len(idx) < num_jets:
            idx = np.concatenate([idx, self.rng.choice(idx, num_jets - len(idx))])

        # Shuffle the jet indicies
        self.rng.shuffle(idx)

        return idx

    def pdf_select_func(self, jets: dict, component: Component) -> np.ndarray:
        # bin jets
        if self.upscale_pdf > 1:
            bins = [subdivide_bins(bins, self.upscale_pdf) for bins in self.config.flat_bins]
        else:
            bins = self.config.flat_bins

        # Create histogram and bins for the given resampling variables
        _hist, binnumbers = bin_jets(
            array=jets[self.config.vars],
            bins=bins,
        )

        # Esnure correct shape if more than one dimension of bins is used
        if binnumbers.ndim > 1:
            binnumbers = tuple(binnumbers[i] for i in range(len(binnumbers)))

        # importance sample with replacement
        num_samples = int(len(jets) * component.sampling_fraction)

        # Calculate the ratios between the target and the to-be-resampled distribution
        ratios = safe_divide(a=self.target.hist.pbin, b=component.hist.pbin)

        # Upscale the ratios if needed
        if self.upscale_pdf > 1:
            ratios = upscale_array_regionally(
                array=ratios,
                upscaling_factor=self.upscale_pdf,
                num_bins=self.num_bins,
            )

        # Get the probabilities for the resampling from the ratios
        probs = ratios[binnumbers]

        # Select the jets (indicies) for the resampled final output
        idx = random.choices(np.arange(len(jets)), weights=probs, k=num_samples)

        return idx

    def track_upsampling_stats(self, idx: np.ndarray, component: Component) -> None:
        """Tracking the upsampling ratio and update the number of unique jets.

        Parameters
        ----------
        idx : np.ndarray
            Numpy array with the chosen indicies.
        component : Component
            Component instance for a given flavour/class.
        """
        unique, ups_counts = np.unique(idx, return_counts=True)
        component._unique_jets += len(unique)
        max_ups = ups_counts.max()
        component._ups_max = max_ups if max_ups > component._ups_max else component._ups_max

    def sample(
        self,
        components: Components,
        stream: Generator[Any, None, None],
        progress: Progress,
        selected_component: str | None = None,
    ) -> None:
        """Sample the jets by the given selected indicies from the resampling function.

        Parameters
        ----------
        components : Components
            Components instance of the components which are to be resampled.
        stream : Generator[Any, None, None]
            Generator of the jets which are to be resampled.
        progress : Progress
            Progress bar instance for updating the shown progress bar.
        selected_component : str | None, optional
            Compontent name that is to be resampled. By default None and all given
            components are resampled.

        Raises
        ------
        ValueError
            If not enough jets for a given component are present.
        """
        # Loop through input file
        for batch in stream:
            # Loop over the different components
            for component in components:
                # If a selected component is given, skip all components that are not selected
                if selected_component and selected_component != component.name:
                    continue

                # Check if the resampling is already completed for this component
                if component._complete:
                    continue

                # Apply selections
                comp_idx, _ = component.flavour.cuts(batch[self.variables.jets_name])
                if len(comp_idx) == 0:
                    continue

                # Get the batch of jets as a separate dict
                batch_out = select_batch(batch, comp_idx)

                # Apply sampling
                idx = np.arange(len(batch_out[self.variables.jets_name]))

                # Check that the component is not the target and a resampling
                # function is set.
                if component != self.target and self.select_func:
                    # Apply the resampling
                    idx = self.select_func(
                        jets=batch_out[self.variables.jets_name],
                        component=component,
                    )
                    if len(idx) == 0:
                        continue

                    # Get the resampled jets in the dict
                    batch_out = select_batch(batch_out, idx)

                # Check for completion and set to True if completed
                if component.writer.num_written + len(idx) >= component.num_jets:
                    keep = component.num_jets - component.writer.num_written
                    idx = idx[:keep]
                    for name, array in batch_out.items():
                        batch_out[name] = array[:keep]
                    component._complete = True

                # Track upsampling stats
                self.track_upsampling_stats(idx=idx, component=component)

                # Write the resampled jets to file
                component.writer.write(batch_out)

                # Update the progress bar
                progress.update(component.pbar, advance=len(idx))

                # When the component is completed, write metadata and close the writer.
                if component._complete:
                    component._ups_ratio = component.writer.num_written / component._unique_jets
                    component.writer.add_attr("upsampling_ratio", component._ups_ratio)
                    component.writer.add_attr("unique_jets", component._unique_jets)
                    component.writer.add_attr("dsid", str(component.sample.dsid))
                    component.writer.close()

            # Check for completion of all components
            if all(component._complete for component in components):
                break

        # If one component couldn't be completed, raise ValueError
        for component in components:
            if not component._complete:
                raise ValueError(
                    f"Ran out of {component} jets after writing {component.writer.num_written:,}"
                )

    def run_on_region(
        self,
        components: Components,
        region: Region,
        selected_component: str | None = None,
    ) -> None:
        """Run the resampling for a complete region and all components in it.

        Parameters
        ----------
        components : Components
            Components instance of all the components which are to be used.
        region : Region
            Region instance of the region which is to be resampled.
        selected_component : str | None, optional
            Compontent name that is to be resampled. By default None and all given
            components are resampled.

        Raises
        ------
        ValueError
            If the equal_jets flag is not the same for all components.
        """
        # Get the target component
        target = [component for component in components if component.is_target(self.config.target)]
        assert len(target) == 1, "Should have 1 target component per region"
        self.target = target[0]

        # Groupby samples
        grouped_samples = components.groupby_sample()

        for sample, components in grouped_samples:
            # Ensure all components have the same equal_jets flag
            equal_jets_flags = [component.equal_jets for component in components]
            if len(set(equal_jets_flags)) != 1:
                raise ValueError("equal_jets must be the same for all components in a sample")
            equal_jets_flag = equal_jets_flags[0]

            # Get the variables which are to be used
            variables = self.variables.add_jet_vars(components.cuts.variables)

            # Setup the Reader for reading the jets
            reader = H5Reader(
                sample.path,
                self.batch_size,
                jets_name=self.jets_name,
                equal_jets=equal_jets_flag,
                transform=self.transform,
            )

            # Define a stream of jets with the cuts for the region and the variables used
            stream = reader.stream(variables.combined(), reader.num_jets, region.cuts)

            # Run with progress bar
            with ProgressBar() as progress:
                # Setup metadata for each component before resampling
                for component in components:
                    # If a selected component is given, set all not-selected components to ready
                    # to ensure that they are not processed or shown
                    if selected_component and selected_component != component.name:
                        component._complete = True
                        component._ups_max = 0.0
                        component._unique_jets = 0

                    else:
                        component._complete = False
                        component._ups_max = 1.0
                        component._unique_jets = 0

                # Add each component to the progress bar
                for component in components:
                    # If a selected component is given, skip all components that are not selected
                    if selected_component and selected_component != component.name:
                        continue

                    component.pbar = progress.add_task(
                        f"[green]Sampling {component.num_jets:,} jets from {component}...",
                        total=component.num_jets,
                    )

                # Run the actual resampling sampling
                self.sample(
                    components=components,
                    stream=stream,
                    progress=progress,
                    selected_component=selected_component,
                )

        # Print upsampling factors
        for component in components:
            # Log only the selected component or all if not selected component is given
            if (selected_component and component.name == selected_component) or (
                not selected_component
            ):
                log.info(
                    f"{component} usampling ratio is {np.mean(component._ups_ratio):.3f}, with"
                    f" {component.num_jets/np.mean(component._ups_ratio):,.0f}/"
                    f"{component.num_jets:,} unique jets."
                    f" Jets are upsampled at most {np.max(component._ups_max):.0f} times"
                )

    def set_component_sampling_fractions(self, component: Component):
        """Automatically set the sampling fraction for each of the components.

        Parameters
        ----------
        component : Component
            Component for which the sampling fraction is set.

        Raises
        ------
        ValueError
            If the sampling fraction is found to be >1 when the countup method is used.
        """
        # Check that the sampling fraction must be found automatically
        if self.config.sampling_fraction == "auto" or self.config.sampling_fraction is None:
            log.info(
                "[bold green]Sampling fraction will be chosen "
                f"automatically for {component.name}..."
            )

            # Target component always gets one as sampling fraction
            if component.is_target(self.config.target):
                component.sampling_fraction = 1

            else:
                sam_frac = component.get_auto_sampling_fraction(
                    num_jets=component.num_jets,
                    cuts=component.cuts,
                )

                # Raise an error/warning if the sampling fraction is above one
                # for the countup/pdf method
                if sam_frac > 1:
                    if self.config.method == "countup":
                        raise ValueError(
                            f"[bold red]Sampling fraction of {sam_frac:.3f}>1 is"
                            f" needed for component {component} This is not supported for"
                            " countup method."
                        )
                    else:
                        log.warning(
                            f"[bold yellow]sampling fraction of {sam_frac:.3f}>1 is"
                            f" needed for component {component}"
                        )

                # Ensure the sampling fraction is at least above 0.1
                component.sampling_fraction = max(sam_frac, 0.1)

        else:
            # Set the sampling fraction for the component to the value defined
            # in the config
            if component.is_target(self.config.target):
                component.sampling_fraction = 1

            else:
                component.sampling_fraction = self.config.sampling_fraction

    def run(self, region: str | None = None, component: str | None = None):
        """Execute the resampling.

        Parameters
        ----------
        region : str | None, optional
            Define which region is to be resampled, by default None
            which works through the regions sequentially
        component : str | None, optional
            Define which component is to be resampled, by default None
            which works through the regions/components sequentially

        Raises
        ------
        ValueError
            If no region was processed during resampling
        """
        # Check that component is only given together with region
        if component and not region:
            raise ValueError("Can't define component for resampling without region!")

        title = " Running resampling "
        log.info(f"[bold green]{title:-^100}")
        log.info(f"Resampling method: {self.config.method}")

        # Setup the different components and readers/writers and their sampling fraction
        for iter_component in self.components:
            # Check if the component is in the region that is to be processed
            if region and region not in iter_component.name:
                continue

            # Check if the component is either the to-be-resampled component or the target
            if component and (
                component != iter_component.name
                and not iter_component.is_target(self.config.target)
            ):
                continue

            # Setup the reader for the components
            iter_component.setup_reader(
                self.batch_size, jets_name=self.jets_name, transform=self.transform
            )

            # If only one component is run, stop here for the target that needs to be
            # read but not written.
            if component and component != iter_component.name:
                continue

            # Setup the writer for the component
            iter_component.setup_writer(self.variables, jets_name=self.jets_name)

            # Set sampling fraction
            self.set_component_sampling_fractions(component=iter_component)

            # Check that enough jets are available
            log.info(
                "[bold green]Checking requested num_jets based on a sampling fraction of"
                f" {self.config.sampling_fraction}..."
            )
            frac = iter_component.sampling_fraction if self.select_func else 1
            iter_component.check_num_jets(
                iter_component.num_jets,
                sampling_fraction=frac,
                cuts=iter_component.cuts,
            )

        # Create check variable to ensure at least one region was processed
        region_processed = not region

        # Run resampling
        for iter_region, iter_components in self.components.groupby_region():
            # Check if a specific region for resampling was chosen
            if region and region != iter_region.name:
                continue

            log.info(f"[bold green]Running over region {iter_region.name}...")
            self.run_on_region(
                components=iter_components,
                region=iter_region,
                selected_component=component,
            )
            region_processed = True

        # Raise error of no region was processed
        if region_processed is False:
            raise ValueError(
                "No region processed during resampling! Check that you correctly spelled "
                "the region name when running with --region!"
            )

        # Finalise the resampling
        if region:
            unique = 0
            for iter_component in self.components:
                # If a region is given, skip all regions that are not selected
                if region in iter_component.name:
                    # If a component is given, skip all components that are not selected
                    if component and iter_component.name != component:
                        continue
                    unique += iter_component.writer.get_attr("unique_jets")
            log.info(
                f"[bold green]Finished resampling of region {region}. "
                f"A total of {self.components.num_jets:,} jets!"
            )
            log.info(f"[bold green]Estimated unique jets: {unique:,.0f}")
            log.info(f"[bold green]Saved to {self.components.out_dir}/")

        else:
            unique = sum(
                iter_component.writer.get_attr("unique_jets") for iter_component in self.components
            )
            log.info(
                f"[bold green]Finished resampling a total of {self.components.num_jets:,} jets!"
            )
            log.info(f"[bold green]Estimated unique jets: {unique:,.0f}")
            log.info(f"[bold green]Saved to {self.components.out_dir}/")

    def get_num_bins_from_config(self) -> list[list[int]]:
        """Get the lengths of the binning regions in each variable from the config.

        Returns
        -------
        list[list[int]]
            lengths of the binning regions in each variable from the config
        """
        num_bins = []
        for row in self.config.bins.values():
            num_bins.append([sub[-1] for sub in row])
        return num_bins
