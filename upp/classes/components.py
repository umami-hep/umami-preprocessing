from __future__ import annotations

import logging as log
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ftag import Cuts, Label, Sample
from ftag.hdf5 import H5Reader, H5Writer

from upp.classes.region import Region
from upp.stages.hist import Hist

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.preprocessing_config import PreprocessingConfig
    from upp.classes.variable_config import VariableConfig


@dataclass
class Component:
    """
    Component class for the different components/flavours.

    It stores the needed information about the component and
    allow for certain features in terms of resampling.

    Attributes
    ----------
    region : Region
        Region instance of the region for which this instance is setup
    sample : Sample
        Sample instance of the sample for which this instance is setup
    flavour : Label
        Flavour for which this instance is setup
    global_cuts : Cuts
        Global cuts that should be applied for this component
    dirname : Path
        Directory of where this component is/will be stored
    num_global_objects : int
        Number of objects that are to be used from this component
    num_global_objects_estimate_available : int
        Estimated available objects for this component
    equal_global_objects : bool
        If the same number of objects should be used from the different samples
    """

    region: Region
    sample: Sample
    flavour: Label
    global_cuts: Cuts
    dirname: Path
    num_global_objects: int
    num_global_objects_estimate_available: int
    equal_global_objects: bool

    def __post_init__(self):
        """Post init setup of internal variables."""
        self.hist = Hist(self.dirname.parent.parent / "hists" / f"hist_{self.name}.h5")
        self._unique_global_objects = -1
        self._complete = None
        self._ups_ratio = None
        self._ups_max = None
        self.sampling_fraction = None

    def setup_reader(
        self,
        batch_size: int,
        global_name: str = "jets",
        fname: Path | str | list[Path | str] | None = None,
        **kwargs,
    ) -> None:
        """Set up the reader of the objects to load them from file.

        Parameters
        ----------
        batch_size : int
            Batch size that is used for loading from file
        global_name : str, optional
            Name of the group in which the objects are stored, by default "jets"
        fname : Path | str | list[Path | str] | None, optional
            Filename of the file(s) from which the objects are loaded, by default None
        **kwargs
            Additional kwargs passed to the H5Reader
        """
        if fname is None:
            fname = self.sample.path

        if "vds_dir" not in kwargs and self.sample.vds_dir is not None:
            kwargs["vds_dir"] = self.sample.vds_dir

        self.reader = H5Reader(
            fname=fname,
            batch_size=batch_size,
            global_objects_name=global_name,
            equal_global_objects=self.equal_global_objects,
            **kwargs,
        )
        log.debug(f"Setup component reader at: {fname}")

    def setup_writer(self, variables: VariableConfig, global_name: str = "jets") -> None:
        """Set up the writer of the objects to file.

        Parameters
        ----------
        variables : VariableConfig
            Instance of VariableConfig in which the variables are stored.
        global_name : str, optional
            Name of the group in which the objects are stored, by default "jets"
        """
        dtypes = self.reader.dtypes(variables.combined())
        # num_global_objects == -1 ("write all") -> 0 leading dim so the writer grows dynamically
        shapes = self.reader.shapes(max(self.num_global_objects, 0), variables.keys())
        self.writer = H5Writer(self.out_path, dtypes, shapes, global_objects_name=global_name)
        log.debug(f"Setup component writer at: {self.out_path}")

    @property
    def name(self) -> str:
        """Return the name of this component.

        Returns
        -------
        str
            Name of the component
        """
        return f"{self.region.name}_{self.sample.name}_{self.flavour.name}"

    @property
    def cuts(self) -> Cuts:
        """Return all cuts that are applied for this component.

        Returns
        -------
        Cuts
            Cuts instance of all the cuts that are applied on the component
        """
        return self.global_cuts + self.flavour.cuts + self.region.cuts

    @property
    def out_path(self) -> Path:
        """Return the output file path.

        Returns
        -------
        Path
            Output file psth
        """
        return self.dirname / f"{self.name}.h5"

    def is_target(self, target_str: str) -> bool:
        """Check if the component is the target component for resampling.

        Parameters
        ----------
        target_str : str
            Target string to check against.

        Returns
        -------
        bool
            If the component is a target or not.
        """
        return self.flavour.name == target_str

    def get_global_objects(
        self, variables: list, num_global_objects: int, cuts: Cuts | None = None
    ) -> dict:
        """Load objects from file.

        Parameters
        ----------
        variables : list
            Variables that are to be loaded
        num_global_objects : int
            Number of objects that are to be loaded
        cuts : Cuts | None, optional
            Cuts instance of the cuts that should be applied on the objects, by default None

        Returns
        -------
        dict
            Dict with the loaded objects
        """
        jn = self.reader.global_objects_name
        return self.reader.load({jn: variables}, num_global_objects, cuts)[jn]

    def check_num_global_objects(
        self,
        num_req: int,
        sampling_fraction: float | None = None,
        cuts: Cuts | None = None,
        silent: bool = False,
        raise_error: bool = True,
    ) -> None:
        """Check the number of available objects.

        If more objects are requested than available, throw an Error.

        Parameters
        ----------
        num_req : int
            Number of requested objects
        sampling_fraction : float | None, optional
            Sampling , by default None
        cuts : Cuts | None, optional
            Cuts instance of the cuts that are to be applied on the objects, by default None
        silent : bool, optional
            Decide, if the debug and info log statements are printed, by default False
        raise_error : bool, optional
            Decide if the error should be raised if not enough objects are available,
            by default True

        Raises
        ------
        ValueError
            If more objects are requsted than available
        """
        # num_req < 0 means "use all available objects" - nothing to check
        if num_req < 0:
            return

        # Check if num_global_objects objects are aviailable after the cuts and sampling fraction
        num_est = (
            None
            if self.num_global_objects_estimate_available <= 0
            else self.num_global_objects_estimate_available
        )
        total = self.reader.estimate_available_global_objects(cuts, num_est)
        available = total
        if sampling_fraction:
            available = int(total * sampling_fraction)

        # check with tolerance to avoid failure midway through preprocessing
        if available < num_req and raise_error:
            raise ValueError(
                f"{num_req:,} objects requested, but only {total:,} are estimated to be"
                f" in {self}. With a sampling fraction of {sampling_fraction}, at most"
                f" {available:,} of these are available. You can either reduce the"
                " number of requested objects or increase the sampling fraction."
            )

        if not silent:
            log.debug(f"Sampling fraction {sampling_fraction}")
            log.info(
                f"Estimated {available:,} {self} objects available - {num_req:,} requested"
                f"({self.reader.num_global_objects:,} in {self.sample})"
            )

    def get_auto_sampling_fraction(
        self,
        num_global_objects: int,
        cuts: Cuts | None = None,
        silent: bool = False,
    ) -> float:
        """Estimate the optimal/auto sampling fraction.

        Parameters
        ----------
        num_global_objects : int
            Number of objects available
        cuts : Cuts | None, optional
            Cuts instance of the cuts that should be applied on the objects, by default None
        silent : bool, optional
            Decide, if the debug and info log statements are printed, by default False

        Returns
        -------
        float
            Automatically estimated sampling fraction
        """
        num_est = (
            None
            if self.num_global_objects_estimate_available <= 0
            else self.num_global_objects_estimate_available
        )
        total = self.reader.estimate_available_global_objects(cuts, num_est)
        auto_sampling_frac = round(1.1 * num_global_objects / total, 3)  # 1.1 is a tolerance factor
        if not silent:
            log.debug(f"optimal sampling fraction {auto_sampling_frac:.3f}")
        return auto_sampling_frac

    def __str__(self) -> str:
        """Return internal name of the component instance.

        Returns
        -------
        str
            Internal name of the component instance
        """
        return self.name

    @property
    def unique_global_objects(self) -> int:
        """Return the number of unique objects for this component.

        Returns
        -------
        int
            Number of unique objects for this component
        """
        if self._unique_global_objects == -1:
            attr = f"unique_{self.reader.global_objects_name}"
            self._unique_global_objects = sum([r.get_attr(attr) for r in self.reader.readers])

        return self._unique_global_objects


class Components:
    """Components class to store and manage multiple Component instances."""

    def __init__(self, components: Components | list):
        self.components = components

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> Components:
        """Create Components instance from PreprocessingConfig instance.

        Parameters
        ----------
        config : PreprocessingConfig
            PreprocessingConfig instance with the loaded config file.

        Returns
        -------
        Components
            Components instance created from the PreprocessingConfig
        """
        component_list = []
        for component in config.config["components"]:
            # Ensure equal_global_objects flag is correctly set
            assert "equal_global_objects" not in component, (
                "equal_global_objects flag should be set in the sample config"
            )

            # Get the region cuts
            region_cuts = (
                Cuts.empty() if config.is_test else Cuts.from_list(component["region"]["cuts"])
            )

            # Get the region and apply the region cuts
            region = Region(component["region"]["name"], region_cuts + config.global_cuts)

            # Load the pattern and the equal_global_objects settings
            pattern = component["sample"]["pattern"]
            equal_global_objects = component["sample"].get("equal_global_objects", True)
            if isinstance(pattern, list):
                pattern = tuple(pattern)

            # Create the Sample instance for the pattern
            sample = Sample(
                pattern=pattern,
                ntuple_dir=config.ntuple_dir,
                name=component["sample"]["name"],
                skip_checks=config.skip_checks,
                vds_dir=config.vds_dir,
            )

            # Create the Component instances for the different flavours
            for name in component["flavours"]:
                num_global_objects = component["num_global_objects"]
                if config.split == "val":
                    num_global_objects = component.get(
                        "num_global_objects_val", num_global_objects // 10
                    )
                elif config.split == "test":
                    num_global_objects = component.get(
                        "num_global_objects_test", num_global_objects // 10
                    )
                component_list.append(
                    Component(
                        region=region,
                        sample=sample,
                        flavour=config.flavour_cont[name],
                        global_cuts=config.global_cuts,
                        dirname=config.components_dir,
                        num_global_objects=num_global_objects,
                        num_global_objects_estimate_available=config.num_global_objects_estimate_available,  # type: ignore
                        equal_global_objects=equal_global_objects,
                    )
                )
        components = cls(component_list)

        # Check the flavour ratios (not meaningful when resampling is skipped)
        if not config.skip_resampling:
            components.check_flavour_ratios()

        return components

    def check_flavour_ratios(self) -> None:
        """Check if the flavour ratios match.

        Raises
        ------
        ValueError
            If inconsistent flavour ratios are found
        """
        ratios = {}
        flavours = self.flavours
        for region, components in self.groupby_region():
            this_ratios = {}
            for f in flavours:
                this_ratios[f.name] = (
                    components[f].num_global_objects / components.num_global_objects
                )
            ratios[region] = this_ratios

        ref = next(iter(ratios.values()))
        ref_region = next(iter(ratios.keys()))
        for i, (region, ratio) in enumerate(ratios.items()):
            if i != 0 and not np.allclose(list(ratio.values()), list(ref.values())):
                raise ValueError(
                    f"Found inconsistent flavour ratios: \n - {ref_region}: {ref} \n -"
                    f" {region}: {ratio}"
                )

    @property
    def regions(self) -> list[str]:
        """Return the regions used.

        Returns
        -------
        list[str]
            List of regions
        """
        return list(dict.fromkeys(c.region for c in self))

    @property
    def samples(self) -> list[str]:
        """Return the samples used.

        Returns
        -------
        list[str]
            List of samples
        """
        return list(dict.fromkeys(c.sample for c in self))

    @property
    def flavours(self) -> list[Label]:
        """Return the flavours used.

        Returns
        -------
        list[Label]
            List of flavours
        """
        return list(dict.fromkeys(c.flavour for c in self))

    @property
    def cuts(self) -> Cuts:
        """Return the cuts that are applied.

        Returns
        -------
        Cuts
            Cuts object with all cuts
        """
        return sum((c.cuts for c in self), Cuts.from_list([]))

    @property
    def num_global_objects(self) -> int:
        """Return the number of objects available.

        Returns
        -------
        int
            Number of available objects
        """
        return sum(c.num_global_objects for c in self)

    @property
    def unique_global_objects(self) -> int:
        """Return the number of unique objects available.

        Returns
        -------
        int
            Number of available unique objects
        """
        return sum(c.unique_global_objects for c in self)

    @property
    def out_dir(self):
        out_dir = {c.out_path.parent for c in self}
        assert len(out_dir) == 1
        return next(iter(out_dir))

    def global_object_counts(self, global_name: str = "jets") -> dict:
        """Return per-component and total object counts.

        Parameters
        ----------
        global_name : str, optional
            Name of the global object, used to key the counts, by default "jets".

        Returns
        -------
        dict
            Counts keyed by ``num_{global_name}`` and ``unique_{global_name}``.
        """
        num_dict = {
            c.name: {
                f"num_{global_name}": int(c.num_global_objects),
                f"unique_{global_name}": int(c.unique_global_objects),
            }
            for c in self
        }
        num_dict["total"] = {
            f"num_{global_name}": int(self.num_global_objects),
            f"unique_{global_name}": int(self.unique_global_objects),
        }
        return num_dict

    @property
    def dsids(self) -> list[str]:
        """Return the DSIDs used.

        Returns
        -------
        list[str]
            List of used DSIDs
        """
        return list(set(sum([c.sample.dsid for c in self], [])))  # noqa: RUF017

    def groupby_region(self) -> list[tuple]:
        """Return the components grouped by region.

        Returns
        -------
        list[tuple]
            List of tuples in the form of (Region, Component)
        """
        return [(r, Components([c for c in self if c.region == r])) for r in self.regions]

    def groupby_sample(self) -> list[tuple]:
        """Return the components grouped by sample.

        Returns
        -------
        list[tuple]
            List of tuples in the form of (Sample, Component)
        """
        return [(s, Components([c for c in self if c.sample == s])) for s in self.samples]

    def __iter__(self):
        yield from self.components

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.components[index]
        if isinstance(index, str | Label):
            return self.components[self.flavours.index(index)]

    def __len__(self):
        return len(self.components)
