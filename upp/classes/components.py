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
    num_jets : int
        Number of jets that are to be used from this component
    num_jets_estimate_available : int
        Estimated available jets for this component
    equal_jets : bool
        If the same number of jets should be used from the different samples
    """

    region: Region
    sample: Sample
    flavour: Label
    global_cuts: Cuts
    dirname: Path
    num_jets: int
    num_jets_estimate_available: int
    equal_jets: bool

    def __post_init__(self):
        """Post init setup of internal variables."""
        self.hist = Hist(self.dirname.parent.parent / "hists" / f"hist_{self.name}.h5")
        self._unique_jets = -1
        self._complete = None
        self._ups_ratio = None
        self._ups_max = None
        self.sampling_fraction = None

    def setup_reader(
        self,
        batch_size: int,
        jets_name: str = "jets",
        fname: Path | str | list[Path | str] | None = None,
        **kwargs,
    ) -> None:
        """Set up the reader of the jets to load them from file.

        Parameters
        ----------
        batch_size : int
            Batch size that is used for loading from file
        jets_name : str, optional
            Name of the group in which the jets are stored, by default "jets"
        fname : Path | str | list[Path | str] | None, optional
            Filename of the file(s) from which the jets are loaded, by default None
        **kwargs
            Additional kwargs passed to the H5Reader
        """
        if fname is None:
            fname = self.sample.path

        self.reader = H5Reader(
            fname=fname,
            batch_size=batch_size,
            jets_name=jets_name,
            equal_jets=self.equal_jets,
            **kwargs,
        )
        log.debug(f"Setup component reader at: {fname}")

    def setup_writer(self, variables: VariableConfig, jets_name: str = "jets") -> None:
        """Set up the writer of the jets to file.

        Parameters
        ----------
        variables : VariableConfig
            Instance of VariableConfig in which the variables are stored.
        jets_name : str, optional
            Name of the group in which the jets are stored, by default "jets"
        """
        dtypes = self.reader.dtypes(variables.combined())
        shapes = self.reader.shapes(self.num_jets, variables.keys())
        self.writer = H5Writer(self.out_path, dtypes, shapes, jets_name=jets_name)
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

    def get_jets(self, variables: list, num_jets: int, cuts: Cuts | None = None) -> dict:
        """Load jets from file.

        Parameters
        ----------
        variables : list
            Variables that are to be loaded
        num_jets : int
            Number of jets that are to be loaded
        cuts : Cuts | None, optional
            Cuts instance of the cuts that should be applied on the jets, by default None

        Returns
        -------
        dict
            Dict with the loaded jets
        """
        jn = self.reader.jets_name
        return self.reader.load({jn: variables}, num_jets, cuts)[jn]

    def check_num_jets(
        self,
        num_req: int,
        sampling_fraction: float | None = None,
        cuts: Cuts | None = None,
        silent: bool = False,
        raise_error: bool = True,
    ) -> None:
        """Check the number of available jets.

        If more jets are requested than available, throw an Error.

        Parameters
        ----------
        num_req : int
            Number of requested jets
        sampling_fraction : float | None, optional
            Sampling , by default None
        cuts : Cuts | None, optional
            Cuts instance of the cuts that are to be applied on the jets, by default None
        silent : bool, optional
            Decide, if the debug and info log statements are printed, by default False
        raise_error : bool, optional
            Decide if the error should be raised if not enough jets are available,
            by default True

        Raises
        ------
        ValueError
            If more jets are requsted than available
        """
        # Check if num_jets jets are aviailable after the cuts and sampling fraction
        num_est = (
            None if self.num_jets_estimate_available <= 0 else self.num_jets_estimate_available
        )
        total = self.reader.estimate_available_jets(cuts, num_est)
        available = total
        if sampling_fraction:
            available = int(total * sampling_fraction)

        # check with tolerance to avoid failure midway through preprocessing
        if available < num_req and raise_error:
            raise ValueError(
                f"{num_req:,} jets requested, but only {total:,} are estimated to be"
                f" in {self}. With a sampling fraction of {sampling_fraction}, at most"
                f" {available:,} of these are available. You can either reduce the"
                " number of requested jets or increase the sampling fraction."
            )

        if not silent:
            log.debug(f"Sampling fraction {sampling_fraction}")
            log.info(
                f"Estimated {available:,} {self} jets available - {num_req:,} requested"
                f"({self.reader.num_jets:,} in {self.sample})"
            )

    def get_auto_sampling_fraction(
        self,
        num_jets: int,
        cuts: Cuts | None = None,
        silent: bool = False,
    ) -> float:
        """Estimate the optimal/auto sampling fraction.

        Parameters
        ----------
        num_jets : int
            Number of jets available
        cuts : Cuts | None, optional
            Cuts instance of the cuts that should be applied on the jets, by default None
        silent : bool, optional
            Decide, if the debug and info log statements are printed, by default False

        Returns
        -------
        float
            Automatically estimated sampling fraction
        """
        num_est = (
            None if self.num_jets_estimate_available <= 0 else self.num_jets_estimate_available
        )
        total = self.reader.estimate_available_jets(cuts, num_est)
        auto_sampling_frac = round(1.1 * num_jets / total, 3)  # 1.1 is a tolerance factor
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
    def unique_jets(self) -> int:
        """Return the number of unique jets for this component.

        Returns
        -------
        int
            Number of unique jets for this component
        """
        if self._unique_jets == -1:
            self._unique_jets = sum([r.get_attr("unique_jets") for r in self.reader.readers])

        return self._unique_jets


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
            # Ensure equal_jets flag is correctly set
            assert (
                "equal_jets" not in component
            ), "equal_jets flag should be set in the sample config"

            # Get the region cuts
            region_cuts = (
                Cuts.empty() if config.is_test else Cuts.from_list(component["region"]["cuts"])
            )

            # Get the region and apply the region cuts
            region = Region(component["region"]["name"], region_cuts + config.global_cuts)

            # Load the pattern and the equal_jets settings
            pattern = component["sample"]["pattern"]
            equal_jets = component["sample"].get("equal_jets", True)
            if isinstance(pattern, list):
                pattern = tuple(pattern)

            # Create the Sample instance for the pattern
            sample = Sample(
                pattern=pattern,
                ntuple_dir=config.ntuple_dir,
                name=component["sample"]["name"],
            )

            # Create the Component instances for the different flavours
            for name in component["flavours"]:
                num_jets = component["num_jets"]
                if config.split == "val":
                    num_jets = component.get("num_jets_val", num_jets // 10)
                elif config.split == "test":
                    num_jets = component.get("num_jets_test", num_jets // 10)
                component_list.append(
                    Component(
                        region=region,
                        sample=sample,
                        flavour=config.flavour_cont[name],
                        global_cuts=config.global_cuts,
                        dirname=config.components_dir,
                        num_jets=num_jets,
                        num_jets_estimate_available=config.num_jets_estimate_available,  # type: ignore
                        equal_jets=equal_jets,
                    )
                )
        components = cls(component_list)

        # Check the flavour ratios
        if config.sampl_cfg.method is not None:
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
                this_ratios[f.name] = components[f].num_jets / components.num_jets
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
    def num_jets(self) -> int:
        """Return the number of jets available.

        Returns
        -------
        int
            Number of available jets
        """
        return sum(c.num_jets for c in self)

    @property
    def unique_jets(self) -> int:
        """Return the number of unique jets available.

        Returns
        -------
        int
            Number of available unique jets
        """
        return sum(c.unique_jets for c in self)

    @property
    def out_dir(self):
        out_dir = {c.out_path.parent for c in self}
        assert len(out_dir) == 1
        return next(iter(out_dir))

    @property
    def jet_counts(self):
        num_dict = {
            c.name: {"num_jets": int(c.num_jets), "unique_jets": int(c.unique_jets)} for c in self
        }
        num_dict["total"] = {
            "num_jets": int(self.num_jets),
            "unique_jets": int(self.unique_jets),
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
        if isinstance(index, (str, Label)):
            return self.components[self.flavours.index(index)]

    def __len__(self):
        return len(self.components)
