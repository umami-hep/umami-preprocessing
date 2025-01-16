from __future__ import annotations

import logging as log
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from ftag import Cuts, Label, Sample
from ftag.hdf5 import H5Reader, H5Writer

from upp.classes.region import Region
from upp.stages.hist import Hist


@dataclass
class Component:
    region: Region
    sample: Sample
    flavour: Label | None
    global_cuts: Cuts
    dirname: Path
    num_jets: int
    num_jets_estimate_available: int
    equal_jets: bool

    def __post_init__(self):
        self.hist = Hist(self.dirname.parent.parent / "hists" / f"hist_{self.name}.h5")

    def setup_reader(
        self,
        batch_size: int,
        jets_name: str = "jets",
        fname: Optional[Path | str | List[Path | str]] = None,
        **kwargs
    ):
        if fname is None:
            fname = list(self.sample.path)

        self.reader = H5Reader(
            fname, batch_size, jets_name=jets_name, equal_jets=self.equal_jets, **kwargs
        )
        log.debug(f"Setup component reader at: {fname}")

    def setup_writer(self, variables, jets_name="jets"):
        dtypes = self.reader.dtypes(variables.combined())
        shapes = self.reader.shapes(self.num_jets, variables.keys())
        self.writer = H5Writer(self.out_path, dtypes, shapes, jets_name=jets_name)
        log.debug(f"Setup component writer at: {self.out_path}")

    @property
    def name(self):
        if self.flavour is None:
            return f"{self.region.name}_{self.sample.name}"
        else:
            return f"{self.region.name}_{self.sample.name}_{self.flavour.name}"

    @property
    def cuts(self):
        if self.flavour is None:
            return self.global_cuts + self.region.cuts
        else:
            return self.global_cuts + self.flavour.cuts + self.region.cuts

    @property
    def out_path(self) -> Path:
        return self.dirname / f"{self.name}.h5"

    def is_target(self, target_str):
        assert self.flavour is not None, (
            "expected is_target to only be called"
            " in resampling code, when self.flavour is expected to be set"
        )
        return self.flavour.name == target_str

    def get_jets(self, variables: list, num_jets: int, cuts: Cuts | None = None):
        jn = self.reader.jets_name
        return self.reader.load({jn: variables}, num_jets, cuts)[jn]

    def check_num_jets(
        self, num_req, sampling_frac=None, cuts=None, silent=False, raise_error=True
    ):
        # Check if num_jets jets are aviailable after the cuts and sampling fraction
        num_est = (
            None if self.num_jets_estimate_available <= 0 else self.num_jets_estimate_available
        )
        total = self.reader.estimate_available_jets(cuts, num_est)
        available = total
        if sampling_frac:
            available = int(total * sampling_frac)

        # check with tolerance to avoid failure midway through preprocessing
        if available < num_req and raise_error:
            raise ValueError(
                f"{num_req:,} jets requested, but only {total:,} are estimated to be"
                f" in {self}. With a sampling fraction of {sampling_frac}, at most"
                f" {available:,} of these are available. You can either reduce the"
                " number of requested jets or increase the sampling fraction."
            )

        if not silent:
            log.debug(f"Sampling fraction {sampling_frac}")
            log.info(
                f"Estimated {available:,} {self} jets available - {num_req:,} requested"
                f"({self.reader.num_jets:,} in {self.sample})"
            )

    def get_auto_sampling_frac(self, num_jets, cuts=None, silent=False):
        num_est = (
            None if self.num_jets_estimate_available <= 0 else self.num_jets_estimate_available
        )
        total = self.reader.estimate_available_jets(cuts, num_est)
        auto_sampling_frac = round(1.1 * num_jets / total, 3)  # 1.1 is a tolerance factor
        if not silent:
            log.debug(f"optimal sampling fraction {auto_sampling_frac:.3f}")
        return auto_sampling_frac

    def __str__(self):
        return self.name

    @property
    def unique_jets(self) -> int:
        return sum([r.get_attr("unique_jets") for r in self.reader.readers])


class Components:
    def __init__(self, components: list[Component]):
        self.components = components

    @classmethod
    def from_config(cls, preprocessing_config: "PreprocessingConfig"):
        components = []
        for components_config in preprocessing_config.config["components"]:
            assert "equal_jets" not in components_config, \
                "equal_jets flag should be set in the sample config"
            region_cuts = (
                Cuts.empty() if preprocessing_config.is_test
                else Cuts.from_list(components_config["region"]["cuts"])
            )
            region = Region(
                components_config["region"]["name"],
                region_cuts + preprocessing_config.global_cuts
            )
            pattern = components_config["sample"]["pattern"]
            equal_jets = components_config["sample"].get("equal_jets", True)
            if isinstance(pattern, list):
                pattern = tuple(pattern)
            sample = Sample(
                pattern=pattern,
                ntuple_dir=preprocessing_config.ntuple_dir,
                name=components_config["sample"]["name"]
            )

            num_jets = components_config["num_jets"]
            if preprocessing_config.split == "val":
                num_jets = components_config.get("num_jets_val", num_jets // 10)
            elif preprocessing_config.split == "test":
                num_jets = components_config.get("num_jets_test", num_jets // 10)

            assert preprocessing_config.num_jets_estimate_available is not None
            if components_config.get("flavours") is None:
                components.append(Component(
                    region,
                    sample,
                    None,
                    preprocessing_config.global_cuts,
                    preprocessing_config.components_dir,
                    num_jets,
                    preprocessing_config.num_jets_estimate_available,
                    equal_jets,
                ))
            else:
                for name in components_config["flavours"]:
                    components.append(Component(
                        region,
                        sample,
                        preprocessing_config.flavour_container[name],
                        preprocessing_config.global_cuts,
                        preprocessing_config.components_dir,
                        num_jets,
                        preprocessing_config.num_jets_estimate_available,
                        equal_jets,
                    ))

        components = cls(components)
        if hasattr(preprocessing_config, "resampling_config") \
            and preprocessing_config.resampling_config.method is not None:
            components.check_flavour_ratios()
        return components

    def check_flavour_ratios(self):
        assert self.flavours is not None, "expected "

        ratios = {}
        for region, components in self.groupby_region():
            this_ratios = {}
            for flavour in self.flavours:
                this_ratios[flavour.name] = (
                    components[flavour].num_jets / components.num_jets
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
    def regions(self):
        return list(set(c.region for c in self))

    @property
    def samples(self):
        return list(set(c.sample for c in self))

    @property
    def flavours(self) -> list[Label] | None:
        if any(c.flavour is None for c in self):
            assert all(c.flavour is None for c in self), \
                "expected to never have mixed components with and without flavours"
            return None
        else:
            # the if is needed to satisfy type checkers, c.flavour should never be None
            # due to the if-statement here
            return list(set(component.flavour for component in self if component.flavour is not None))

    @property
    def cuts(self):
        return sum((c.cuts for c in self), Cuts.from_list([]))

    @property
    def num_jets(self):
        return sum(c.num_jets for c in self)

    @property
    def unique_jets(self):
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
    def dsids(self):
        return list(set(sum([c.sample.dsid for c in self], [])))  # noqa: RUF017

    def groupby_region(self) -> list[tuple[Region, Components]]:
        return [(r, Components([c for c in self if c.region == r])) for r in self.regions]

    def groupby_sample(self):
        return [(s, Components([c for c in self if c.sample == s])) for s in self.samples]

    def __iter__(self):
        yield from self.components

    def __getitem__(self, index_or_label: int | Label):
        if isinstance(index_or_label, int):
            return self.components[index_or_label]
        elif isinstance(index_or_label, Label):
            assert self.flavours is not None, \
                "expected to only index components by label when flavours are available"
            return self.components[self.flavours.index(index_or_label)]
        else:
            assert False

    def __len__(self):
        return len(self.components)
