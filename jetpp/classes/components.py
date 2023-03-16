import logging as log
import math
from dataclasses import dataclass
from pathlib import Path

from ftag import Cuts, Flavour, Flavours, Sample
from ftag.hdf5 import H5Reader, H5Writer

from jetpp.classes.region import Region
from jetpp.stages.hist import Hist


@dataclass
class Component:
    region: Region
    sample: Sample
    flavour: Flavour
    global_cuts: Cuts
    dirname: Path
    num_jets: int
    num_jets_estimate: int

    def __post_init__(self):
        self.hist = Hist(self.dirname.parent.parent / "hists" / f"hist_{self.name}.h5")

    def setup_reader(self, batch_size, fname=None):
        if fname is None:
            fname = self.sample.path
        self.reader = H5Reader(fname, batch_size)
        log.debug(f"Setup component reader at: {fname}")

    def setup_writer(self, variables):
        self.writer = H5Writer(
            self.reader.vds_path, self.out_path, variables.combined(), self.num_jets
        )
        log.debug(f"Setup component writer at: {self.out_path}")

    @property
    def name(self):
        return f"{self.region.name}_{self.sample.name}_{self.flavour.name}"

    @property
    def cuts(self):
        return self.global_cuts + self.flavour.cuts + self.region.cuts

    @property
    def out_path(self):
        return self.dirname / f"{self.name}.h5"

    def is_target(self, target_str):
        return self.flavour.name == target_str

    def get_jets(
        self,
        num_jets: int,
        jet_vars: list | None = None,
        cuts: Cuts | None = None,
        sel: bool = True,
    ):
        if cuts is None:
            cuts = self.cuts
        if jet_vars is None:
            jet_vars = []
        var = {"jets": jet_vars.copy() + cuts.variables}
        return self.reader.load(var, num_jets, cuts if sel else None)["jets"]

    def num_available(self, cuts=None):
        """Return a conservative estimate for the number of available jets."""
        if cuts is None:
            cuts = self.cuts
        all_jets = self.get_jets(self.num_jets_estimate, sel=False)
        sel_jets = cuts(all_jets).values
        estimated_num_jets = len(sel_jets) / len(all_jets) * self.reader.num_jets
        return math.floor(estimated_num_jets / 1000) * 1000

    def check_num_jets(self, num_jets, sampling_frac=None, cuts=None, silent=False):
        total = self.num_available(cuts)
        available = total
        if sampling_frac:
            available = int(total * sampling_frac)

        # check with tolerance to avoid failure midway through preprocessing
        if available < num_jets * 1.01:
            raise ValueError(
                f"{num_jets:,} jets requested, but only {total:,} are estimated to be in"
                f" {self}. With a sampling fraction of {sampling_frac}, at most {available:,} of"
                " these are available. You can either reduce the number of requested jets or"
                " increase the sampling fraction."
            )

        if not silent:
            log.info(f"Estimated {available:,} {self} jets available - {num_jets:,} requested")

    def __str__(self):
        return self.name


class Components:
    def __init__(self, components: list[Component]):
        self.components = components

    @classmethod
    def from_config(cls, pp_cfg):
        components = []
        for c in pp_cfg.config["components"]:
            region_cuts = Cuts.empty() if pp_cfg.is_test else Cuts.from_list(c["region"]["cuts"])
            region = Region(c["region"]["name"], region_cuts + pp_cfg.global_cuts)
            pattern = c["sample"]["pattern"]
            if isinstance(pattern, list):
                pattern = tuple(pattern)
            sample = Sample(ntuple_dir=pp_cfg.ntuple_dir, name=c["sample"]["name"], pattern=pattern)
            for name in c["flavours"]:
                flavour = Flavours[name]
                num_jets = c["num_jets"]
                if pp_cfg.split == "val":
                    num_jets = num_jets // 10
                elif pp_cfg.split == "test":
                    num_jets = c.get("num_jets_test", num_jets // 10)
                components.append(
                    Component(
                        region,
                        sample,
                        flavour,
                        pp_cfg.global_cuts,
                        pp_cfg.components_dir,
                        num_jets,
                        pp_cfg.num_jets_estimate,
                    )
                )
        return cls(components)

    @property
    def regions(self):
        return list(dict.fromkeys(c.region for c in self))

    @property
    def samples(self):
        return list(dict.fromkeys(c.sample for c in self))

    @property
    def flavours(self):
        return list(dict.fromkeys(c.flavour for c in self))

    @property
    def cuts(self):
        return sum((c.cuts for c in self), Cuts.from_list([]))

    @property
    def num_jets(self):
        return sum(c.num_jets for c in self)

    @property
    def out_dir(self):
        out_dir = {c.out_path.parent for c in self}
        assert len(out_dir) == 1
        return list(out_dir)[0]

    def groupby_region(self):
        return [(r, Components([c for c in self if c.region == r])) for r in self.regions]

    def groupby_sample(self):
        return [(s, Components([c for c in self if c.sample == s])) for s in self.samples]

    def __iter__(self):
        yield from self.components

    def __getitem__(self, index):
        return self.components[index]

    def __len__(self):
        return len(self.components)
