from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jetpp.classes.cuts import Cuts


@dataclass(frozen=True)
class Region:
    name: str
    cuts: Cuts

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.cuts[0].value < other.cuts[0].value

    def __eq__(self, other):
        return self.name == other.name


@dataclass(frozen=True)
class Sample:
    ntuple_dir: Path
    vds_dir: Path
    name: str
    pattern: str

    @property
    def path(self) -> Path:
        return self.ntuple_dir / self.pattern

    @property
    def vds_path(self) -> Path:
        return self.vds_dir / f"combined_{self.name}_vds.h5"

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name


@dataclass(frozen=True)
class Flavour:
    name: str
    cuts: Cuts

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name
