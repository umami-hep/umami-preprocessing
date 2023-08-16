from __future__ import annotations

import dataclasses
import functools
import logging as log
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from subprocess import check_output
from typing import Literal

import yaml
from ftag import Cuts
from ftag.transform import Transform
from yamlinclude import YamlIncludeConstructor

from upp.classes.components import Components
from upp.classes.resampling_config import ResamplingConfig
from upp.classes.variable_config import VariableConfig
from upp.utils import path_append

# support inclusion of yaml files in the config dir
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.SafeLoader, base_dir=Path(__file__).parent.parent / "configs"
)


Split = Literal["train", "val", "test"]


@dataclass
class PreprocessingConfig:
    config_path: Path
    split: Split
    config: dict
    base_dir: Path
    ntuple_dir: Path = Path("ntuples")
    components_dir: Path = Path("components")
    out_dir: Path = Path("output")
    out_fname: Path = Path("pp_output.h5")
    batch_size: int = 100_000
    num_jets_estimate: int = 1_000_000
    merge_test_samples: bool = False
    jets_name: str = "jets"

    def __post_init__(self):
        # postprocess paths
        for field in dataclasses.fields(self):
            if field.type == "Path" and field.name != "out_fname":
                setattr(self, field.name, self.get_path(Path(getattr(self, field.name))))
        if not self.ntuple_dir.exists():
            raise FileNotFoundError(f"Path {self.ntuple_dir} does not exist")
        self.components_dir = self.components_dir / self.split
        self.out_fname = self.out_dir / path_append(self.out_fname, self.split)

        # configure classes
        sampl_cfg = copy(self.config["resampling"])
        self.sampl_cfg = ResamplingConfig(sampl_cfg.pop("variables"), **sampl_cfg)
        self.components = Components.from_config(self)
        self.variables = VariableConfig(self.config["variables"], self.jets_name, self.is_test)
        self.variables = self.variables.add_jet_vars(
            list(self.config["resampling"]["variables"].keys()), "labels"
        )
        self.transform = (
            Transform(**self.config["transform"]) if "transform" in self.config else None
        )

        # copy config
        git_hash = check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent)
        self.git_hash = git_hash.decode("ascii").strip()
        self.config["pp_git_hash"] = self.git_hash
        self.copy_config()

    @classmethod
    def from_file(cls, config_path: Path, split: Split):
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} does not exist - check your --config arg")
        with open(config_path) as file:
            config = yaml.safe_load(file)
            return cls(config_path, split, config, **config["global"])

    def get_path(self, path: Path):
        return path if path.is_absolute() else (self.base_dir / path).absolute()

    @property
    def is_test(self):
        return self.split == "test"

    @functools.cached_property
    def global_cuts(self):
        cuts_list = self.config["global_cuts"].get("common", [])
        cuts_list += self.config["global_cuts"][self.split]
        if not self.is_test:
            for resampling_var, cfg in self.config["resampling"]["variables"].items():
                cuts_list.append([resampling_var, ">", cfg["bins"][0][0]])
                cuts_list.append([resampling_var, "<", cfg["bins"][-1][1]])
        return Cuts.from_list(cuts_list)

    def copy_config(self):
        copy_config_path = self.out_dir / path_append(Path(self.config_path.name), self.split)
        log.info(f"Copying config to {copy_config_path}")
        copy_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(copy_config_path, "w") as file:
            yaml.dump(self.config, file, sort_keys=False)
