from __future__ import annotations

import logging as log
from pathlib import Path

import yaml
from yamlinclude import YamlIncludeConstructor

from jetpp.classes.components import Components
from jetpp.classes.misc import Cuts
from jetpp.classes.resampling_config import ResamplingConfig
from jetpp.classes.variable_config import VariableConfig
from jetpp.utils import path_append

# support inclusion of yaml files in the config dir
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.SafeLoader, base_dir=Path(__file__).parent.parent / "configs"
)


class PreprocessingConfig:
    def __init__(self, config_path: Path | str, out_type: str):
        self.config_path = Path(config_path)
        with open(config_path) as file:
            config = yaml.safe_load(file)
            gc = config["global"]

        self.config = config
        self.out_type = out_type

        self.sampl_cfg = ResamplingConfig(self.config)

        self.base_dir = Path(gc["base_dir"])
        self.ntuple_dir = self.get_path(gc.get("ntuple_dir", "ntuples"))
        self.vds_dir = self.get_path(gc.get("vds_dir", "vds"))
        self.components_dir = self.get_path(gc.get("components_dir", "components")) / self.out_type
        self.out_dir = self.get_path(gc.get("out_dir", "output"))
        out_fname = self.out_dir / gc["out_fname"]
        self.out_fname = path_append(out_fname, self.out_type)

        assert self.ntuple_dir.exists(), self.ntuple_dir

        self.batch_size = gc["batch_size"]
        self.num_jets_estimate = gc["num_jets_estimate"]
        self.flavours = config["flavours"]
        self.merge_test_samples = gc.get("merge_test_samples", False)

        # apply selections from resampling bin min and max edges
        cuts_list = config["global_cuts"].get("common", []) + config["global_cuts"][self.out_type]
        for resampling_var, cfg in config["resampling"]["variables"].items():
            cuts_list.append([resampling_var, ">", cfg["bins"][0][0]])
            cuts_list.append([resampling_var, "<", cfg["bins"][-1][1]])
        self.global_cuts = Cuts.from_list(cuts_list)

        self.components = Components.from_config(config["components"], self)
        self.variables = VariableConfig(config["variables"], gc["jets_name"], self.is_test)

        # reduce number of jets for non-train pipeline by a factor of ten
        # TODO: improve this
        if self.out_type != "train":
            for c in self.components:
                c.num_jets = int(c.num_jets * 0.1)

    @property
    def is_test(self):
        return self.out_type == "test"

    def get_path(self, path: Path | str):
        """Creates an absolute path from an absolute path or relative path and
        base_dir."""
        path = Path(path)
        if path.is_absolute():
            return path
        return (self.base_dir / path).absolute()

    def copy_configs(self):
        copy_config_path = self.out_dir / self.config_path.name
        copy_config_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Copying config to {copy_config_path}")
        with open(copy_config_path, "w") as file:
            yaml.dump(self.config, file, sort_keys=False)
