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
    def __init__(self, config_path: Path | str, split: str):
        self.config_path = Path(config_path)
        with open(config_path) as file:
            config = yaml.safe_load(file)
            gc = config["global"]

        self.config = config
        self.split = split
        self.sampl_cfg = ResamplingConfig(self.config)

        # configure paths
        self.base_dir = Path(gc["base_dir"])
        self.ntuple_dir = self.get_path(gc.get("ntuple_dir", "ntuples"))
        self.vds_dir = self.get_path(gc.get("vds_dir", "vds"))
        self.components_dir = self.get_path(gc.get("components_dir", "components")) / self.split
        self.out_dir = self.get_path(gc.get("out_dir", "output"))
        out_fname = self.out_dir / gc.get("out_fname", "pp_output.h5")
        self.out_fname = path_append(out_fname, self.split)
        assert self.ntuple_dir.exists(), f"{self.ntuple_dir} does not exist"

        # read global config
        self.flavours = config["flavours"]
        self.batch_size = gc["batch_size"]
        self.num_jets_estimate = gc["num_jets_estimate"]
        self.merge_test_samples = gc.get("merge_test_samples", False)

        # get cuts
        cuts_list = config["global_cuts"].get("common", []) + config["global_cuts"][self.split]
        if not self.is_test:
            for resampling_var, cfg in config["resampling"]["variables"].items():
                cuts_list.append([resampling_var, ">", cfg["bins"][0][0]])
                cuts_list.append([resampling_var, "<", cfg["bins"][-1][1]])
        self.global_cuts = Cuts.from_list(cuts_list)

        # load components and variables
        self.components = Components.from_config(self)
        self.variables = VariableConfig(
            config["variables"], gc.get("jets_name", "jets"), self.is_test
        )

    @property
    def is_test(self):
        return self.split == "test"

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
