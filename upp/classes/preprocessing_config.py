from __future__ import annotations

import dataclasses
import functools
import logging as log
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotmap import DotMap
from ftag import Cuts, Extended_Flavours, Flavours
from ftag.git_check import get_git_hash
from ftag.labels import LabelContainer
from ftag.track_selector import TrackSelector
from ftag.transform import Transform
from yamlinclude import YamlIncludeConstructor

from upp import __version__
from upp.classes.components import Components
from upp.classes.resampling_config import ResamplingConfig
from upp.classes.variable_config import VariableConfig
from upp.utils.tools import path_append

# support inclusion of yaml files in the config dir
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.SafeLoader, base_dir=Path(__file__).parent.parent / "configs"
)


Split = Literal["train", "val", "test"]


@dataclass
class PreprocessingConfig:
    """
    Global options for the preprocessing.

    These options are specified in the config file
    under the `global:` key. They are passed as kwargs to PreprocessingConfig.
    The config file is also copied to the output directory.

    For example:
    ```yaml
    global:
        jets_name: jets
        batch_size: 1_000_000
        num_jets_estimate: 5_000_000
        base_dir: /my/stuff/
        ntuple_dir: h5-inputs # resolved path: /my/stuff/h5-inputs/
    ```

    Attributes
    ----------
    config_path : Path
        Path to the config yaml file that is used. Does not need to be set in config.
    split : Split
        For which part the preprocessing is run. Either train, val or test. This needs
        to be set as a command line argument when running the programm. Does not need
        to be set in config.
    config : dict
        Dict with the loaded config. Does not need to be set in config.
    base_dir : Path
        Base directory for all other paths.
    ntuple_dir : Path, optional
        Directory containing the input h5 ntuples. If a relative path is given, it is
        interpreted as relative to base_dir. By default Path("ntuples")
    components_dir : Path, optional
        Directory for intermediate component files. If a relative path is given, it is
        interpreted as relative to base_dir. By default Path("components")
    out_dir : Path, optional
        Directory for output files. If a relative path is given, it is interpreted as
        relative to base_dir. By default Path("output")
    out_fname : Path, optional
        Filename stem for the output files. By default Path("pp_output.h5")
    batch_size : int, optional
        Batch size for the preprocessing. For each batch select
        `sampling_fraction*batch_size_after_cuts`. It is recommended to choose high batch sizes
        especially to the `countup` method to achive best agreement of target and resampled
        distributions. By default 100_000
    num_jets_estimate : int, optional
        Any of the further three arguments that are not specified will default to this value
        Is equal to 1_000_000 by default.
    num_jets_estimate_available : int | None, optional
        A sabsample taken from the whole sample to estimate the number of jets after the cuts.
        Please keep this number high in order to not get poisson error of more then 5%.
        If time allows you can use -1 to get a precise number of jets and not just an estimate
        although it will be slow for large datasets. Is equal to num_jets_estimate by default.
    num_jets_estimate_hist : int | None, optional
        Number of jets of each flavour that are used to construct histograms for probability
        density function estimation. Larger numbers give a better quality estmate of the pdfs.
        Is equal to num_jets_estimate by default.
    num_jets_estimate_norm : int | None, optional
        Number of jets of each flavour that are used to estimate shifting and scaling during
        normalisation step. Larger numbers give a better quality estmates.
        Is equal to num_jets_estimate by default.
    num_jets_estimate_plotting : int | None, optional
        Number of jets of each flavour used for plotting the initial and the final resampling
        variable distributions. Larger numbers give a better estimate of the full distributions.
        Is equal to num_jets_estimate by default.
    merge_test_samples : bool, optional
        Merge the test samples of the different processes into one file. By default False.
    jets_name : str, optional
        Name of the jets dataset in the input file. By default "jets".
    flavour_config : Path | None, optional
        Flavour config yaml file which is to be used. By default None
    flavour_category : str, optional
        Flavour categories that are to be used. By default, the "standard" (non-extended)
        labels are loaded. The extended labels can be used by setting this value to "extended".
        By default "standard". To use this option, flavour_config must be None.
    num_jets_per_output_file : int | None, optional
        Number of jets per final output file. If the number of total jets is larger
        than this number, the final h5 output files are splitted in multiple smaller
        files with this number of jets per file. By default None which produces one
        huge output file.
    skip_config_copy : bool, optional
        Decide, if the config copying is skipped or not. By default False
    """

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
    num_jets_estimate_available: int | None = None
    num_jets_estimate_hist: int | None = None
    num_jets_estimate_norm: int | None = None
    num_jets_estimate_plotting: int | None = None
    merge_test_samples: bool = False
    jets_name: str = "jets"
    flavour_config: Path | None = None
    flavour_category: str = "standard"
    num_jets_per_output_file: int | None = None
    skip_config_copy: bool = False

    def __post_init__(self):
        # postprocess paths
        if self.num_jets_estimate:
            if self.num_jets_estimate_available is None:
                self.num_jets_estimate_available = max(self.num_jets_estimate, int(1e6))
            if self.num_jets_estimate_hist is None:
                self.num_jets_estimate_hist = self.num_jets_estimate
            if self.num_jets_estimate_norm is None:
                self.num_jets_estimate_norm = self.num_jets_estimate
            if self.num_jets_estimate_plotting is None:
                self.num_jets_estimate_plotting = self.num_jets_estimate

        for field in dataclasses.fields(self):
            if field.type == "Path" and field.name != "out_fname" and field.name != "base_dir":
                setattr(self, field.name, self.get_path(Path(getattr(self, field.name))))
        if not self.ntuple_dir.exists():
            raise FileNotFoundError(f"Path {self.ntuple_dir} does not exist")
        self.components_dir = self.components_dir / self.split
        self.out_fname = self.out_dir / path_append(self.out_fname, self.split)

        # Define the content of the flavour label container
        if self.flavour_config:
            self.flavour_cont = LabelContainer.from_yaml(
                yaml_path=self.flavour_config,
            )

        elif self.flavour_category == "standard":
            self.flavour_cont = Flavours

        elif self.flavour_category == "extended":
            self.flavour_cont = Extended_Flavours

        else:
            raise ValueError(
                f"flavour_category {self.flavour_category} is not supported in the default "
                "flavours! If you want to use your own flavour config yaml file, please "
                "provide flavour_config!"
            )

        # configure classes
        sampl_cfg = copy(self.config["resampling"])
        if self.is_test:
            sampl_cfg["method"] = None
        self.sampl_cfg = ResamplingConfig(**sampl_cfg)
        self.components = Components.from_config(self)

        # get track selectors
        vc = self.config["variables"]
        selectors = {}
        for name, groups in vc.items():
            if selection := groups.get("selection", None):
                selectors[name] = TrackSelector(Cuts.from_list(selection))

        # configure variables
        self.variables = VariableConfig(
            self.config["variables"], self.jets_name, self.is_test, selectors
        )
        self.variables = self.variables.add_jet_vars(
            list(self.config["resampling"]["variables"].keys()), "labels"
        )
        self.transform = (
            Transform(**self.config["transform"]) if "transform" in self.config else None
        )

        # reproducibility
        self.git_hash = get_git_hash(Path(__file__).parent)
        if self.git_hash is None:
            self.git_hash = __version__
        self.config["upp_hash"] = self.git_hash

        # copy config
        if not self.skip_config_copy:
            self.copy_config()

    @classmethod
    def from_file(cls, config_path: Path, split: Split, skip_config_copy: bool = False):
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} does not exist - check your --config arg")
        with open(config_path) as file:
            config = yaml.safe_load(file)
            return cls(
                config_path=config_path,
                split=split,
                config=config,
                skip_config_copy=skip_config_copy,
                **config["global"],
            )

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

    def copy_config(self, suffix: str | None = None, out_dir: str | Path | None = None) -> None:
        """Copy the configuration file to a new location with an optional suffix and out directory.

        Parameters
        ----------
        suffix : str | None, optional
            A suffix to append to the configuration file name. If None, the current
            `self.split` value will be used as the suffix (default is None).

        out_dir : str | Path | None, optional
            The output directory where the copied configuration file will be saved.
            If None, the current `self.out_dir` value will be used as the output directory
            (default is None).
        """
        if suffix is None:
            suffix = self.split
        if out_dir is None:
            out_dir = self.out_dir
        copy_config_path = out_dir / path_append(Path(self.config_path.name), suffix)
        log.info(f"Copying config to {copy_config_path}")
        copy_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(copy_config_path, "w") as file:
            yaml.dump(self.config, file, sort_keys=False)

    def get_umami_general(self) -> DotMap:
        """
        Return the arguments to be fed into GeneralSettings class in umami.

        Notes
        -----
        This function provides a workaround to avoid importing 'umami' in 'upp'.
        Instead, all 'umami' class initialization is
        performed in 'umami' code, and the resulting objects are passed to 'upp' if needed.

        Returns
        -------
        DotMap
            An instance of 'umami.preprocessing_tools.configuration.GeneralSettings'
            class configured with the
            necessary parameters.
        """
        self.config["umami"]["general"].update({"outfile_name": str(self.out_fname)})
        return DotMap(
            self.config["umami"]["general"],
            _dynamic=False,
        )

    def mimic_umami_config(self, general: Any) -> PreprocessingConfig:
        """Make the config mimic the umami config structure and behaviour.

        Parameters
        ----------
        general : Any
            first initialised in umami.preprocessing_tools.configuration.Configuration
            class in umami using get_umami_general() for arguments
            then feed it into mimic_umami_config() to get the rest of the config
            mimiking the umami config structure and behaviour

        Returns
        -------
        PreprocessingConfig
            Mimicing PreprocessingConfig instance
        """
        self.general = general
        self.sampling = DotMap(self.config["umami"]["sampling"], _dynamic=False)
        self.sampling.class_labels = [flav.name for flav in self.components.flavours]
        if self.config["umami"].get("parameters", None) is not None:
            self.parameters = self.config["umami"]["parameters"]
        else:
            self.parameters = {}
            self.parameters[""] = self.out_dir
            self.parameters[""] = self.out_dir
        if self.config["umami"].get("convert_to_tfrecord", None) is not None:
            self.general.convert_to_tfrecord = self.config["umami"]["convert_to_tfrecord"]
        return self

    def get_file_name(self, option: str) -> Path | str:
        """Mimics the 'get_file_name()' function in PreprocessingConfig class in umami.

        Parameters
        ----------
        option : str
            The option specifying the desired file name:
            - 'resampled': Returns the current output file name.
            - 'resampled_scaled_shuffled': Returns a modified file name based on the
            original output file name with '_resampled_scaled_shuffled' appended.
            This option is used to create a new file name.

        Returns
        -------
        Path | str
            The resulting file name based on the specified 'option'.

        Raises
        ------
        ValueError
            If the option value is not supported
        """
        if option == "resampled":
            return self.out_fname
        elif option == "resampled_scaled_shuffled":
            return (
                str(self.out_fname.parent)
                + "/"
                + self.out_fname.stem
                + "_resampled_scaled_shuffled"
                + self.out_fname.suffix
            )
        else:
            raise ValueError(
                f"Option value {option} is not supported! "
                "Only resampled and resampled_scaled_shuffled are."
            )
