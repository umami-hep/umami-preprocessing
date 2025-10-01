from __future__ import annotations

import copy
import glob
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import yaml
from ftag import Cuts
from ftag.hdf5 import H5Reader
from ftag.hdf5.h5writer import H5Writer
from ftag.vds import create_virtual_file
from numpy.lib import recfunctions as rfn

from upp.classes.preprocessing_config import PreprocessingConfig

if TYPE_CHECKING:  # pragma: no cover
    pass


# TODO move these to some util place
def get_all_datasets(file: Path | str) -> dict[str, None]:
    """Return a dictionary with all the groups in the h5 file.

    Parameters
    ----------
    file : Path | str
        Path to the h5 file

    Returns
    -------
    dict[str, None]
        A dictionary with all the groups in the h5 file as keys and None as values,
        such that h5read.stream(all_groups) will return all the groups in the file.
    """
    with h5py.File(file, "r") as f:
        return {dset: None for dset in f if isinstance(f[dset], h5py.Dataset)}


def get_all_vars(file: Path | str) -> list[str]:
    """Return a list with all the variables in all datasets in the h5 file.

    Parameters
    ----------
    file : Path | str
        Path to the h5 file

    Returns
    -------
    list[str]
        A list with all the variables in all datasets in the h5 file.
    """
    with h5py.File(file, "r") as f:
        all_vars = []
        for dset in f:
            # For now, we ignore groups, we only want datasets
            if isinstance(f[dset], h5py.Group):
                continue
            dset_vars = list(f[dset].dtype.names)
            all_vars.extend(dset_vars)
        return all_vars


def get_all_fp_vars(file: Path | str) -> list[str]:
    """Return a list of all the variables in the file that should be kept at full precision.

    Returns a list of all the variables in the file that should be kept at full precision -
    this is done by finding variables that contain 'pt', 'energy', or 'mass' in their names.

    Parameters
    ----------
    file : Path | str
        Path to the h5 file

    Returns
    -------
    list[str]
        A list of all the variables in the file that should be kept at full precision.
    """
    all_vars = get_all_vars(file)
    # combine the values in this dict into a single list

    fp_vars = [
        v for v in all_vars if ("pt" in v.lower() or "energy" in v.lower() or "mass" in v.lower())
    ]

    return fp_vars


def parse_variables(variables: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    vars_lists = {}

    for group, group_vars in variables.items():
        vars_list = []
        if "inputs" in group_vars:
            vars_list += group_vars["inputs"]
        if "labels" in group_vars:
            vars_list += group_vars["labels"]
        vars_lists[group] = vars_list

    return vars_lists


class SplitContainers:
    # When running on the grid we need to limit the batch size to avoid memory issues
    MAX_BATCH_SIZE = 1_00_000

    def __init__(self, config_path):
        self.config_path = copy.deepcopy(config_path)
        self.config = PreprocessingConfig.from_file(self.config_path, "train", skip_checks=True)

    def split_file(
        self,
        input_file: Path,
        output_dir: Path,
        cuts_by_component: dict[str, Cuts],
        batch_size: int = 1_00_000,
        limit_num_batches: int | None = None,
        verbose_freq: int = 1,
        overwrite: bool = False,
        output_name=None,
        variables: dict[str, dict[str, list[str]]] | None = None,
        flavour_label_list: list[str] | None = None,
    ):
        if isinstance(input_file, str):
            input_file = Path(input_file)
        add_flavour_label = flavour_label_list is not None
        # All variables for test file
        all_variables = get_all_datasets(input_file)
        print("All variables: ", all_variables, flush=True)
        # Subset of variables for train/val files
        parsed_variables: dict[str, list[str]] | dict[str, None] = (
            parse_variables(variables) if variables is not None else all_variables
        )
        print("parsed variables: ", parsed_variables, flush=True)
        start = time.time()
        reader = H5Reader(input_file, batch_size=batch_size, shuffle=False)
        if output_name is None:
            output_name = input_file.name
        num_jets = reader.num_jets
        num_batches = num_jets // batch_size + (1 if num_jets % batch_size != 0 else 0)
        if num_jets == 0:
            print(f"File {input_file} has no jets. Skipping it", flush=True)
            return
        sample_components = []
        writers_by_sample_components = {}
        cuts_by_sample_components = {}
        print(f"Input file: {input_file} has {num_jets} jets", flush=True)
        print(f"Running {num_batches} batches of size {batch_size}", flush=True)
        print(f"Output directory: {output_dir}", flush=True)

        fp_vars = get_all_fp_vars(input_file)
        for split, component_cuts in cuts_by_component.items():
            sample_components.append(split)
            output_file = output_dir / f"{output_name}_{split}.h5"
            if output_file.exists() and not overwrite:
                print(f"At least 1 output file exists for {input_file}. Skipping it", flush=True)
                return

            writers_by_sample_components[split] = H5Writer.from_file(
                input_file,
                num_jets=None,
                dst=output_file,
                precision="half",
                full_precision_vars=fp_vars,
                shuffle=False,
                variables=all_variables if "test" in split else parsed_variables,
                compression="gzip",
                add_flavour_label=add_flavour_label,
            )
            cuts_by_sample_components[split] = component_cuts
            print(f"Creating writer for {split} saved to {output_file}", flush=True)

        if add_flavour_label:
            assert flavour_label_list is not None  # for mypy
            _flavour_label_by_component = {
                component: [
                    i for i, label in enumerate(flavour_label_list) if f"_{label}" in component
                ]
                for component in sample_components
            }

            assert all(
                len(_flavour_label_by_component[component]) == 1 for component in sample_components
            ), f"Each component must have exactly 1 flavour label not {_flavour_label_by_component}"
            flavour_label_by_component: dict[str, int] = {  # noqa: no-redef
                component: _flavour_label_by_component[component][0]
                for component in _flavour_label_by_component
            }

        for i, batch in enumerate(reader.stream(all_variables)):
            for sample_component in sample_components:
                writer = writers_by_sample_components[sample_component]
                cuts = cuts_by_sample_components[sample_component]
                sel_idx = cuts(batch["jets"]).idx
                sel_batch = {k: v[sel_idx] for k, v in batch.items()}
                if add_flavour_label:
                    this_flavour_label = flavour_label_by_component[sample_component]
                    tfl_arr = (
                        np.ones(sel_batch["jets"].shape[0], dtype=np.int32) * this_flavour_label
                    )

                    # I think this is only going to happen during tests, as the mock file has
                    # flavour_label in, but we wont
                    if "flavour_label" in sel_batch["jets"].dtype.names:
                        if i == 0:
                            print(
                                f"Warning: {sample_component} already has a flavour label. "
                                "We will overwrite it now.",
                                flush=True,
                            )
                        sel_batch["jets"]["flavour_label"] = tfl_arr
                    else:
                        # Get the
                        sel_batch["jets"] = rfn.append_fields(
                            sel_batch["jets"],
                            "flavour_label",
                            tfl_arr,
                            usemask=False,
                        )

                writer.write(sel_batch)

            if (i + 1) % verbose_freq == 0:
                message = (
                    f"File {input_file.name} batch {i + 1}/{num_batches} "
                    f"[{time.time() - start:.2f}s]"
                )
                print(message, flush=True)
            if limit_num_batches is not None and i > limit_num_batches:
                break

        num_written_by_sample_components = {
            sample_component: writer.num_written
            for sample_component, writer in writers_by_sample_components.items()
        }
        sum_written = sum(num_written_by_sample_components.values())
        message = "-" * 20
        message += f"\nFor file {input_file.name}:\nComponent Summary\n"
        message += "-" * 20
        message += "\n"
        for sample_component, writer in writers_by_sample_components.items():
            perc = 100 * (num_written_by_sample_components[sample_component] / sum_written)
            message += f"{sample_component}: {writer.num_written} jets ({perc:.2f}%)\n"

            writer.close()

        time_taken = time.time() - start
        message += "-" * 20 + "\n"
        message += "Summary\n"
        message += "-" * 20 + "\n"
        message += f"Total number of jets: {num_jets}\n"
        message += f"Total number of batches: {num_batches}\n"
        message += f"Total number of written jets: {sum_written}\n"
        message += f"Total time taken: {time_taken:.2f} seconds\n"
        message += "-" * 20
        print(message, flush=True)

    @contextmanager
    def _make_tmp_vds(self, files: list[str] | str | Path) -> Generator[Path, None, None]:
        with tempfile.TemporaryDirectory() as tmp:
            if not isinstance(files, list):
                yield Path(files)
                return
            tmp_dir = Path(tmp)
            # Create a sum link for each h5 file such that we ensure they all exist in the same
            # directory
            for file in files:
                file_path = Path(file)
                (tmp_dir / file_path.name).symlink_to(file_path.resolve())

            tmp_out_path = tmp_dir / "combined.h5"
            create_virtual_file(str(tmp_dir / "*.h5"), tmp_out_path, overwrite=True)
            h5vds = H5Reader(
                tmp_out_path,
            )
            print(
                f"Created combined virtual dataset with {h5vds.num_jets} jets at {tmp_out_path}",
                flush=True,
            )
            yield tmp_out_path

    def run(
        self,
        container=None,
        files=None,
        output_dir: Path | str | None = None,
    ):
        all_flavours = [f.name for f in self.config.components.flavours]

        containers_with_split_cuts = PreprocessingConfig.get_input_files_with_split_components(
            self.config_path
        )

        if container is not None:
            if container not in containers_with_split_cuts:
                raise ValueError(f"Container {container} not found in config.")
            containers_with_split_cuts = {container: containers_with_split_cuts[container]}

        print("All containers", containers_with_split_cuts.keys())
        if files is not None:
            assert container is not None, "Can only specify files if a container is specified"

        for container, cuts_by_component in containers_with_split_cuts.items():
            this_out_dir = (
                Path(self.config.base_dir) / "split-components" / container
                if output_dir is None
                else Path(".")
            )
            if files is None:
                this_files = Path(self.config.ntuple_dir) / container

                if this_files.is_dir():
                    this_files = [f for f in this_files.glob("*.h5")]
                elif "*" in str(this_files):
                    this_files = [Path(p) for p in glob.glob(str(this_files))]
            else:
                this_files = files
            # return
            # Split css into a list of files
            if isinstance(this_files, str):
                this_files = [Path(f) for f in this_files.split(",")]

            print(self.config.ntuple_dir, container, this_files, type(this_files), flush=True)
            print("The fiules are", this_files, flush=True)

            # Create a virtual dataset of all input files
            with self._make_tmp_vds(this_files) as input_file:
                self.split_file(
                    input_file=input_file,
                    output_dir=this_out_dir,
                    cuts_by_component=cuts_by_component,
                    batch_size=min(self.config.batch_size, self.MAX_BATCH_SIZE),
                    variables=self.config.config["variables"],
                    flavour_label_list=all_flavours,
                    output_name="output",
                )

    def create_meta_data(self):
        #
        output_dir = Path(self.config.base_dir) / "split-components"
        flavours = [f.name for f in self.config.components.flavours]

        print("THE FLAVOURS ARE", flavours, flush=True)
        files = {split: {f: [] for f in flavours} for split in ["train", "val", "test"]}
        for container in output_dir.glob("*"):
            if not container.is_dir():
                continue
            for split in ["train", "val", "test"]:
                for flavour in flavours:
                    file = list(container.glob(f"*{split}*_{flavour}*.h5"))
                    if len(file) == 0:
                        print("Could not find file for", container, split, flavour, flush=True)
                        continue

                    if len(file) > 1:
                        raise FileExistsError(
                            f"Found multiple files for {container} {split} {flavour}. "
                            f"Please check the output directory. : {file}"
                        )

                    files[split][flavour].append(str(file[0]))

        num_jets = {
            split: {flavour: H5Reader(files[split][flavour]).num_jets for flavour in files[split]}
            for split in files
        }
        metadata = {
            "files": files,
            "num_jets": num_jets,
        }

        output_file = output_dir / "organised-components.yaml"
        print(f"Writing metadata to {output_file}", flush=True)
        with open(output_file, "w") as f:
            yaml.dump(
                metadata,
                f,
                default_flow_style=False,
            )
