from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from ftag.hdf5 import H5Reader, H5Writer, join_structured_arrays
from ftag.vds import create_virtual_file

from upp.stages.hist import bin_jets

# from ftag_rw.weights.rw_utils import get_sample_weights
from upp.stages.reweight import Reweight


def _assign_weights(this_rw, bins, to_dump):
    this_weights = np.zeros(to_dump.shape, dtype=float)

    # This is SUPER slow - as we iterate each object, but its
    for i in range(this_weights.shape[0]):
        bin_idx = bins[:, i]

        cls = to_dump[i]

        thishist = this_rw["weights"][str(cls)][tuple(bin_idx)]
        this_weights[i] = thishist
    return this_weights


def get_sample_weights(batch, calculated_weights):
    """
    Parameters
    ----------
    batch : dict
        A dictionary of numpy arrays, where the keys are the group names
        and the values are the structured arrays of the data
    calculated_weights : dict
        A dictionary of the calculated weights, as returned by `calculate_weights`
    """
    sample_weights = {}
    for group, reweights in calculated_weights.items():
        if group not in sample_weights:
            sample_weights[group] = {}

        is_1d = len(batch[group].shape) == 1
        if is_1d:
            valid_indices = None
            to_dump = batch[group]
        else:
            valid_indices = np.nonzero(batch[group]["valid"])
            to_dump = batch[group][batch[group]["valid"]]

        for rwkey, rw in reweights.items():
            rw_vars = rw["rw_vars"]
            class_var = rw["class_var"]

            _, bins = bin_jets(to_dump[rw_vars], rw["bins"])
            # Enforce that bins are of shape (nvars, num_objects)
            if len(rw["bins"]) == 1:
                bins = np.expand_dims(bins, axis=0)

            try:
                # Note - I tried vectorising this but its not the bottleneck so
                # I'm leaving it as is
                this_weights = _assign_weights(rw, bins, to_dump[class_var])

            except Exception:
                print(f"Error in {group} {rwkey}")
                raise

            if valid_indices is not None:
                weights_out = np.zeros(batch[group].shape, dtype=float)
                weights_out[valid_indices] = this_weights
                sample_weights[group][rwkey] = weights_out
            else:
                sample_weights[group][rwkey] = this_weights

    sample_w_as_struct_arr = {}

    for group, reweights in sample_weights.items():
        dtype = [(key, arr.dtype) for key, arr in reweights.items()]

        structured_array = np.zeros(next(iter(reweights.values())).shape, dtype=dtype)
        for key in reweights:
            structured_array[key] = reweights[key]
        sample_w_as_struct_arr[group] = structured_array

    return sample_w_as_struct_arr


def do_merge_with_weights(
    reader_kwargs: dict[str, dict],
    output_file: str,
    weights: dict,
    num_jets_per_flavour: dict,
    variables: dict[str, list[str]] | None = None,
    skip_batches: int = 0,
    writer_id=0,
    limit_batches=5,
):
    """
    Takes a series of input files, merges them into a single final output file,
    while also adding in the relevent weights.
    """
    print("Writing weights to ", output_file, flush=True)
    reader = H5Reader(
        **reader_kwargs,
    )
    batch_size = reader.batch_size

    # reader = H5Reader(input_file)
    # num_jets = reader.num_jets if N == -1 else N
    writer: H5Writer = None
    additional_vars = {}

    for group, reweight in weights.items():
        additional_vars[group] = list(reweight.keys())

    dtypes = {k: v.descr for k, v in reader.dtypes(variables).items()}
    if variables is None:
        variables = {k: None for k in dtypes}

    for group, rw_output_names in additional_vars.items():
        for rw_name in rw_output_names:
            dtypes[group] += [(rw_name, "f4")]
    for group in dtypes:
        dtypes[group] = np.dtype(dtypes[group])

    num_jets_total = sum(num_jets_per_flavour.values())
    if limit_batches:
        num_batches = limit_batches
    else:
        num_batches = num_jets_total // batch_size + (1 if num_jets_total % batch_size != 0 else 0)

    print(f"Writer {writer_id} has batches: {num_batches} ", flush=True)

    start_time = time.time()

    for i, batch in enumerate(reader.stream(variables, skip_batches=skip_batches)):
        print(f"Writer {writer_id} Combined batch {i} has {len(batch['jets'])} jets", flush=True)
        all_sample_weights = get_sample_weights(batch, weights)
        to_write = {}
        for key in batch.keys():
            if key in all_sample_weights:
                to_write[key] = join_structured_arrays([batch[key], all_sample_weights[key]])
            else:
                to_write[key] = batch[key]
        if writer is None:
            shapes = {k: (None,) + v.shape[1:] for k, v in to_write.items()}
            writer = H5Writer(output_file, dtypes, shapes, shuffle=True, compression="gzip")
            # writer.copy_attrs(input_file)
        writer.write(to_write)
        cur_time = time.time()
        elapsed_time = cur_time - start_time
        exp_remaining_time = (num_batches - i - 1) * (elapsed_time / (i + 1))
        print(
            f"For writer {writer_id} Batch {i + 1} / {num_batches} written in {elapsed_time:.2f}s [{exp_remaining_time:.2f}s remaining]",
            flush=True,
        )
        if limit_batches and i + 1 >= limit_batches:
            print(f"Limit batches reached for writer {writer_id}, stopping early", flush=True)
            break
    print(f"Writer {writer_id} batches complete - closing writer", flush=True)
    writer.close()
    message = f"Writer {writer_id} finished writing {writer.num_written} jets in {time.time() - start_time:.2f}s"
    print(message, flush=True)


def start_mp(
    function,
    args_list,
    n_threads=1,
):
    if n_threads == 1:
        for args in args_list:
            function(*args)
    else:
        with Pool(n_threads) as pool:
            # apply the function using the pool
            pool.starmap(function, args_list)

    # Wait for all processes to finish
    print("All processes finished", flush=True)


class RWMerge:
    def __init__(self, config, outfile_idx_range=None):
        self.config = config
        self.rw_config = config.rw_config
        if outfile_idx_range is not None:
            assert isinstance(outfile_idx_range, tuple) and len(outfile_idx_range) == 2
        self.outfile_idx_range = outfile_idx_range

        self.hists_file = self.config.out_dir / "histograms.h5"
        assert self.hists_file.exists(), f"Histograms file not found: {self.hists_file}"
        self.organised_components_config = (
            Path(config.base_dir) / "split-components/organised-components.yaml"
        )
        assert (
            self.organised_components_config.exists()
        ), f"Organised components config file not found: {self.organised_components_config}"

    def run(self):
        weights = Reweight.load_weights_hdf5(self.hists_file)

        with open(self.organised_components_config) as f:
            organised_components = yaml.safe_load(f)
        # Get the number of jets per flavour
        files_by_flavour = organised_components["files"][self.config.split]
        num_jets_per_flavours = organised_components["num_jets"][self.config.split]
        all_files = sum(
            [files_by_flavour[f] for f in files_by_flavour],
            [],
        )
        total_jets = sum(num_jets_per_flavours.values())

        batch_size = 250_000
        reader_kwargs = {
            "fname": all_files,
            "batch_size": batch_size,
            "shuffle": False,
        }
        output_dir = self.config.out_dir / self.config.split
        output_dir.mkdir(parents=True, exist_ok=True)
        num_jets_per_file = self.config.num_jets_per_output_file or total_jets

        batches_per_file = num_jets_per_file // batch_size or 1
        num_batches = (
            total_jets // batch_size + (1 if total_jets % num_jets_per_file != 0 else 0)
        ) or 1

        variables = self.config.variables.combined() if self.config.split != "test" else None
        if variables and "flavour_label" not in variables:
            variables["jets"] += ["flavour_label"]
        args_list = []
        for i, bi in enumerate(range(0, num_batches, batches_per_file)):
            args_list.append(
                (
                    reader_kwargs,
                    output_dir / f"pp_output_train-full_{i}.h5",
                    weights,
                    num_jets_per_flavours,
                    variables,
                    bi,
                    i,
                    batches_per_file
                    if (bi + batches_per_file) < num_batches
                    else (num_batches - bi),
                )
            )
        print("Running with ", self.rw_config.merge_num_proc, "processes")
        start_mp(
            do_merge_with_weights,
            args_list,
            n_threads=self.rw_config.merge_num_proc,
        )
        create_virtual_file(
            str(output_dir / "*.h5"),
            self.config.out_dir / f"{self.config.out_fname}".replace(".h5", "_vds.h5"),
        )
