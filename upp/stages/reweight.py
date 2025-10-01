from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ftag.hdf5 import H5Reader
from puma import Histogram, HistogramPlot

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.hist import bin_jets


class Reweight:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.rw_config = config.rw_config
        self.flavours = [f.name for f in config.components.flavours]
        assert (
            self.rw_config is not None
        ), "Reweighting configuration is not set in the preprocessing config"
        self.organised_components_config = (
            Path(config.base_dir) / "split-components/organised-components.yaml"
        )
        assert (
            self.organised_components_config.exists()
        ), f"Organised components config file not found: {self.organised_components_config}"

    @property
    def hists_path(self):
        return self.config.out_dir / "histograms.h5"

    @property
    def num_jets_estimate(self):
        return self.rw_config.num_jets_estimate or self.config.num_jets_estimate

    def get_input_readers(self):
        components_config = self.organised_components_config
        with open(components_config) as f:
            components = yaml.safe_load(f)
        files_by_flavour = components["files"]["train"]
        print("Files by flavour:", files_by_flavour)
        input_readers = {
            f: H5Reader(
                files_by_flavour[f],
                batch_size=self.config.batch_size,
            )
            for f in files_by_flavour
        }
        for f, r in input_readers.items():
            assert r.num_jets >= self.num_jets_estimate, (
                f"Requested {self.num_jets_estimate} jets per flavour, but found "
                f"{r.num_jets} jets in {f}."
            )
            print(
                f"Flavour {f} has {r.num_jets} jets, reading in batches of {self.config.batch_size}"
            )
        return list(input_readers.values())

    def calculate_weights(
        self,
    ):
        """Generate all the calculate_weights for the reweighting and returns them in a dict.

        This takes the form:
        {
            'group_name' : {
                'repr(reweight)' : {
                    'bins': np.ndarray, # The bins used for the histogram
                    'histograms' : {
                            label_0 : hists_for_label_0, # np.ndarray
                            label_1 : hists_for_label_1,
                            ...
                        }
                    }

        }

        """
        reweights = self.rw_config.reweights
        print(f"Calculating weights for {len(reweights)} reweights")
        readers = self.get_input_readers()
        for reader in readers:
            assert (
                reader.batch_size == readers[0].batch_size
            ), "All readers must have the same batch size"
        batch_size_per_file = readers[0].batch_size
        all_vars = {}
        existing_vars = {}
        with h5py.File(readers[0].fname[0], "r") as f:
            for group in f:
                if isinstance(f[group], h5py.Dataset):
                    existing_vars[group] = list(f[group].dtype.names)

        rw_groups = list(set([rw.group for rw in reweights]))
        print("Found rw groups : ", rw_groups)
        print("Batch size : ", batch_size_per_file)
        print("N per file : ", self.num_jets_estimate)

        # Get the variables we need to reweight
        for rw in reweights:
            rw_group = rw.group
            if rw_group not in all_vars:
                all_vars[rw_group] = []
            if rw.class_var is not None:
                all_vars[rw_group].append(rw.class_var)
            all_vars[rw_group].extend(rw.reweight_vars)
            if "valid" in existing_vars[rw_group]:
                all_vars[rw_group] += ["valid"]
        if "jets" not in all_vars:
            all_vars["jets"] = ["pt"]
        all_vars = {k: list(set(v)) for k, v in all_vars.items()}
        num_in_hists = {}
        all_histograms = {}
        print("Setting up streams with vars: ", all_vars, flush=True)
        reader_streams = [r.stream(all_vars, num_jets=self.num_jets_estimate) for r in readers]
        num_batches = self.num_jets_estimate // batch_size_per_file + (
            1 if self.num_jets_estimate % batch_size_per_file != 0 else 0
        )
        start_time = time.time()
        for i in range(num_batches):
            all_batches = [next(reader_streams[j]) for j in range(len(readers))]
            combined_batch = {}
            for batch in all_batches:
                for k, v in batch.items():
                    if k not in combined_batch:
                        combined_batch[k] = [v]
                    else:
                        combined_batch[k] += [v]
            combined_batch = {k: np.concatenate(v, axis=0) for k, v in combined_batch.items()}
            # Keep track of how many items we've used to generate our histograms
            for k, v in combined_batch.items():
                if k not in num_in_hists:
                    num_in_hists[k] = v.shape[0]
                else:
                    num_in_hists[k] += v.shape[0]

            for rw in reweights:
                rw_group = rw.group
                if rw_group not in combined_batch:
                    continue
                data = combined_batch[rw_group]
                # Iterate the RW vars, any which have a '.' in them, we
                # split by . to make group.var. We then check if the group is the global group,
                # and if so, we copy the relevent variable from the global group to this group

                if len(data.shape) != 1:
                    if "ftagTruthOriginLabel" in data.dtype.names:
                        data = data[data["ftagTruthOriginLabel"] != -1]
                    else:
                        assert "valid" in data.dtype.names
                        data = data[data["valid"]]
                classes = np.unique(data[rw.class_var]) if rw.class_var is not None else [None]

                for cls in classes:
                    mask = data[rw.class_var] == cls
                    hist, outbins = bin_jets(data[mask][rw.reweight_vars], rw.flat_bins)
                    if rw.class_var is not None:
                        cls = str(cls)
                    if rw_group not in all_histograms:
                        all_histograms[rw_group] = {}
                    if repr(rw) not in all_histograms[rw_group]:
                        all_histograms[rw_group][repr(rw)] = {
                            "bins": rw.flat_bins,
                            "histograms": {},
                        }
                    if cls not in all_histograms[rw_group][repr(rw)]["histograms"]:
                        all_histograms[rw_group][repr(rw)]["histograms"][cls] = hist.copy()
                    else:
                        all_histograms[rw_group][repr(rw)]["histograms"][cls] += hist
            perc = 100 * (i + 1) / num_batches
            time_taken = time.time() - start_time
            print(
                f"Processed {i + 1}/{num_batches} batches ({perc:.2f}%) in {time_taken:.2f}s",
                flush=True,
            )

        # Define for each RW what the target histogram should be. This is either a single
        # flavour, mean of multiple/all flavours
        all_targets = {}
        for rw in reweights:
            rw_group = rw.group
            if rw_group not in all_histograms:
                raise ValueError(f"Group {rw_group} not found in histograms... What?")

            if rw_group not in all_targets:
                all_targets[rw_group] = {}

            rw_rep = repr(rw)

            target = None

            if isinstance(rw.class_target, int):
                target = all_histograms[rw_group][rw_rep]["histograms"][str(rw.class_target)]
            elif isinstance(rw.class_target, str) and rw.class_target == "mean":
                for hist in all_histograms[rw_group][rw_rep]["histograms"].values():
                    if target is None:
                        target = hist.copy()
                    else:
                        target += hist
                target /= len(all_histograms[rw_group][rw_rep]["histograms"])
            elif isinstance(rw.class_target, str) and rw.class_target == "min":
                for hist in all_histograms[rw_group][rw_rep]["histograms"].values():
                    target = np.minimum(target, hist) if target is not None else hist.copy()
            elif isinstance(rw.class_target, str) and rw.class_target == "max":
                for hist in all_histograms[rw_group][rw_rep]["histograms"].values():
                    target = np.maximum(target, hist) if target is not None else hist.copy()
            elif isinstance(rw.class_target, str) and rw.class_target == "uniform":
                target = np.ones_like(all_histograms[rw_group][rw_rep]["histograms"][str(0)])
            elif isinstance(rw.class_target, list | tuple):
                for cls, hist in all_histograms[rw_group][rw_rep]["histograms"].items():
                    cast_cls_target = tuple(map(str, rw.class_target))
                    if cls in cast_cls_target:
                        if target is None:
                            target = hist.copy()
                        else:
                            target += hist
                target /= len(rw.class_target)
            else:
                raise ValueError("Unknown class_target type")

            if np.any(target == 0):
                num_zeros = np.sum(target == 0)
                print(
                    f"Target histogram has {num_zeros} bins with zero entries out of total"
                    f" {target.shape} : {rw!r}"
                )
            if np.any(target < 0):
                raise ValueError(f"Target histogram has bins with negative entries : {rw!r}")
            if np.any(np.isnan(target)):
                raise ValueError(f"Target histogram has bins with NaN entries : {rw!r}")

            # Apply the target histogram function
            if rw.target_hist_func is not None:
                target = rw.target_hist_func(target)

            all_targets[rw_group][rw_rep] = target

        output_weights = {}
        for rw in reweights:
            rw_group = rw.group
            rw_rep = repr(rw)
            if rw_group not in output_weights:
                output_weights[rw_group] = {}
            if rw_rep not in output_weights[rw_group]:
                output_weights[rw_group][rw_rep] = {}
            output_weights[rw_group][rw_rep] = {
                "weights": {},
                "bins": all_histograms[rw_group][rw_rep]["bins"],
                "rw_vars": rw.reweight_vars,
                "class_var": rw.class_var,
            }
            idx_below_min = None
            for cls, hist in all_histograms[rw_group][rw_rep]["histograms"].items():
                this_idx_below_min = hist == 0  # | (all_targets[rw_group][rw_rep] == 0)
                output_weights[rw_group][rw_rep]["weights"][cls] = np.where(
                    hist > 0, all_targets[rw_group][rw_rep] / hist, 0
                )
                if idx_below_min is None:
                    idx_below_min = this_idx_below_min
                else:
                    idx_below_min |= this_idx_below_min
            # If we have any bins where we have 0 of a given flavour, we set all the
            # weights to 0
            if np.any(idx_below_min):
                for cls in all_histograms[rw_group][rw_rep]["histograms"]:
                    output_weights[rw_group][rw_rep]["weights"][cls][idx_below_min] = 0

        return output_weights

    def plot_rw_histograms(
        self,
        histograms_path: Path,
    ):
        histograms = h5py.File(histograms_path, "r")
        plot_dir = histograms_path.parent / "plots"
        if not plot_dir.exists():
            plot_dir.mkdir(parents=True, exist_ok=True)

        # Now, iterate each re-weighting
        for group in histograms:
            for rw in histograms[group]:
                rw_dir = plot_dir / group / rw
                if not rw_dir.exists():
                    rw_dir.mkdir(parents=True, exist_ok=True)
                rw_vars = [v.decode("utf-8") for v in histograms[group][rw]["rw_vars"][:]]
                bins = histograms[group][rw]["bins"]
                class_var = histograms[group][rw]["class_var"][0].decode("utf-8")
                if class_var == "flavour_label":
                    class_mapping = {self.flavours.index(f): f for f in self.flavours}
                else:
                    class_mapping = {}
                weights = histograms[group][rw]["weights"]

                # First, make a 1d histogram of each class
                # fig = plt.figure(figsize=(10, 6))
                all_weights = np.concatenate([w[:] for w in weights.values()])
                wmin, wmax = all_weights.min(), all_weights.max()
                bins_1d = np.linspace(wmin, wmax, 100)

                plot = HistogramPlot(
                    ylabel="Frequency",
                    xlabel="Weight value",
                    atlas_second_tag=f"Distribution of weights for {group} : {rw}",
                    y_scale=1.5,
                    logy=True,
                    figsize=(10, 6),
                )
                for cls in weights:
                    w = weights[cls][:]
                    w_flat = w.flatten()
                    histo = Histogram(
                        values=w_flat,
                        label=class_mapping.get(int(cls), cls),
                        bins=bins_1d,
                    )
                    plot.add(histo)
                plot.draw()
                plot.savefig(rw_dir / "weights_distribution.png")
                for var1, var2 in combinations(rw_vars, 2):
                    print(f"Plotting 2D histogram for {var1} vs {var2} in {rw}")
                    for cls in weights:
                        fig, ax = plt.subplots(figsize=(10, 6))

                        var1_bin_idx = rw_vars.index(var1)
                        var2_bin_idx = rw_vars.index(var2)

                        def sanitize_edges(edges, eps=1e-3):
                            """Replace -inf and inf in bin edges with finite values."""
                            edges = np.array(edges, dtype=np.float64)
                            if np.isneginf(edges[0]):
                                edges[0] = edges[1] - eps * abs(edges[1])
                            if np.isposinf(edges[-1]):
                                edges[-1] = edges[-2] + eps * abs(edges[-2])
                            return edges

                        var1_bin_edges = sanitize_edges(bins[f"bin_{var1_bin_idx}"][:])
                        var2_bin_edges = sanitize_edges(bins[f"bin_{var2_bin_idx}"][:])

                        w = weights[cls]
                        w = np.moveaxis(w, [var1_bin_idx, var2_bin_idx], [0, 1])

                        axes_to_sum = tuple(
                            range(2, w.ndim)
                        )  # Sum over all axes except the first 2
                        w_2d = w if w.ndim == 2 else w.sum(axis=axes_to_sum)

                        assert w_2d.shape[0] == len(var1_bin_edges) - 1, "Mismatch in x bin edges"
                        assert w_2d.shape[1] == len(var2_bin_edges) - 1, "Mismatch in y bin edges"

                        w_2d = w_2d.astype(np.float32)

                        # Plot using pcolormesh
                        pcm = ax.pcolormesh(
                            var1_bin_edges,
                            var2_bin_edges,
                            w_2d.T,  # Transpose for correct orientation
                            shading="auto",  # Avoids shape mismatch
                        )
                        fig.colorbar(pcm, ax=ax, label="Weights")

                        ax.set_xlabel(var1)
                        ax.set_ylabel(var2)
                        ax.set_title(f"Reweighting {rw} for {cls} ({group})")

                        plt.tight_layout()
                        plt.savefig(rw_dir / f"{cls}_{var1}_vs_{var2}.png")
                        plt.close(fig)

    def run(self):
        """Run the reweighting stage."""
        print("Running reweighting stage...")

        histograms = self.calculate_weights()
        print("Completed histogram calculation, saving to file...")
        self.hists_path.parent.mkdir(parents=True, exist_ok=True)
        Reweight.save_weights_hdf5(histograms, self.hists_path)
        print("Histograms saved to:", self.hists_path)
        print("Making plots...")
        self.plot_rw_histograms(self.hists_path)

    @staticmethod
    def save_weights_hdf5(weights_dict, filename):
        """Save the weights to an HDF5 file.

        It takes the following structure:
        {
            'group_name' : {
                'repr(reweight)' : {
                    'bins': np.ndarray, # The bins used for the histogram
                    'histograms' : {
                            label_0 : hists_for_label_0, # np.ndarray
                            label_1 : hists_for_label_1,
                            ...
                        }
                    }
                ...
        }
        Such that they can be loaded by `load_weights_hdf5`
        """
        filename.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(filename, "w") as f:
            for group, data in weights_dict.items():
                group_obj = f.create_group(group)
                for reweight_name, reweight_data in data.items():
                    reweight_group = group_obj.create_group(reweight_name)

                    # Create a group for bins, as it's a list of arrays
                    bins_group = reweight_group.create_group("bins")
                    for i, bin_array in enumerate(reweight_data["bins"]):
                        bins_group.create_dataset(f"bin_{i}", data=bin_array)

                    reweight_group.create_dataset(
                        "rw_vars",
                        data=np.array(reweight_data["rw_vars"], dtype=h5py.special_dtype(vlen=str)),
                    )
                    reweight_group.create_dataset(
                        "class_var",
                        data=np.array(
                            [reweight_data["class_var"]], dtype=h5py.special_dtype(vlen=str)
                        ),
                    )

                    # Save histograms
                    hist_group = reweight_group.create_group("weights")
                    for label, hist in reweight_data["weights"].items():
                        hist_group.create_dataset(f"{label}", data=hist)

    @staticmethod
    def load_weights_hdf5(filename):
        """Load the weights from an HDF5 file, see `save_weights_hdf5` for the structure."""
        weights_dict = {}
        with h5py.File(filename, "r") as f:
            # Iterate through the groups in the file (top-level groups represent 'group_name')
            for group in f:
                weights_dict[group] = {}
                group_obj = f[group]
                # For each group, iterate through the reweight names
                for reweight_name in group_obj:
                    reweight_group = group_obj[reweight_name]

                    # Load the bins, which is now a list of arrays
                    bins_group = reweight_group["bins"]
                    bins = [bins_group[f"bin_{i}"][:] for i in range(len(bins_group))]

                    reweight_vars = [var.decode("utf-8") for var in reweight_group["rw_vars"][:]]
                    class_var = next(var.decode("utf-8") for var in reweight_group["class_var"][:])
                    # Load the histograms
                    histograms = {}
                    hist_group = reweight_group["weights"]
                    for label in hist_group:
                        histograms[label] = hist_group[label][:]

                    # Reconstruct the structure
                    weights_dict[group][reweight_name] = {
                        "bins": bins,
                        "weights": histograms,
                        "rw_vars": reweight_vars,
                        "class_var": class_var,
                    }
        return weights_dict
