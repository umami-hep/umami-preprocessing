from __future__ import annotations

import logging as log

import h5py
import numpy as np
import yaml
from ftag.hdf5 import H5Reader

from upp.logger import ProgressBar


class Normalisation:
    def __init__(self, config):
        self.ppc = config
        self.components = config.components
        self.variables = config.variables
        self.jets_name = self.ppc.jets_name
        self.num_jets = config.num_jets_estimate_norm
        self.norm_fname = config.out_dir / config.config.get("norm_fname", "norm_dict.yaml")
        self.class_fname = config.out_dir / config.config.get("class_fname", "class_dict.yaml")

    @staticmethod
    def combine_mean_std(mean_A, mean_B, std_A, std_B, num_A, num_B):
        combined_mean = np.average([mean_A, mean_B], weights=[num_A, num_B])
        u_A = (mean_A - combined_mean) ** 2 + std_A**2
        u_B = (mean_B - combined_mean) ** 2 + std_B**2
        combined_std = np.sqrt((u_A * num_A + u_B * num_B) / (num_A + num_B))
        return float(combined_mean), float(combined_std)

    def get_norm_dict(self, batch):
        norm_dict = {k: {} for k in self.variables}
        for name, array in batch.items():
            if name != self.variables.jets_name:
                array = array[array["valid"]]
            for var in self.variables[name]["inputs"]:
                if var in ["valid"]:
                    continue
                mean = float(np.nanmean(array[var]))
                std = float(np.nanstd(array[var]))
                norm_dict[name][var] = {"mean": mean, "std": std}
        return norm_dict, len(batch[self.variables.jets_name])

    def combine_norm_dict(self, norm_A, norm_B, num_A, num_B):
        combined = {}
        for name in norm_A:
            dict_A = norm_A[name]
            dict_B = norm_B[name]
            combined[name] = {}

            assert dict_A.keys() == dict_B.keys()

            for var in dict_A:
                tf_A = dict_A[var]
                tf_B = dict_B[var]

                combined_mean, combined_std = self.combine_mean_std(
                    tf_A["mean"], tf_B["mean"], tf_A["std"], tf_B["std"], num_A, num_B
                )
                combined[name][var] = {"mean": combined_mean, "std": combined_std}

        return combined

    def get_class_dict(self, batch):
        ignore = ["VertexIndex", "ftagTruthParentBarcode", "barcode", "eventNumber", "jetFoldHash"]
        class_dict = {k: {} for k in self.variables}
        for name, array in batch.items():
            if name != self.variables.jets_name:
                array = array[array["valid"]]
            # separate case for flavour_label
            if name == self.variables.jets_name and "flavour_label" in array.dtype.names:
                counts = np.unique(array["flavour_label"], return_counts=True)
                class_dict[name]["flavour_label"] = counts
            for var in self.variables[name].get("labels", []):
                if not np.issubdtype(array[var].dtype, np.integer) or any(s in var for s in ignore):
                    continue
                counts = np.unique(array[var], return_counts=True)
                class_dict[name][var] = counts
        return class_dict

    @staticmethod
    def combine_class_dict(class_dict_A, class_dict_B):
        for name, var in class_dict_B.items():
            for v, stats in var.items():
                labels, counts = stats
                for i, label in enumerate(labels):
                    if len(class_dict_A[name][v][0]) != len(class_dict_A[name][v][1]):
                        raise ValueError(
                            "Class dict A has arrays of different lengths for the same"
                            " variable. This should not happen."
                        )
                    counts_A = dict(zip(*class_dict_A[name][v]))
                    counts[i] += counts_A.get(label, 0)
                var[v] = (labels, counts)
        return class_dict_B

    def write_norm_dict(self, norm_dict):
        for norms in norm_dict.values():
            for var, tf in norms.items():
                assert not np.isinf(tf["mean"]), f"{var} mean is not finite"
                assert not np.isinf(tf["std"]), f"{var} std is not finite"
                assert not np.isnan(tf["mean"]), f"{var} mean is nan"
                assert not np.isnan(tf["std"]), f"{var} std is nan"
                assert tf["std"] != 0, f"{var} std is 0"
        with open(self.norm_fname, "w") as file:
            yaml.dump(norm_dict, file, sort_keys=False)

    def write_class_dict(self, class_dict):
        for labels in class_dict.values():
            for v, (_, counts) in labels.items():
                weights = sum(counts) / counts
                labels[v] = np.around(weights / weights.min(), 2).tolist()
        with open(self.class_fname, "w") as file:
            yaml.dump(class_dict, file, sort_keys=False)

    def run(self):
        title = " Computing Normalisations "
        log.info(f"[bold green]{title:-^100}")

        # setup reader
        reader = H5Reader(
            self.ppc.out_fname, self.ppc.batch_size, precision="full", jets_name=self.jets_name
        )
        log.debug(f"Setup reader at: {self.ppc.out_fname}")

        norm_dict = None
        class_dict = None
        total = None
        vars = self.variables.combined()
        with h5py.File(reader.files[0]) as f:
            if "flavour_label" in f[self.jets_name].dtype.names:
                vars[self.jets_name].append("flavour_label")
        stream = reader.stream(vars, self.num_jets)

        with ProgressBar() as progress:
            task = progress.add_task(
                f"[green]Computing normalisations using {self.num_jets:,} jets...",
                total=self.num_jets,
            )

            for i, batch in enumerate(stream):
                this_norm_dict, num = self.get_norm_dict(batch)
                this_class_dict = self.get_class_dict(batch)
                if i == 0:
                    norm_dict = this_norm_dict
                    class_dict = this_class_dict
                    total = num
                else:
                    class_dict = self.combine_class_dict(class_dict, this_class_dict)
                    norm_dict = self.combine_norm_dict(norm_dict, this_norm_dict, total, num)
                    total += num

                progress.update(task, advance=len(batch[self.variables.jets_name]))

        log.info(f"[bold green]Finished computing normalisation params on {self.num_jets:,} jets!")
        self.write_norm_dict(norm_dict)
        self.write_class_dict(class_dict)
        log.info(f"[bold green]Saved norm dict to {self.norm_fname}")
        log.info(f"[bold green]Saved class dict to {self.class_fname}")
