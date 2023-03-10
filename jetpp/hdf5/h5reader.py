import logging as log
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from jetpp.classes.cuts import Cuts
from jetpp.classes.variable_config import VariableConfig
from jetpp.hdf5.h5utils import get_dtype


@dataclass
class H5Reader:
    fname: Path
    batch_size: int = 100_000
    jets_name: str = "jets"
    as_full: bool = False

    @property
    def num_jets(self) -> int:
        with h5py.File(self.fname) as f:
            return len(f[self.jets_name])

    def get_attr(self, name, group=None):
        with h5py.File(self.fname) as f:
            obj = f[group] if group else f
            return obj.attrs[name]

    def empty(self, ds: h5py.Dataset, variables: list[str]) -> np.array:
        """Return a new empty numpy array with the given dtype."""
        return np.array(0, dtype=get_dtype(ds, variables, self.as_full))

    def read_chunk(self, ds: h5py.Dataset, array: np.array, low: int) -> np.array:
        high = min(low + self.batch_size, self.num_jets)
        shape = (high - low,) + ds.shape[1:]
        array.resize(shape, refcheck=False)
        ds.read_direct(array, np.s_[low:high])
        return array

    def remove_inf(self, data):
        keep_idx = np.full(len(data[self.jets_name]), True)
        for name, array in data.items():
            for var in array.dtype.names:
                isinf = np.isinf(array[var])
                keep_idx = keep_idx & ~isinf.any(axis=-1)
                if num_inf := isinf.sum():
                    log.warn(
                        f"{num_inf} inf values detected for variable {var} in"
                        f" {name} array. Removing the affected jets."
                    )
        for name, array in data.items():
            data[name] = array[keep_idx]
        return data

    def stream(
        self,
        variables: VariableConfig,
        num_jets: int,
        cuts: Cuts | None = None,
        shuffle: bool = True,
    ) -> np.array:
        rng = np.random.default_rng(42)

        if num_jets > self.num_jets:
            raise ValueError(
                f"{num_jets:,} jets requested but only {self.num_jets:,} present in {self.fname}"
            )

        total = 0
        jets_name = variables.jets_name
        with h5py.File(self.fname) as f:
            data = {}
            for name, var in variables.combined():
                data[name] = self.empty(f[name], var)

            # start loop
            indices = list(range(0, self.num_jets, self.batch_size))
            if shuffle:
                rng.shuffle(indices)

            # read arrays
            for low in indices:
                for name in variables:
                    data[name] = self.read_chunk(f[name], data[name], low)

                # apply selections
                if cuts:
                    idx = cuts(data[jets_name])[0]
                    for name, array in data.items():
                        data[name] = array[idx]

                # inf value check
                data = self.remove_inf(data)

                # check for completion
                total += len(data[jets_name])
                if total >= num_jets:
                    keep = num_jets - (total - len(data[jets_name]))
                    for name, array in data.items():
                        data[name] = array[:keep]
                    yield data
                    break

                yield data
