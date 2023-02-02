from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from jetpp.classes.variable_config import VariableConfig
from jetpp.hdf5.h5utils import get_dtype


@dataclass
class H5Writer:
    fname: Path
    variables: VariableConfig
    num_jets: int
    compression: str = "lzf"
    _num_written: int = 0
    rng = np.random.default_rng(42)

    def create_ds(self, name: str, variables: list[str], add_flavour_label: bool = False) -> None:
        with h5py.File(self.src_fname) as srcfile:
            ds = srcfile[name]
            dtype = get_dtype(ds, variables)
            if add_flavour_label:
                dtype = np.dtype(dtype.descr + [("flavour_label", "i4")])
            num_tracks = ds.shape[1:]
            shape = (self.num_jets,) + num_tracks
            # optimal jet chunking is around 100 jets
            chunks = (100,) + num_tracks if num_tracks else None

        # note: enabling the hd5 shuffle filter doesn't improve anything
        self.file.create_dataset(
            name, dtype=dtype, shape=shape, compression=self.compression, chunks=chunks
        )

    def setup_file(self, src_fname, add_flavour_label=None) -> None:
        self.fname.parent.mkdir(parents=True, exist_ok=True)
        self.src_fname = src_fname
        self.file = h5py.File(self.fname, "w")
        for name, var in self.variables.combined():
            self.create_ds(name, var, name == add_flavour_label)

    def close(self) -> None:
        with h5py.File(self.fname) as f:
            out_len = len(f[self.variables.jets_name])
        if self._num_written != out_len:
            raise ValueError(
                f"Attemped to close a file {self.fname} when only {self._num_written:,} out of"
                f" {out_len:,} jets have been written"
            )
        self.file.close()

    def write(self, data: dict[str, np.array], shuffle=True) -> None:
        idx = np.arange(len(data[self.variables.jets_name]))
        if shuffle:
            self.rng.shuffle(idx)
            for name, array in data.items():
                data[name] = array[idx]

        low = self._num_written
        high = low + len(idx)
        for n in self.variables:
            self.file[n][low:high] = data[n]
        self._num_written += len(idx)

    def get_attr(self, name, group=None):
        with h5py.File(self.fname) as f:
            obj = f[group] if group else f
            return obj.attrs[name]

    def add_attr(self, name, data, group=None):
        obj = self.file[group] if group else self.file
        obj.attrs.create(name, data)
