"""
Unit-tests (no disk IO) for upp.stages.merging.Merging.

We monkey-patch upp.stages.merging.H5Writer with an in-memory stub so that
_open_writer / write_chunk / write_components can be exercised without
creating real HDF5 files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
from ftag import Flavours

import upp.stages.merging as merging_mod
from upp.classes.preprocessing_config import PreprocessingConfig


# Stub H5Writer that just records calls, does NO real IO
class MemWriter:
    def __init__(self, dst, dtypes, shapes, jets_name, **_):
        self.dst = Path(dst)
        self.dtypes = dtypes
        self.shapes = shapes
        self.num_jets = next(iter(shapes.values()))[0]
        self.jets_name = jets_name
        self.num_written = 0
        self.attrs = {}

    # API that Merging calls
    def write(self, data: dict):
        self.num_written += len(data[self.jets_name])

    def close(self):
        assert self.num_written == self.num_jets

    def add_attr(self, name, value, *_):
        self.attrs[name] = value


# Minimal helpers to fabricate Components / Variables for unit tests
def _jets_struct(n):
    """Return a (n,) structured array with two float vars."""
    dtype = np.dtype([("pt", "f4"), ("eta", "f4")])
    rng = np.random.default_rng(42)
    arr = np.empty(n, dtype=dtype)
    arr["pt"] = rng.random(n)
    arr["eta"] = rng.random(n)
    return arr


class DummyComponent(SimpleNamespace):
    """Dummy component for testing.

    Simulates the subset of the real Component interface that Merging uses:
    - flavour
    - stream  (generator of batches)
    - num_jets
    - complete flag
    """

    def __init__(self, flavour, jet_batches):
        super().__init__()
        self.flavour = Flavours[flavour]
        self._batches = list(jet_batches)
        self.num_jets = sum(len(b["jets"]) for b in self._batches)
        self.complete = False
        self.out_path = Path("/dev/null")

    def setup_reader(self, *_args, **_kw):
        pass

    @property
    def stream(self):
        def _gen():
            yield from self._batches

        return _gen()


def _minimal_merging(monkeypatch, jets_per_file=10) -> merging_mod.Merging:
    """Create modded Merging version.

    Return a fully-initialised Merging instance whose writer is the MemWriter
    and whose config / variables are minimal SimpleNamespace objects.
    """
    # Patch H5Writer
    monkeypatch.setattr(merging_mod, "H5Writer", MemWriter)

    # Minimal Variables object
    variables = SimpleNamespace(
        variables=["jets"],
        selectors={},
        combined=lambda: {"jets": None},
        keys=lambda: ["jets"],
    )

    # Setup cfg & components
    cfg = SimpleNamespace(
        components=SimpleNamespace(flavours=[Flavours["bjets"]]),
        variables=variables,
        batch_size=100,
        jets_name="jets",
        num_jets_per_output_file=jets_per_file,
        file_tag="split",
        out_fname=Path("/tmp/merged.h5"),
        git_hash="deadbeef",
        config={},
        is_test=False,
        merge_test_samples=False,
    )

    return merging_mod.Merging(cast(PreprocessingConfig, cfg))


# Actual tests
def test_add_jet_flavour_label(monkeypatch):
    """A flavour label is added exactly once and matches the flavour list."""
    merge = _minimal_merging(monkeypatch, jets_per_file=7)

    jets = _jets_struct(5)
    comp = SimpleNamespace(flavour=Flavours["bjets"])
    tagged = merge.add_jet_flavour_label(jets, comp)

    assert "flavour_label" in tagged.dtype.names
    assert np.all(tagged["flavour_label"] == 0)

    # Calling again must not duplicate the column
    same = merge.add_jet_flavour_label(tagged, comp)
    assert same.dtype == tagged.dtype


def test_open_writer_names_and_shapes(monkeypatch):
    """_open_writer should create a MemWriter with the correct dst name and shapes."""
    merge = _minimal_merging(monkeypatch, jets_per_file=7)

    # Minimal shapes / dtypes upfront
    merge.base_shapes = {"jets": (42,)}
    merge.dtypes = {"jets": _jets_struct(1).dtype}

    merge._open_writer(
        sample=None,
        jets_in_file=7,
        file_idx=0,
        components=SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[]),
    )

    writer = merge.writer
    assert isinstance(writer, MemWriter)
    assert writer.num_jets == 7
    assert writer.dst.name.startswith("merged_split_000")
    assert "flavour_label" in writer.attrs


def test_write_chunk_splits(monkeypatch):
    """Test split writing of write_chunk.

    With num_jets_per_output_file=5 and a batch of 8 jets, write_chunk must
    create two MemWriter instances and write all jets.
    """
    merge = _minimal_merging(monkeypatch, jets_per_file=5)

    # Prepare internal bookkeeping identical to write_components()
    jets1 = {"jets": _jets_struct(8)}
    comp = DummyComponent("bjets", [jets1])
    comps = [comp]

    # Fake dtypes / shapes and open first writer
    merge.dtypes = {"jets": jets1["jets"].dtype}
    merge.base_shapes = {"jets": (8,)}
    merge.total_jets = 8
    merge._file_idx = 0
    merge.jets_written = 0
    merge.current_components = SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[])
    merge._sample = None

    merge._open_writer(None, 5, 0, merge.current_components)

    # Run write_chunk once
    n_written = merge.write_chunk(comps)

    # Assertions
    assert n_written == 8

    # First writer closed & second opened
    assert merge._file_idx == 1
    assert merge.writer.num_written == 3


def test_write_chunk_rollover(monkeypatch):
    """Check the rollover behaviour.

    When the current MemWriter is already full (capacity_left == 0)
    `write_chunk` must

    1. close that writer,
    2. open a fresh one (file_idx increments),
    3. write the incoming batch into the new file,
    4. update jets_written.
    """
    # Create a merger with 5 jets / file
    merge = _minimal_merging(monkeypatch, jets_per_file=5)

    # Incoming batch: 3 jets
    batch = {"jets": _jets_struct(3)}
    comp = DummyComponent("bjets", [batch])
    comps = [comp]

    # Fake the bookkeeping exactly as write_components() would
    merge.dtypes = {"jets": batch["jets"].dtype}
    merge.base_shapes = {"jets": (8,)}
    merge.total_jets = 8
    merge.jets_written = 5
    merge._file_idx = 0
    merge._sample = None
    merge.current_components = SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[])

    # Open the first writer with capacity 5 and mark it as "full"
    merge._open_writer(None, 5, 0, merge.current_components)
    merge.writer.num_written = 5

    # Call write_chunk
    n = merge.write_chunk(comps)

    assert n == 3
    assert merge._file_idx == 1
    assert merge.writer.num_written == 3
    assert merge.jets_written == 8


def test_write_chunk_returns_zero_when_no_space_left(monkeypatch):
    """Check early exit."""
    merge = _minimal_merging(monkeypatch, jets_per_file=4)

    # One dummy component that would supply more jets
    jets_batch = {"jets": _jets_struct(3)}
    comp = DummyComponent("bjets", [jets_batch])
    comps = [comp]

    # Mimic state after everything has already been written
    merge.total_jets = 4
    merge.jets_written = 4
    merge._file_idx = 0
    merge.current_components = SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[])
    merge._sample = None

    # We still need valid dtypes / shapes for _open_writer
    merge.dtypes = {"jets": jets_batch["jets"].dtype}
    merge.base_shapes = {"jets": (4,)}

    # Open a writer that is already full (num_written == num_jets)
    merge._open_writer(None, 4, 0, merge.current_components)
    merge.writer.num_written = merge.writer.num_jets

    n_written = merge.write_chunk(comps)

    # it must early-return with 0 and NOT raise or open new writers
    assert n_written == 0

    # _file_idx is incremented once, but _open_writer was not called again
    assert merge._file_idx == 1
    assert merge.writer.num_written == merge.writer.num_jets
